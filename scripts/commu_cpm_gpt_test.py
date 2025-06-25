import torch
from transformers import AutoTokenizer, GPT2Model, AutoModelWithLMHead, GPT2Config, T5Config, AutoModelForCausalLM
import json
import torch.nn.functional as F
from torch.optim import AdamW
import os
import random

# --- 1. 全局配置 ---
# 根据你的环境修改路径
GPT2_MODEL_PATH = "/ubsnhome/23063003r/refgame_project/models/gpt2"
CPM_MODEL_PATH = "/ubsnhome/23063003r/refgame_project/models/CPM-Generate"
DATA_FILE = "/ubsnhome/23063003r/refgame_project/data/bilingual_game_data.json" # 需要新的数据格式
OUTPUT_DIR = "/ubsnhome/23063003r/refgame_project/output/"

# 学习参数
LEARNING_RATE_GPT2 = 5e-6
LEARNING_RATE_CPM = 2e-6
REWARD_CORRECT = 0.1 # 猜对时, 损失降低 10%
PENALTY_WRONG = 1.1  # 猜错时, 损失增加 10%
ALIGNMENT_LOSS_WEIGHT = 0.5 # 对齐损失在总损失中的权重

# 创建输出目录
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)
    print(f"创建输出目录: {OUTPUT_DIR}")

# --- 2. Agent 封装 【关键设计】---
# 使用一个简单的类来管理每个 Agent 的资产，使代码更清晰
class Agent:
    def __init__(self, model_name, model, tokenizer, optimizer):
        self.name = model_name
        self.model = model
        self.tokenizer = tokenizer
        self.optimizer = optimizer
        self.error_log = [] # 用于记录“我说的话 -> 对方猜错了”

    def get_semantic_embedding(self, text):
        """获取单个句子的语义嵌入 (取[CLS]或第一个token的输出)"""
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True).to(self.model.device)
        outputs = self.model(**inputs)
        # 兼容 GPT2Model 和 CPM-Generate 的输出结构
        if hasattr(outputs, 'last_hidden_state'):
            embedding = outputs.last_hidden_state[:, 0, :]
        else: # 适配 CPM-Generate (可能在 model.transformer.h 中)
            embedding = outputs.hidden_states[-1][:, 0, :]
        return embedding

    def get_candidate_embeddings(self, sentences):
        """获取多个候选句子的嵌入"""
        # 注意：为了获得可比较的嵌入，最好逐个处理句子而不是批处理后平均
        # 因为每个句子的长度和token化结果不同
        embeddings = [self.get_semantic_embedding(s) for s in sentences]
        return torch.cat(embeddings, dim=0)

# --- 3. 初始化两个 Agent ---

# 确定设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

# 加载 GPT-2
print("--- 初始化英文 Agent (GPT-2) ---")
gpt2_tokenizer = AutoTokenizer.from_pretrained(GPT2_MODEL_PATH)
# GPT-2 的原生 Tokenizer 没有 pad token
if gpt2_tokenizer.pad_token is None:
    gpt2_tokenizer.add_special_tokens({'pad_token': '[PAD]'})
gpt2_model = GPT2Model.from_pretrained(GPT2_MODEL_PATH).to(device)
gpt2_model.resize_token_embeddings(len(gpt2_tokenizer)) # 适配 pad token
gpt2_optimizer = AdamW(gpt2_model.parameters(), lr=LEARNING_RATE_GPT2)
agent_gpt2 = Agent("GPT-2", gpt2_model, gpt2_tokenizer, gpt2_optimizer)
agent_gpt2.model.train()

# 加载 CPM
print("--- 初始化中文 Agent (CPM) ---")
# AutoModelWithLMHead 已被弃用，建议使用 AutoModelForCausalLM
cpm_tokenizer = AutoTokenizer.from_pretrained(CPM_MODEL_PATH)
cpm_model = AutoModelForCausalLM.from_pretrained(CPM_MODEL_PATH).to(device)
cpm_optimizer = AdamW(cpm_model.parameters(), lr=LEARNING_RATE_CPM)
# CPM 的核心模型在 .transformer 属性里
agent_cpm = Agent("CPM", cpm_model.transformer, cpm_tokenizer, cpm_optimizer)
agent_cpm.model.train() # 将内部的 transformer 设为训练模式

# 【重要】让 GPT-2 Tokenizer 能够处理中文字符
# 这是你原始代码中非常好的一个实践，我们保留它
print("--- 扩展 GPT-2 Tokenizer 以支持中文 ---")
# 假设数据文件中包含了所有可能出现的中文概念
try:
    with open(DATA_FILE, 'r', encoding='utf-8') as f:
        full_game_data = json.load(f)
    all_chinese_concepts = set()
    for entry in full_game_data:
        all_chinese_concepts.add(entry['concept_chinese'])
        all_chinese_concepts.update(entry['distractors_chinese'])

    new_tokens = list(all_chinese_concepts - set(agent_gpt2.tokenizer.get_vocab().keys()))
    agent_gpt2.tokenizer.add_tokens(new_tokens)
    agent_gpt2.model.resize_token_embeddings(len(agent_gpt2.tokenizer))
    print(f"GPT-2 tokenizer 已扩展，新词汇表大小: {len(agent_gpt2.tokenizer)}")
except Exception as e:
    print(f"❌ 加载数据或扩展Tokenizer失败: {e}")
    print("将使用默认的少量中文字符进行扩展。")
    agent_gpt2.tokenizer.add_tokens(list("飞机老师橘子学校猫狗太阳书"))
    agent_gpt2.model.resize_token_embeddings(len(agent_gpt2.tokenizer))


# --- 4. 准备数据 ---
# **请注意：** 数据格式需要调整以支持双向游戏
# 建议的 `bilingual_game_data.json` 格式:
# [
#   {
#     "concept_english": "airplane",
#     "description_english": "It flies in the sky and carries people.",
#     "distractors_english": ["teacher", "orange", "school"],
#     "concept_chinese": "飞机",
#     "description_chinese": "它在天上飞，用来载人。",
#     "distractors_chinese": ["老师", "橘子", "学校"]
#   }, ...
# ]
try:
    with open(DATA_FILE, 'r', encoding='utf-8') as f:
        all_game_rounds_data = json.load(f)
    print(f"✅ 成功加载 {len(all_game_rounds_data)} 轮游戏数据。")
except Exception as e:
    print(f"❌ 错误: 无法加载或解析数据文件 '{DATA_FILE}'. {e}")
    exit()

# --- 5. 主训练循环 ---
total_rounds = len(all_game_rounds_data)
training_log = []
total_loss_sum = 0.0
correct_predictions_count = 0

print(f"\n--- 开始进行 {total_rounds * 2} 轮交互式游戏 ---")

for i, game_data in enumerate(all_game_rounds_data):
    # 每个数据点可以玩两轮，角色互换
    for turn in range(2):
        round_idx = i * 2 + turn + 1

        # 【关键设计】角色分配
        if turn == 0:
            # 轮次 1: GPT-2 是 Speaker, CPM 是 Listener
            speaker_agent = agent_gpt2
            listener_agent = agent_cpm
            description = game_data['description_english']
            correct_option = game_data['concept_chinese']
            distractors = game_data['distractors_chinese']
        else:
            # 轮次 2: CPM 是 Speaker, GPT-2 是 Listener
            speaker_agent = agent_cpm
            listener_agent = agent_gpt2
            description = game_data['description_chinese']
            correct_option = game_data['concept_english']
            distractors = game_data['distractors_english']

        print(f"\n--- 游戏回合 {round_idx} ---")
        print(f"🗣️ Speaker ({speaker_agent.name}) 描述: \"{description}\"")

        # 准备选项
        options = distractors + [correct_option]
        random.shuffle(options) # 随机打乱选项顺序
        correct_index = options.index(correct_option)
        print(f"👂 Listener ({listener_agent.name}) 选项: {options}")
        print(f"✅ 正确选项索引: {correct_index}")

        # --- 交互与计算 ---

        # 1. Listener 处理描述和选项，得到嵌入
        # Listener 需要处理的是“外语”
        desc_embedding = listener_agent.get_semantic_embedding(description)
        option_embeddings = listener_agent.get_candidate_embeddings(options)

        # 2. Listener 进行猜测
        similarities = F.cosine_similarity(desc_embedding, option_embeddings, dim=1)
        predicted_index = torch.argmax(similarities).item()

        # --- 反馈与学习 ---
        is_correct = (predicted_index == correct_index)
        correct_index_tensor = torch.tensor([correct_index], device=device)

        # 3. 计算 Listener 的选择损失 (cross_entropy)
        listener_loss = F.cross_entropy(similarities.unsqueeze(0), correct_index_tensor)

        # 根据对错调整 Listener 损失
        if is_correct:
            final_listener_loss = listener_loss * (1 - REWARD_CORRECT)
            outcome_message = f"🎉 {listener_agent.name} 猜对了!"
            correct_predictions_count += 1
        else:
            final_listener_loss = listener_loss * PENALTY_WRONG
            outcome_message = f"💔 {listener_agent.name} 猜错了!"
            # 【关键设计】记录错误，用于 Speaker 的自我改进
            speaker_agent.error_log.append({
                "description_given": description,
                "options": options,
                "listener_chose": options[predicted_index],
                "correct_answer": correct_option,
                "round": round_idx
            })

        # 4. 【关键设计】计算 Speaker 的对齐损失 (Alignment Loss)
        # 目标：让 Speaker 产生的描述的嵌入，与其“正确翻译”的嵌入对齐
        speaker_desc_embedding = speaker_agent.get_semantic_embedding(description)
        # 正确选项的文本由 Listener 的语言决定
        correct_option_text = correct_option
        # 我们用 Listener Agent 来编码正确答案，作为“目标语义”
        # 使用 .detach() 来阻止梯度流回 Listener，因为这个损失是为 Speaker 设计的
        target_semantic_embedding = listener_agent.get_semantic_embedding(correct_option_text).detach()

        # alignment_loss 衡量 Speaker 描述和目标语义的差距，我们希望这个差距越小越好
        # 使用负的余弦相似度作为损失：相似度越高，损失越小
        alignment_loss = -F.cosine_similarity(speaker_desc_embedding, target_semantic_embedding).mean()

        # 5. 合并损失并反向传播
        # 这是两个 Agent 协同学习的核心
        total_loss = final_listener_loss + ALIGNMENT_LOSS_WEIGHT * alignment_loss
        total_loss_sum += total_loss.item()

        # 清空两个 Agent 的梯度
        speaker_agent.optimizer.zero_grad()
        listener_agent.optimizer.zero_grad()

        # 反向传播，梯度会自动流向计算图中涉及的所有参数
        # 即 speaker_agent.model 和 listener_agent.model 的参数都会被计算梯度
        total_loss.backward()

        # 分别更新两个 Agent 的权重
        speaker_agent.optimizer.step()
        listener_agent.optimizer.step()

        # --- 打印和记录 ---
        print(outcome_message)
        print(f"🔮 相似度: {[f'{s:.3f}' for s in similarities.tolist()]}, 预测索引: {predicted_index}")
        print(f"📉 总损失: {total_loss.item():.4f} (Listener Loss: {final_listener_loss.item():.4f}, Speaker Alignment Loss: {alignment_loss.item():.4f})")

        training_log.append({
            "round_idx": round_idx,
            "speaker": speaker_agent.name,
            "listener": listener_agent.name,
            "description": description,
            "options": options,
            "correct_index": correct_index,
            "predicted_index": predicted_index,
            "is_correct": is_correct,
            "total_loss": total_loss.item()
        })


# --- 6. 训练结束，汇总与保存 ---
print("\n--- 训练总结 ---")
total_interactions = len(all_game_rounds_data) * 2
final_accuracy = (correct_predictions_count / total_interactions) * 100 if total_interactions > 0 else 0
print(f"总交互轮数: {total_interactions}")
print(f"总猜对次数: {correct_predictions_count}")
print(f"最终准确率: {final_accuracy:.2f}%")
print(f"平均损失: {total_loss_sum / total_interactions:.4f}")

# 保存结果
output_data = {
    "config": {
        "gpt2_lr": LEARNING_RATE_GPT2,
        "cpm_lr": LEARNING_RATE_CPM,
        "reward": REWARD_CORRECT,
        "penalty": PENALTY_WRONG,
        "alignment_weight": ALIGNMENT_LOSS_WEIGHT
    },
    "summary": {
        "total_interactions": total_interactions,
        "correct_predictions": correct_predictions_count,
        "accuracy": final_accuracy,
        "average_loss": total_loss_sum / total_interactions
    },
    "gpt2_error_log": agent_gpt2.error_log,
    "cpm_error_log": agent_cpm.error_log,
    "detailed_log": training_log
}

output_file_path = os.path.join(OUTPUT_DIR, "interactive_training_results.json")
with open(output_file_path, 'w', encoding='utf-8') as f:
    json.dump(output_data, f, ensure_ascii=False, indent=2)

print(f"\n🎉 完整训练日志和结果已保存到: {output_file_path}")

# 可以选择性地保存微调后的模型
# agent_gpt2.model.save_pretrained(os.path.join(OUTPUT_DIR, "gpt2_finetuned"))
# agent_cpm.model.save_pretrained(os.path.join(OUTPUT_DIR, "cpm_finetuned"))
