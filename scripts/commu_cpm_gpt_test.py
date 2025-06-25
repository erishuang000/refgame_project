import torch
import torch.nn as nn
from transformers import AutoTokenizer, GPT2Model, AutoModelForCausalLM
import json
import torch.nn.functional as F
from torch.optim import AdamW
import os
import random

# --- 1. 全局配置 ---
# ❗️ 请务必根据你的环境修改以下路径
GPT2_MODEL_PATH = "/ubsnhome/23063003r/refgame_project/models/gpt2"
CPM_MODEL_PATH = "/ubsnhome/23063003r/refgame_project/models/CPM-Generate"
DATA_FILE = "/ubsnhome/23063003r/refgame_project/data/bilingual_game_data.json" # ❗️ 确保使用支持双向游戏的新数据格式
OUTPUT_DIR = "/ubsnhome/23063003r/refgame_project/output/"

# 学习超参数
SHARED_EMBEDDING_DIM = 768      # 投射到的共享空间维度
LEARNING_RATE_GPT2 = 5e-6       # GPT-2 的学习率
LEARNING_RATE_CPM = 2e-6        # CPM 的学习率
REWARD_CORRECT = 0.1            # 猜对时, 损失降低 10% (乘以 1-0.1=0.9)
PENALTY_WRONG = 1.1             # 猜错时, 损失增加 10% (乘以 1.1)
ALIGNMENT_LOSS_WEIGHT = 0.5     # 对齐损失在总损失中的权重

# --- 2. Agent 封装类 ---
class Agent:
    """封装 Agent 的所有组件，包括模型、投射层和优化器"""
    def __init__(self, model_name, model, tokenizer, learning_rate, device):
        self.name = model_name
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

        # 获取模型自身的 hidden size
        if hasattr(model.config, 'hidden_size'):
            native_hidden_size = model.config.hidden_size  # For CPM
        elif hasattr(model.config, 'n_embd'):
            native_hidden_size = model.config.n_embd      # For GPT-2
        else:
            raise ValueError(f"Cannot determine hidden size for model {self.name}")

        # 创建从原生维度到共享维度的投射层
        self.projection = nn.Linear(native_hidden_size, SHARED_EMBEDDING_DIM).to(self.device)

        # 优化器需要包含模型和投射层的参数
        self.optimizer = AdamW(
            list(self.model.parameters()) + list(self.projection.parameters()),
            lr=learning_rate
        )
        self.error_log = []

    def get_semantic_embedding(self, text):
        """获取单个句子的语义嵌入，并通过投射层"""
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512).to(self.device)

        # 在推理模式下获取原始嵌入，避免不必要的梯度计算
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(**inputs)
        self.model.train() # 切换回训练模式

        # 兼容不同模型的输出格式
        if hasattr(outputs, 'last_hidden_state'):
            raw_embedding = outputs.last_hidden_state[:, 0, :]
        else:
            # 适用于某些返回元组输出的模型
            raw_embedding = outputs[0][:, 0, :]

        # 将原生嵌入通过投射层，此步会连接计算图
        projected_embedding = self.projection(raw_embedding)
        return projected_embedding

    def get_candidate_embeddings(self, sentences):
        """获取多个候选句子的嵌入"""
        embeddings = [self.get_semantic_embedding(s) for s in sentences]
        return torch.cat(embeddings, dim=0)

    def train_mode(self):
        """将模型和投射层都设置为训练模式"""
        self.model.train()
        self.projection.train()

# --- 3. 初始化工作 ---
# 创建输出目录
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)
    print(f"创建输出目录: {OUTPUT_DIR}")

# 确定设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

# 初始化英文 Agent (GPT-2)
print("--- 初始化英文 Agent (GPT-2) ---")
gpt2_tokenizer = AutoTokenizer.from_pretrained(GPT2_MODEL_PATH)
if gpt2_tokenizer.pad_token is None:
    gpt2_tokenizer.add_special_tokens({'pad_token': '[PAD]'})
gpt2_model = GPT2Model.from_pretrained(GPT2_MODEL_PATH).to(device)
gpt2_model.resize_token_embeddings(len(gpt2_tokenizer))
agent_gpt2 = Agent("GPT-2", gpt2_model, gpt2_tokenizer, LEARNING_RATE_GPT2, device)
agent_gpt2.train_mode()

# 初始化中文 Agent (CPM)
print("--- 初始化中文 Agent (CPM) ---")
cpm_tokenizer = AutoTokenizer.from_pretrained(CPM_MODEL_PATH)
cpm_model_full = AutoModelForCausalLM.from_pretrained(CPM_MODEL_PATH).to(device)
# 我们只将 Transformer 主体部分作为模型传入 Agent
agent_cpm = Agent("CPM", cpm_model_full.transformer, cpm_tokenizer, LEARNING_RATE_CPM, device)
agent_cpm.train_mode()


# --- 4. 准备数据 ---
try:
    with open(DATA_FILE, 'r', encoding='utf-8') as f:
        all_game_rounds_data = json.load(f)
    print(f"✅ 成功加载 {len(all_game_rounds_data)} 轮游戏数据。")
except Exception as e:
    print(f"❌ 错误: 无法加载或解析数据文件 '{DATA_FILE}'. 请检查路径和JSON格式。 {e}")
    exit()

# --- 5. 主训练循环 ---
total_rounds = len(all_game_rounds_data)
training_log = []
total_loss_sum = 0.0
correct_predictions_count = 0

print(f"\n--- 开始进行 {total_rounds * 2} 轮交互式游戏 ---")

# 每个数据点可以玩两轮，角色互换
for i, game_data in enumerate(all_game_rounds_data):
    for turn in range(2):
        round_idx = i * 2 + turn + 1

        # 角色分配
        if turn == 0:
            # 轮次 1: GPT-2 是 Speaker, CPM 是 Listener
            speaker_agent, listener_agent = agent_gpt2, agent_cpm
            description = game_data['description_english']
            correct_option = game_data['concept_chinese']
            distractors = game_data['distractors_chinese']
        else:
            # 轮次 2: CPM 是 Speaker, GPT-2 是 Listener
            speaker_agent, listener_agent = agent_cpm, agent_gpt2
            description = game_data['description_chinese']
            correct_option = game_data['concept_english']
            distractors = game_data['distractors_english']

        print(f"\n--- 游戏回合 {round_idx} ---")
        print(f"🗣️ Speaker ({speaker_agent.name}) 描述: \"{description}\"")

        # 准备选项并随机排序
        options = distractors + [correct_option]
        random.shuffle(options)
        correct_index = options.index(correct_option)
        print(f"👂 Listener ({listener_agent.name}) 选项: {options}")

        # --- 交互与计算 ---
        desc_embedding = listener_agent.get_semantic_embedding(description)
        option_embeddings = listener_agent.get_candidate_embeddings(options)

        similarities = F.cosine_similarity(desc_embedding, option_embeddings, dim=1)
        predicted_index = torch.argmax(similarities).item()

        # --- 反馈与学习 ---
        is_correct = (predicted_index == correct_index)
        correct_index_tensor = torch.tensor([correct_index], device=device)

        # 计算 Listener 的选择损失
        listener_loss = F.cross_entropy(similarities.unsqueeze(0), correct_index_tensor)

        if is_correct:
            final_listener_loss = listener_loss * (1 - REWARD_CORRECT)
            outcome_message = f"🎉 {listener_agent.name} 猜对了!"
            correct_predictions_count += 1
        else:
            final_listener_loss = listener_loss * PENALTY_WRONG
            outcome_message = f"💔 {listener_agent.name} 猜错了! (正确答案: {correct_option})"
            speaker_agent.error_log.append({"description": description, "listener_chose": options[predicted_index], "correct": correct_option, "round": round_idx})

        # 计算 Speaker 的对齐损失 (无论对错，都以正确答案为引导)
        speaker_desc_embedding = speaker_agent.get_semantic_embedding(description)
        # 使用 Listener 编码正确答案的文本，作为“目标语义”
        target_semantic_embedding = listener_agent.get_semantic_embedding(correct_option).detach()
        alignment_loss = -F.cosine_similarity(speaker_desc_embedding, target_semantic_embedding).mean()

        # 合并损失
        total_loss = final_listener_loss + ALIGNMENT_LOSS_WEIGHT * alignment_loss
        total_loss_sum += total_loss.item()

        # 反向传播并更新两个 Agent
        speaker_agent.optimizer.zero_grad()
        listener_agent.optimizer.zero_grad()
        total_loss.backward()
        speaker_agent.optimizer.step()
        listener_agent.optimizer.step()

        # --- 打印和记录 ---
        print(outcome_message)
        print(f"🔮 相似度: {[f'{s:.3f}' for s in similarities.tolist()]}, 预测: {predicted_index}, 正确: {correct_index}")
        print(f"📉 总损失: {total_loss.item():.4f} (Listener Loss: {final_listener_loss.item():.4f}, Speaker Alignment Loss: {alignment_loss.item():.4f})")

        training_log.append({"round_idx": round_idx, "speaker": speaker_agent.name, "listener": listener_agent.name, "is_correct": is_correct, "total_loss": total_loss.item()})


# --- 6. 训练结束，汇总与保存 ---
print("\n--- 训练总结 ---")
total_interactions = len(all_game_rounds_data) * 2
final_accuracy = (correct_predictions_count / total_interactions) * 100 if total_interactions > 0 else 0
print(f"总交互轮数: {total_interactions}")
print(f"总猜对次数: {correct_predictions_count}")
print(f"最终准确率: {final_accuracy:.2f}%")
print(f"平均损失: {total_loss_sum / total_interactions:.4f}")

# 保存详细结果
output_data = {
    "config": {"shared_dim": SHARED_EMBEDDING_DIM, "gpt2_lr": LEARNING_RATE_GPT2, "cpm_lr": LEARNING_RATE_CPM, "alignment_weight": ALIGNMENT_LOSS_WEIGHT},
    "summary": {"accuracy": final_accuracy, "avg_loss": total_loss_sum / total_interactions},
    "gpt2_error_log": agent_gpt2.error_log,
    "cpm_error_log": agent_cpm.error_log,
    "detailed_log": training_log
}
output_file_path = os.path.join(OUTPUT_DIR, "final_training_results.json")
with open(output_file_path, 'w', encoding='utf-8') as f:
    json.dump(output_data, f, ensure_ascii=False, indent=2)

print(f"\n🎉 完整训练日志和结果已保存到: {output_file_path}")
