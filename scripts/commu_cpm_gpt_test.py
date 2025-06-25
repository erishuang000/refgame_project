# ✅ 最终修正版代码，附带清晰的、用于调试和分析的详细过程打印
# 1. 【已修正】确保 GPT-2 Tokenizer 能处理中文，使其能扮演 Listener 角色
# 2. 【已修正】确保 Speaker Agent 使用自己语言的概念生成描述
# 3. 【打印增强】对所有交互和计算步骤，都加入了清晰的打印语句

import torch
import torch.nn as nn
from transformers import AutoTokenizer, GPT2LMHeadModel, AutoModelForCausalLM
import json
import torch.nn.functional as F
from torch.optim import AdamW
import os
import random

# --- 全局配置 ---
# ❗️ 请根据您的环境修改路径
GPT2_MODEL_PATH = "/ubsnhome/23063003r/refgame_project/models/gpt2"
CPM_MODEL_PATH = "/ubsnhome/23063003r/refgame_project/models/CPM-Generate"
DATA_FILE = "/ubsnhome/23063003r/refgame_project/data/bilingual_game_data.json"
OUTPUT_DIR = "/ubsnhome/23063003r/refgame_project/output/"

# --- 超参数配置 ---
SHARED_EMBEDDING_DIM = 768
LEARNING_RATE_GPT2 = 5e-6
LEARNING_RATE_CPM = 2e-6
REWARD_CORRECT = 0.1
PENALTY_WRONG = 1.1
ALIGNMENT_LOSS_WEIGHT = 0.5
ANTI_ALIGNMENT_WEIGHT = 0.5
CONTRASTIVE_LOSS_WEIGHT = 0.5
CONTRASTIVE_TEMPERATURE = 0.1

# --- Agent 类定义 ---
class Agent:
    def __init__(self, model_name, model, tokenizer, learning_rate, device):
        self.name = model_name
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        if hasattr(model.config, 'hidden_size'): native_hidden_size = model.config.hidden_size
        elif hasattr(model.config, 'n_embd'): native_hidden_size = model.config.n_embd
        else: raise ValueError(f"Cannot determine hidden size for model {self.name}")
        self.projection = nn.Linear(native_hidden_size, SHARED_EMBEDDING_DIM).to(self.device)
        self.optimizer = AdamW(list(self.model.parameters()) + list(self.projection.parameters()), lr=learning_rate)
        self.error_log = []

    def get_semantic_embedding(self, text: str):
        self.model.eval()
        with torch.no_grad():
            inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512).to(self.device)
            outputs = self.model(**inputs)
        self.model.train()
        if hasattr(outputs, 'last_hidden_state'): raw_embedding = outputs.last_hidden_state[:, 0, :]
        elif hasattr(outputs, 'logits'): raw_embedding = self.model.transformer(inputs['input_ids'])[0][:,0,:]
        else: raw_embedding = outputs[0][:, 0, :]
        return self.projection(raw_embedding)

    def get_candidate_embeddings(self, sentences: list):
        embeddings = [self.get_semantic_embedding(s) for s in sentences]
        return torch.cat(embeddings, dim=0)

    def train_mode(self):
        self.model.train()
        self.projection.train()

# --- 辅助函数 ---
def generate_description(concept: str, agent: Agent, max_tokens=30) -> str:
    # --- 【‼️ 关键修正 ‼️】根据Agent的类型，动态选择更优的Prompt格式 ---
    if agent.name == "GPT-2":
        # GPT-2 对这种指令式Prompt响应良好
        prompt = f"A short description of the concept '{concept}' is:"
    elif agent.name == "CPM":
        # 对于CPM，使用更像“文本补全”的格式，诱导其进行描述
        # 格式1：最简单的补全
        prompt = f"{concept}是"
        # 格式2：稍微丰富的上下文
        # prompt = f"关于“{concept}”的简单介绍：{concept}"
    else:
        prompt = f"Describe: {concept}\n"

    input_ids = agent.tokenizer(prompt, return_tensors="pt").input_ids.to(agent.device)
    input_length = input_ids.shape[1] # 记录输入部分的长度

    agent.model.eval()
    output_ids = agent.model.generate(
        input_ids,
        max_new_tokens=max_tokens,
        do_sample=True,
        top_k=50,
        temperature=0.7,
        pad_token_id=agent.tokenizer.eos_token_id
    )
    agent.model.train()

    # --- 【‼️ 关键修正 ‼️】使用更鲁棒的方式解码，避免复述Prompt ---
    # 只解码生成的部分，而不是全部
    generated_ids = output_ids[0][input_length:]
    description = agent.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

    return description if description else f"一个叫做 {concept} 的东西"
    
# --- 初始化流程 ---
os.makedirs(OUTPUT_DIR, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --- 扩展 GPT-2 Tokenizer 以处理中文 ---
print("Loading data to collect Chinese vocabulary for GPT-2...")
with open(DATA_FILE, 'r', encoding='utf-8') as f:
    data = json.load(f)

all_chinese_words = set()
for sample in data:
    all_chinese_words.add(sample['concept_chinese'])
    all_chinese_words.update(sample['distractors_chinese'])
all_chinese_chars = set(''.join(list(all_chinese_words)))
new_chinese_tokens = list(all_chinese_words.union(all_chinese_chars))
print(f"Found {len(new_chinese_tokens)} unique Chinese words/characters to add to GPT-2 tokenizer.")

# 初始化 GPT-2 Agent
print("Loading and configuring GPT-2...")
gpt2_tokenizer = AutoTokenizer.from_pretrained(GPT2_MODEL_PATH)
if gpt2_tokenizer.pad_token is None:
    gpt2_tokenizer.pad_token = gpt2_tokenizer.eos_token

gpt2_tokenizer.add_tokens(new_chinese_tokens)
print(f"GPT-2 tokenizer vocabulary extended. New size: {len(gpt2_tokenizer)}")

gpt2_model = GPT2LMHeadModel.from_pretrained(GPT2_MODEL_PATH).to(device)
gpt2_model.resize_token_embeddings(len(gpt2_tokenizer))
print("GPT-2 model embedding layer resized.")

gpt2_agent = Agent("GPT-2", gpt2_model, gpt2_tokenizer, LEARNING_RATE_GPT2, device)
gpt2_agent.train_mode()

# 初始化 CPM Agent
print("Loading CPM...")
cpm_tokenizer = AutoTokenizer.from_pretrained(CPM_MODEL_PATH)
cpm_model_full = AutoModelForCausalLM.from_pretrained(CPM_MODEL_PATH).to(device)
cpm_agent = Agent("CPM", cpm_model_full, cpm_tokenizer, LEARNING_RATE_CPM, device)
cpm_agent.train_mode()

# --- 主训练循环 (附带详细打印) ---
log = []
total_loss_sum = 0.0
correct_count = 0

print(f"\n--- Starting {len(data) * 2} rounds of interaction (Verbose Mode) ---")
for i, sample in enumerate(data):
    for turn in range(2):
        round_idx = i * 2 + turn + 1
        print(f"\n\n{'='*25} 游戏回合 {round_idx} {'='*25}")

        # --- 阶段 1: 角色分配与任务设置 ---
        print("\n▶️  阶段 1: 角色分配与任务设置")
        if turn == 0:
            speaker, listener = gpt2_agent, cpm_agent
            concept_for_speaker = sample['concept_english']
            correct_option_for_listener = sample['concept_chinese']
            options_for_listener = sample['distractors_chinese'] + [correct_option_for_listener]
        else:
            speaker, listener = cpm_agent, gpt2_agent
            concept_for_speaker = sample['concept_chinese']
            correct_option_for_listener = sample['concept_english']
            options_for_listener = sample['distractors_english'] + [correct_option_for_listener]
        print(f"    🗣️  Speaker: {speaker.name}")
        print(f"    👂  Listener: {listener.name}")

        # --- 阶段 2: Speaker 生成描述 ---
        print("\n▶️  阶段 2: Speaker 生成描述")
        print(f"    Speaker接收到的概念: '{concept_for_speaker}'")
        random.shuffle(options_for_listener)
        correct_index = options_for_listener.index(correct_option_for_listener)
        description = generate_description(concept_for_speaker, speaker)
        print(f"    Speaker生成的描述: '{description}'")

        # --- 阶段 3: Listener 理解与猜测 ---
        print("\n▶️  阶段 3: Listener 理解与猜测")
        print(f"    Listener接收到的选项: {options_for_listener}")
        print(f"    (本轮正确答案是索引 {correct_index}: '{correct_option_for_listener}')")
        desc_embed = listener.get_semantic_embedding(description)
        opt_embeds = listener.get_candidate_embeddings(options_for_listener)
        sims = F.cosine_similarity(desc_embed, opt_embeds, dim=1)
        pred_idx = torch.argmax(sims).item()

        print("    计算出的相似度:")
        for k, sim_val in enumerate(sims.tolist()):
            print(f"        - 选项 '{options_for_listener[k]}': {sim_val:.4f}")

        print(f"    Listener的选择: 索引 {pred_idx} -> '{options_for_listener[pred_idx]}'")
        is_correct = (pred_idx == correct_index)
        if is_correct:
            print("    ✅ 结论: 猜测正确！")
        else:
            print("    ❌ 结论: 猜测错误！")

        # --- 阶段 4: 复合损失计算 (详细分解) ---
        print("\n▶️  阶段 4: 复合损失计算 (详细分解)")

        # A. Listener Loss
        correct_tensor = torch.tensor([correct_index], device=device)
        listener_loss = F.cross_entropy(sims.unsqueeze(0), correct_tensor)
        reward_penalty_factor = (1 - REWARD_CORRECT) if is_correct else PENALTY_WRONG
        final_listener_loss = listener_loss * reward_penalty_factor
        print(f"    [A] Listener Loss:")
        print(f"        - 基础交叉熵损失: {listener_loss.item():.4f}")
        print(f"        - 奖惩系数: {reward_penalty_factor:.2f} ({'奖励' if is_correct else '惩罚'})")
        print(f"        -  => 最终 Listener Loss = {final_listener_loss.item():.4f}")

        # 获取用于 Speaker 学习的嵌入向量
        speaker_embed = speaker.get_semantic_embedding(description)
        target_embed = listener.get_semantic_embedding(correct_option_for_listener).detach()

        # B. Alignment Loss (吸引力)
        alignment_loss = -F.cosine_similarity(speaker_embed, target_embed).mean()
        print(f"    [B] Alignment Loss (吸引力):")
        print(f"        - 目标: 拉近'{description[:20]}...'和'{correct_option_for_listener}'的语义距离")
        print(f"        -  => 计算出的 Alignment Loss = {alignment_loss.item():.4f}")

        # C. Anti-Alignment Loss (排斥力)
        anti_loss = torch.tensor(0.0, device=device)
        print(f"    [C] Anti-Alignment Loss (排斥力):")
        if not is_correct:
            misleading_option = options_for_listener[pred_idx]
            misleading_embed = listener.get_semantic_embedding(misleading_option).detach()
            anti_loss = F.cosine_similarity(speaker_embed, misleading_embed).mean()
            print(f"        - 目标: 推远'{description[:20]}...'和被错选的'{misleading_option}'的语义距离")
            print(f"        -  => 计算出的 Anti-Alignment Loss = {anti_loss.item():.4f}")
        else:
            print("        - (猜测正确，无需计算此项损失)")

        # D. Contrastive Loss (区分力)
        print(f"    [D] Contrastive Loss (区分力):")
        negative_options = [opt for opt in options_for_listener if opt != correct_option_for_listener]
        if negative_options:
            negative_embeds = listener.get_candidate_embeddings(negative_options).detach()
            pos_sim = F.cosine_similarity(speaker_embed, target_embed) / CONTRASTIVE_TEMPERATURE
            neg_sims = F.cosine_similarity(speaker_embed, negative_embeds) / CONTRASTIVE_TEMPERATURE
            all_logits = torch.cat([pos_sim, neg_sims.view(-1)])
            contrastive_labels = torch.tensor([0], device=device)
            contrastive_loss = F.cross_entropy(all_logits.unsqueeze(0), contrastive_labels)
            print(f"        - 目标: 让'{description[:20]}...'与'{correct_option_for_listener}'的相似度远高于其他所有选项")
            print(f"        -  => 计算出的 Contrastive Loss = {contrastive_loss.item():.4f}")
        else:
            contrastive_loss = torch.tensor(0.0, device=device)
            print("        - (没有负样本，无需计算此项损失)")

        # E. 合并总损失
        print(f"    [E] 总损失计算:")
        total_loss = final_listener_loss + \
                     ALIGNMENT_LOSS_WEIGHT * alignment_loss + \
                     ANTI_ALIGNMENT_WEIGHT * anti_loss + \
                     CONTRASTIVE_LOSS_WEIGHT * contrastive_loss
        print("        " + "-"*40)
        print(f"        Total Loss = Listener_Loss + (W_align * Align_L) + (W_anti * Anti_L) + (W_contrast * Contrast_L)")
        print(f"                   = {final_listener_loss.item():.4f} + ({ALIGNMENT_LOSS_WEIGHT} * {alignment_loss.item():.4f}) + ({ANTI_ALIGNMENT_WEIGHT} * {anti_loss.item():.4f}) + ({CONTRASTIVE_LOSS_WEIGHT} * {contrastive_loss.item():.4f})")
        print(f"                   = {final_listener_loss.item():.4f} + {ALIGNMENT_LOSS_WEIGHT*alignment_loss.item():.4f} + {ANTI_ALIGNMENT_WEIGHT*anti_loss.item():.4f} + {CONTRASTIVE_LOSS_WEIGHT*contrastive_loss.item():.4f}")
        print(f"        ----------------------------------------")
        print(f"        ==> 💸 最终总损失 (Final Total Loss): {total_loss.item():.4f}")

        # 4. 协同更新
        print("\n▶️  阶段 5: 模型协同更新")
        print("    执行 total_loss.backward() 计算两个Agent所有相关参数的梯度...")
        speaker.optimizer.zero_grad()
        listener.optimizer.zero_grad()
        total_loss.backward()
        print(f"    执行 speaker.optimizer.step() 更新 {speaker.name} 的权重...")
        speaker.optimizer.step()
        print(f"    执行 listener.optimizer.step() 更新 {listener.name} 的权重...")
        listener.optimizer.step()
        print("    ✅ 更新完成!")

        # 5. 日志记录
        total_loss_sum += total_loss.item()
        if is_correct:
            correct_count += 1
        else:
            speaker.error_log.append({
                "round": round_idx,
                "speaker_concept": concept_for_speaker,
                "description": description,
                "listener_chose": options_for_listener[pred_idx],
                "correct_answer": correct_option_for_listener
            })
        log.append({"round": round_idx, "speaker": speaker.name, "correct": is_correct, "loss": total_loss.item()})

# --- 总结与保存 ---
total_games = len(data) * 2
accuracy = (correct_count / total_games * 100) if total_games > 0 else 0
avg_loss = total_loss_sum / total_games if total_games > 0 else 0

summary = {
    "config": {
        "shared_dim": SHARED_EMBEDDING_DIM, "gpt2_lr": LEARNING_RATE_GPT2, "cpm_lr": LEARNING_RATE_CPM,
        "alignment_weight": ALIGNMENT_LOSS_WEIGHT, "anti_align_weight": ANTI_ALIGNMENT_WEIGHT,
        "contrastive_weight": CONTRASTIVE_LOSS_WEIGHT, "temperature": CONTRASTIVE_TEMPERATURE
    },
    "summary": {"accuracy": accuracy, "avg_loss": avg_loss},
    "gpt2_error_log": gpt2_agent.error_log,
    "cpm_error_log": cpm_agent.error_log,
    "log": log
}

output_file_path = os.path.join(OUTPUT_DIR, "final_training_run_results_verbose.json")
with open(output_file_path, 'w', encoding='utf-8') as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)

print(f"\n\n{'='*25} 训练结束 {'='*25}")
print(f"✅ 准确率: {accuracy:.2f}% | 平均损失: {avg_loss:.4f}")
print(f"📄 详细结果已保存至: {output_file_path}")
