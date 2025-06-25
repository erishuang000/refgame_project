# ✅ 已重构完整代码，并采纳了关键修正和优化
# 1. 【已修正】Speaker 使用自回归语言模型为自己语言的概念生成描述
# 2. 【已优化】get_semantic_embedding 中加入 eval() 模式切换，提升效率
# 3. 【已优化】Contrastive loss 计算避免循环，提升效率
# 4. 保留所有四种损失函数，形成强大的协同学习机制

import torch
import torch.nn as nn
from transformers import AutoTokenizer, GPT2LMHeadModel, GPT2Model, AutoModelForCausalLM
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

        if hasattr(model.config, 'hidden_size'):
            native_hidden_size = model.config.hidden_size
        elif hasattr(model.config, 'n_embd'):
            native_hidden_size = model.config.n_embd
        else:
            raise ValueError(f"Cannot determine hidden size for model {self.name}")

        self.projection = nn.Linear(native_hidden_size, SHARED_EMBEDDING_DIM).to(self.device)
        self.optimizer = AdamW(list(self.model.parameters()) + list(self.projection.parameters()), lr=learning_rate)
        self.error_log = []

    def get_semantic_embedding(self, text: str):
        # 切换到评估模式以禁用 dropout 等，并阻止梯度计算，以加速和节省内存
        self.model.eval()
        with torch.no_grad():
            inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512).to(self.device)
            outputs = self.model(**inputs)
        # 完成推理后，立即切回训练模式，以便后续的梯度计算
        self.model.train()

        # 兼容不同模型的输出格式
        if hasattr(outputs, 'last_hidden_state'):
             # GPT2Model 或者 CPM 的 transformer 主体
            raw_embedding = outputs.last_hidden_state[:, 0, :]
        elif hasattr(outputs, 'logits'):
             # GPT2LMHeadModel
            raw_embedding = self.model.transformer(inputs['input_ids'])[0][:,0,:]
        else:
            raw_embedding = outputs[0][:, 0, :]

        # 通过投射层连接计算图
        return self.projection(raw_embedding)

    def get_candidate_embeddings(self, sentences: list):
        # 确保以一致的方式获取嵌入
        # 注意：此处为简化，逐句处理。若性能瓶颈可优化为批处理。
        embeddings = [self.get_semantic_embedding(s) for s in sentences]
        return torch.cat(embeddings, dim=0)

    def train_mode(self):
        self.model.train()
        self.projection.train()

# --- 辅助函数 ---
def generate_description(concept: str, agent: Agent, max_tokens=30) -> str:
    # ❗️ 使用更健壮的 prompt 格式
    prompt = f"A short description of the concept '{concept}' is:"
    input_ids = agent.tokenizer(prompt, return_tensors="pt").input_ids.to(agent.device)

    # 在生成时也使用 eval 模式
    agent.model.eval()
    output_ids = agent.model.generate(
        input_ids,
        max_new_tokens=max_tokens,
        do_sample=True,
        top_k=50,
        temperature=0.7,
        pad_token_id=agent.tokenizer.eos_token_id  # 避免 pad token warning
    )
    agent.model.train() # 切回训练模式

    # 解码并清洗文本
    full_text = agent.tokenizer.decode(output_ids[0], skip_special_tokens=True)
    description = full_text.replace(prompt, '').strip().split('\n')[0] # 只取第一行，避免生成多余内容
    return description if description else f"a thing called {concept}" # 避免空描述

# --- 初始化流程 ---
os.makedirs(OUTPUT_DIR, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

print("Loading GPT-2...")
gpt2_tokenizer = AutoTokenizer.from_pretrained(GPT2_MODEL_PATH)
if gpt2_tokenizer.pad_token is None:
    gpt2_tokenizer.pad_token = gpt2_tokenizer.eos_token
gpt2_model = GPT2LMHeadModel.from_pretrained(GPT2_MODEL_PATH).to(device)
gpt2_model.resize_token_embeddings(len(gpt2_tokenizer))
gpt2_agent = Agent("GPT-2", gpt2_model, gpt2_tokenizer, LEARNING_RATE_GPT2, device)
gpt2_agent.train_mode()

print("Loading CPM...")
cpm_tokenizer = AutoTokenizer.from_pretrained(CPM_MODEL_PATH)
cpm_model_full = AutoModelForCausalLM.from_pretrained(CPM_MODEL_PATH).to(device)
# 注意：CPM-Generate 的生成能力在完整模型上，但我们用于提取 embedding 的是其 transformer 主体
cpm_agent = Agent("CPM", cpm_model_full, cpm_tokenizer, LEARNING_RATE_CPM, device)
cpm_agent.train_mode()

print("Loading data...")
with open(DATA_FILE, 'r', encoding='utf-8') as f:
    data = json.load(f)

# --- 主训练循环 ---
log = []
total_loss_sum = 0.0
correct_count = 0

print(f"--- Starting {len(data) * 2} rounds of interaction ---")
for i, sample in enumerate(data):
    for turn in range(2):
        round_idx = i * 2 + turn + 1

        # --- 【核心修正】根据角色，正确分配各自语言的概念 ---
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

        # 1. Speaker 生成描述
        random.shuffle(options_for_listener)
        correct_index = options_for_listener.index(correct_option_for_listener)
        description = generate_description(concept_for_speaker, speaker)

        # 2. Listener 猜测
        desc_embed = listener.get_semantic_embedding(description)
        opt_embeds = listener.get_candidate_embeddings(options_for_listener)
        sims = F.cosine_similarity(desc_embed, opt_embeds, dim=1)
        pred_idx = torch.argmax(sims).item()

        # 3. 计算复合损失
        is_correct = (pred_idx == correct_index)
        correct_tensor = torch.tensor([correct_index], device=device)

        # A. Listener Loss
        listener_loss = F.cross_entropy(sims.unsqueeze(0), correct_tensor)
        final_listener_loss = listener_loss * (1 - REWARD_CORRECT if is_correct else PENALTY_WRONG)

        # 获取 Speaker 描述的嵌入，用于后续损失计算
        speaker_embed = speaker.get_semantic_embedding(description)
        # 获取 Listener 对正确选项的嵌入，作为对齐目标
        target_embed = listener.get_semantic_embedding(correct_option_for_listener).detach()

        # B. Alignment Loss (吸引力)
        alignment_loss = -F.cosine_similarity(speaker_embed, target_embed).mean()

        # C. Anti-Alignment Loss (排斥力)
        anti_loss = torch.tensor(0.0, device=device)
        if not is_correct:
            misleading_option = options_for_listener[pred_idx]
            misleading_embed = listener.get_semantic_embedding(misleading_option).detach()
            # 我们希望 speaker_embed 和 misleading_embed 的相似度变小，即 -sim 的值变小。
            # 所以损失函数是 sim 本身，最小化这个损失就是最小化相似度。
            anti_loss = F.cosine_similarity(speaker_embed, misleading_embed).mean()

        # D. Contrastive Loss (区分力) - 【已优化】
        negative_options = [opt for opt in options_for_listener if opt != correct_option_for_listener]
        negative_embeds = listener.get_candidate_embeddings(negative_options).detach()

        pos_sim = F.cosine_similarity(speaker_embed, target_embed) / CONTRASTIVE_TEMPERATURE
        neg_sims = F.cosine_similarity(speaker_embed, negative_embeds) / CONTRASTIVE_TEMPERATURE

        all_logits = torch.cat([pos_sim, neg_sims.view(-1)]) # 形状 [1+N]
        contrastive_labels = torch.tensor([0], device=device)
        contrastive_loss = F.cross_entropy(all_logits.unsqueeze(0), contrastive_labels)

        # E. 合并总损失
        total_loss = final_listener_loss + \
                     ALIGNMENT_LOSS_WEIGHT * alignment_loss + \
                     ANTI_ALIGNMENT_WEIGHT * anti_loss + \
                     CONTRASTIVE_LOSS_WEIGHT * contrastive_loss

        # 4. 协同更新
        speaker.optimizer.zero_grad()
        listener.optimizer.zero_grad()
        total_loss.backward()
        speaker.optimizer.step()
        listener.optimizer.step()

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
        print(f"Round {round_idx}: Speaker={speaker.name}, Correct={is_correct}, Loss={total_loss.item():.4f}")

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

output_file_path = os.path.join(OUTPUT_DIR, "final_training_run_results.json")
with open(output_file_path, 'w', encoding='utf-8') as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)

print(f"\n--- Training Finished ---")
print(f"✅ Accuracy: {accuracy:.2f}% | Avg Loss: {avg_loss:.4f}")
print(f"📄 Results saved to {output_file_path}")
