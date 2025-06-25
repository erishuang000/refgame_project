# ✅ 已重构完整代码，新增功能：
# 1. Speaker 使用自回归语言模型主动生成描述句
# 2. Speaker 加入 Anti-alignment loss，从 Listener 错误中学习
# 3. 加入 Contrastive loss 强化语义区分度
# 4. 保留 alignment loss 和 listener cross-entropy loss

import torch
import torch.nn as nn
from transformers import AutoTokenizer, GPT2LMHeadModel, GPT2Model, AutoModelForCausalLM
import json
import torch.nn.functional as F
from torch.optim import AdamW
import os
import random

# --- 全局配置 ---
GPT2_MODEL_PATH = "/ubsnhome/23063003r/refgame_project/models/gpt2"
CPM_MODEL_PATH = "/ubsnhome/23063003r/refgame_project/models/CPM-Generate"
DATA_FILE = "/ubsnhome/23063003r/refgame_project/data/bilingual_game_data.json"
OUTPUT_DIR = "/ubsnhome/23063003r/refgame_project/output/"

SHARED_EMBEDDING_DIM = 768
LEARNING_RATE_GPT2 = 5e-6
LEARNING_RATE_CPM = 2e-6
REWARD_CORRECT = 0.1
PENALTY_WRONG = 1.1
ALIGNMENT_LOSS_WEIGHT = 0.5
ANTI_ALIGNMENT_WEIGHT = 0.5
CONTRASTIVE_LOSS_WEIGHT = 0.5

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

    def get_semantic_embedding(self, text):
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512).to(self.device)
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(**inputs)
        self.model.train()
        raw_embedding = outputs.last_hidden_state[:, 0, :] if hasattr(outputs, 'last_hidden_state') else outputs[0][:, 0, :]
        return self.projection(raw_embedding)

    def get_candidate_embeddings(self, sentences):
        embeddings = [self.get_semantic_embedding(s) for s in sentences]
        return torch.cat(embeddings, dim=0)

    def train_mode(self):
        self.model.train()
        self.projection.train()

def generate_description(concept, agent: Agent, max_tokens=30):
    prompt = f"Describe this concept: {concept}\nDescription:"
    input_ids = agent.tokenizer(prompt, return_tensors="pt").input_ids.to(agent.device)
    output = agent.model.generate(input_ids, max_new_tokens=max_tokens, do_sample=True, top_k=50, temperature=0.7)
    return agent.tokenizer.decode(output[0], skip_special_tokens=True).replace(prompt, '').strip()

# 初始化设备和目录
os.makedirs(OUTPUT_DIR, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 初始化模型
print("Loading GPT-2...")
gpt2_tokenizer = AutoTokenizer.from_pretrained(GPT2_MODEL_PATH)
gpt2_tokenizer.pad_token = gpt2_tokenizer.pad_token or '[PAD]'
gpt2_model = GPT2LMHeadModel.from_pretrained(GPT2_MODEL_PATH).to(device)
gpt2_model.resize_token_embeddings(len(gpt2_tokenizer))
gpt2_agent = Agent("GPT-2", gpt2_model, gpt2_tokenizer, LEARNING_RATE_GPT2, device)
gpt2_agent.train_mode()

print("Loading CPM...")
cpm_tokenizer = AutoTokenizer.from_pretrained(CPM_MODEL_PATH)
cpm_model_full = AutoModelForCausalLM.from_pretrained(CPM_MODEL_PATH).to(device)
cpm_agent = Agent("CPM", cpm_model_full.transformer, cpm_tokenizer, LEARNING_RATE_CPM, device)
cpm_agent.train_mode()

# 加载数据
with open(DATA_FILE, 'r', encoding='utf-8') as f:
    data = json.load(f)

log = []
total_loss_sum = 0.0
correct_count = 0

for i, sample in enumerate(data):
    for turn in range(2):
        round_idx = i * 2 + turn + 1
        if turn == 0:
            speaker, listener = gpt2_agent, cpm_agent
            concept = sample['concept_chinese']
            options = sample['distractors_chinese'] + [concept]
        else:
            speaker, listener = cpm_agent, gpt2_agent
            concept = sample['concept_english']
            options = sample['distractors_english'] + [concept]

        random.shuffle(options)
        correct_index = options.index(concept)
        description = generate_description(concept, speaker)

        desc_embed = listener.get_semantic_embedding(description)
        opt_embeds = listener.get_candidate_embeddings(options)
        sims = F.cosine_similarity(desc_embed, opt_embeds, dim=1)
        pred_idx = torch.argmax(sims).item()
        is_correct = (pred_idx == correct_index)
        correct_tensor = torch.tensor([correct_index], device=device)
        listener_loss = F.cross_entropy(sims.unsqueeze(0), correct_tensor)
        final_loss = listener_loss * (1 - REWARD_CORRECT if is_correct else PENALTY_WRONG)

        speaker_embed = speaker.get_semantic_embedding(description)
        target_embed = listener.get_semantic_embedding(concept).detach()
        alignment_loss = -F.cosine_similarity(speaker_embed, target_embed).mean()

        # Anti-alignment from wrong option
        anti_loss = 0.0
        if not is_correct:
            misleading = options[pred_idx]
            misleading_embed = listener.get_semantic_embedding(misleading).detach()
            anti_loss = F.cosine_similarity(speaker_embed, misleading_embed).mean()

        # Contrastive Loss: Description vs (correct, negatives)
        negatives = [opt for idx, opt in enumerate(options) if idx != correct_index]
        negative_embeds = listener.get_candidate_embeddings(negatives).detach()
        temperature = 0.1
        pos_sim = F.cosine_similarity(speaker_embed, target_embed) / temperature
        neg_sims = torch.cat([F.cosine_similarity(speaker_embed, neg.unsqueeze(0)) / temperature for neg in negative_embeds], dim=0)
        all_logits = torch.cat([pos_sim.unsqueeze(0), neg_sims])
        contrastive_labels = torch.zeros(1, dtype=torch.long, device=device)
        contrastive_loss = F.cross_entropy(all_logits.unsqueeze(0), contrastive_labels)

        total_loss = final_loss + ALIGNMENT_LOSS_WEIGHT * alignment_loss + \
                      ANTI_ALIGNMENT_WEIGHT * anti_loss + \
                      CONTRASTIVE_LOSS_WEIGHT * contrastive_loss

        total_loss_sum += total_loss.item()
        if is_correct:
            correct_count += 1
        else:
            speaker.error_log.append({"description": description, "listener_chose": options[pred_idx], "correct": concept, "round": round_idx})

        speaker.optimizer.zero_grad()
        listener.optimizer.zero_grad()
        total_loss.backward()
        speaker.optimizer.step()
        listener.optimizer.step()

        log.append({"round": round_idx, "speaker": speaker.name, "listener": listener.name, "correct": is_correct, "loss": total_loss.item()})

# 总结保存
accuracy = correct_count / (2 * len(data)) * 100
avg_loss = total_loss_sum / (2 * len(data))

with open(os.path.join(OUTPUT_DIR, "final_training_results.json"), 'w', encoding='utf-8') as f:
    json.dump({
        "config": {
            "shared_dim": SHARED_EMBEDDING_DIM,
            "gpt2_lr": LEARNING_RATE_GPT2,
            "cpm_lr": LEARNING_RATE_CPM,
            "alignment_weight": ALIGNMENT_LOSS_WEIGHT,
            "anti_align_weight": ANTI_ALIGNMENT_WEIGHT,
            "contrastive_weight": CONTRASTIVE_LOSS_WEIGHT
        },
        "summary": {"accuracy": accuracy, "avg_loss": avg_loss},
        "gpt2_error_log": gpt2_agent.error_log,
        "cpm_error_log": cpm_agent.error_log,
        "log": log
    }, f, ensure_ascii=False, indent=2)

print(f"✅ Accuracy: {accuracy:.2f}% | Avg Loss: {avg_loss:.4f}")
