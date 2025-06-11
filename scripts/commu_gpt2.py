import torch
from transformers import AutoTokenizer, GPT2Model, AdamW
import json
import torch.nn.functional as F

# --- 配置 ---
MODEL_PATH = "/puhome/23063003r/refgame_project/models/gpt2" # 本地 GPT-2 模型路径
DATA_FILE = "/puhome/23063003r/refgame_project/data/test_data.json" # 模拟数据库文件
D_HIDDEN = 768 # GPT-2的隐藏层维度，也是我们共享语义空间的维度

# --- Agent B (Listener - GPT-2) 视角 ---

# 1. 加载 GPT-2 模型和 Tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = GPT2Model.from_pretrained(MODEL_PATH) # 使用 GPT2Model 获取句子嵌入，而不是 AutoModelForCausalLM
model.train() # 设置为训练模式

# 2. 扩展 GPT-2 的 Tokenizer 以支持中文
# 假设我们已经预先收集了所有可能的中文汉字
# 实际项目中，你需要从你的中文语料中提取所有不重复的汉字并添加到这里
all_chinese_chars_in_corpus = set("一个苹果掉到了地上。猫跳到了桌子上。一辆红色的汽车开在街上。") # 示例汉字
tokenizer.add_special_tokens({'additional_special_tokens': list(all_chinese_chars_in_corpus)})
model.resize_token_embeddings(len(tokenizer)) # 调整 embedding 层大小以适应新词汇表

print(f"✅ GPT-2 tokenizer 已扩展，新的词汇表大小: {len(tokenizer)}")
print(f"✅ GPT-2 模型 Embedding 层已调整。")

# 3. 定义投影层 (将GPT-2的输出映射到共享语义空间)
# 共享语义空间维度与GPT-2的隐藏层维度相同，因此这里可以理解为恒等映射或微调。
# 也可以显式定义一个线性层：torch.nn.Linear(model.config.hidden_size, D_HIDDEN)
# 但为了简化，我们直接使用GPT-2的输出作为共享语义向量。
# 如果隐藏层大小和共享空间维度不同，则需要此层。
# 这里 D_HIDDEN 应该与 model.config.hidden_size 保持一致，对于gpt2是768
assert D_HIDDEN == model.config.hidden_size, "D_HIDDEN must match GPT-2's hidden_size for direct use."

# 4. 优化器
optimizer = AdamW(model.parameters(), lr=1e-5)

# --- 模拟一轮游戏 ---

# 5. 从模拟数据库加载一轮游戏数据
try:
    with open(DATA_FILE, 'r', encoding='utf-8') as f:
        game_data_list = json.load(f)
    game_round = game_data_list[0] # 取第一轮游戏数据
except FileNotFoundError:
    print(f"❌ 错误: 数据库文件 '{DATA_FILE}' 未找到。请创建它。")
    exit()
except Exception as e:
    print(f"❌ 错误加载数据: {e}")
    exit()

print("\n--- 模拟游戏开始 ---")
print(f"🎯 目标中文句子 (CPM '说'): {game_round['target_sentence_chinese_raw']}")
print(f"📚 候选英文句子 (Agent B 选择): {game_round['candidate_english_sentences_raw']}")
print(f"✅ 正确索引: {game_round['correct_candidate_index']}")

# 6. Agent A (CPM 视角) '说' (提供中文句子作为乱码源)
# 我们扮演CPM，提供目标中文句子。GPT-2将把它视为乱码。
cpm_spoken_chinese_sentence = game_round['target_sentence_chinese_raw']

# 7. Agent B (GPT-2) 处理中文乱码输入
# 拿到 embedding 层，用于比较变化
embedding_before = model.wte.weight.clone().detach() # wte 是 Word Token Embeddings

# 将中文句子拆分为字符，并用 GPT-2 的 tokenizer 处理
# tokenizer会自动识别每个汉字为添加的特殊token
inputs_cn_symbolic = tokenizer(cpm_spoken_chinese_sentence, return_tensors="pt")

# 获取中文乱码序列的语义表示
# 使用 model(input_ids).last_hidden_state 来获取 Encoder 输出
outputs_cn_symbolic = model(**inputs_cn_symbolic)
# 这里我们取序列的第一个token的隐藏状态作为句子表示
# (对于句子级别的任务，通常会这么做，或者进行平均池化)
semantic_vector_B_from_A = outputs_cn_symbolic.last_hidden_state[:, 0, :]
# 注意：这里没有额外的投影层，因为 D_HIDDEN == model.config.hidden_size

print(f"\n Agent B 接收到中文乱码序列，将其编码为语义向量 (形状: {semantic_vector_B_from_A.shape})")

# 8. Agent B 处理英文候选句子
# 对每个英文候选句子进行编码，并获取其语义表示
semantic_vectors_B_candidates = []
for eng_sentence in game_round['candidate_english_sentences_raw']:
    inputs_en = tokenizer(eng_sentence, return_tensors="pt")
    outputs_en = model(**inputs_en)
    vec_en = outputs_en.last_hidden_state[:, 0, :]
    semantic_vectors_B_candidates.append(vec_en)

semantic_vectors_B_candidates = torch.cat(semantic_vectors_B_candidates, dim=0)
print(f" Agent B 将英文候选句子编码为语义向量 (形状: {semantic_vectors_B_candidates.shape})")


# 9. Agent B 猜测 (计算相似度并预测)
# semantic_vector_B_from_A: (1, D_HIDDEN)
# semantic_vectors_B_candidates: (num_candidates, D_HIDDEN)
similarities = F.cosine_similarity(semantic_vector_B_from_A, semantic_vectors_B_candidates, dim=1)
predicted_index = torch.argmax(similarities).item()

print(f"\n 相似度得分 (越高越相似): {similarities.tolist()}")
print(f" Agent B 猜测的索引: {predicted_index}")

# 10. 反馈与权重更新 (Agent B 学习)
correct_index_tensor = torch.tensor([game_round['correct_candidate_index']], device=similarities.device)

# 使用交叉熵损失，将相似度视为 logits
# 注意：CrossEntropyLoss期望的输入是未经归一化的logits，这里我们直接用cosine_similarity作为logits
# 如果相似度值域不是0到1，或者你需要更严格的分类，可能需要调整或添加线性层。
# 但对于简单的排名任务，直接用相似度作为“得分”并计算交叉熵是可行的。
loss = F.cross_entropy(similarities.unsqueeze(0), correct_index_tensor)

optimizer.zero_grad() # 清零梯度
loss.backward()      # 反向传播计算梯度
optimizer.step()     # 更新模型权重

# 11. 比较 Embedding 变化
embedding_after = model.wte.weight.detach()
diff = torch.norm(embedding_after - embedding_before).item()

print(f"\n 权重更新完成。")
print(f" 本轮游戏损失: {loss.item():.4f}")
print(f" Embedding (word token embeddings) 改变量: {diff:.6f}")
print(f" Agent B 猜对了吗？: {predicted_index == game_round['correct_candidate_index']}")
