import torch
from transformers import AutoTokenizer, GPT2Model
import json
import torch.nn.functional as F
from torch.optim import AdamW

# --- 配置 ---
MODEL_PATH = "/puhome/23063003r/refgame_project/models/gpt2" # 本地 GPT-2 模型路径
# 注意：这里修改为你的 data.json 路径
DATA_FILE = "/hpc2/puhome/23063003r/refgame_project/data/test_data.json" # 模拟数据库文件
D_HIDDEN = 768 # GPT-2的隐藏层维度

# 定义奖励和惩罚值（可以根据实验调整）
REWARD_CORRECT = 0.1 # 猜对时的“奖励”强度 (减小损失)
PENALTY_WRONG = 1.0  # 猜错时的“惩罚”强度 (增大损失)

# --- Agent B (Listener - GPT-2) 视角 ---

# 1. 加载 GPT-2 模型和 Tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = GPT2Model.from_pretrained(MODEL_PATH) # 使用 GPT2Model 获取句子嵌入
model.train() # 设置为训练模式

# 2. 扩展 GPT-2 的 Tokenizer 以支持中文
# 收集所有数据中的中文字符以确保tokenizer覆盖
all_chinese_chars_in_corpus = set()
try:
    with open(DATA_FILE, 'r', encoding='utf-8') as f:
        full_game_data = json.load(f)
    for entry in full_game_data:
        all_chinese_chars_in_corpus.update(list(entry['target_sentence_chinese_raw']))
except Exception as e:
    print(f"❌ 错误加载数据以收集中文符号: {e}")
    # 如果加载失败，使用一个默认的汉字集以防程序中断，但实际训练可能不完整
    all_chinese_chars_in_corpus = set("一个苹果掉到了地上。猫跳到了桌子上。一辆红色的汽车开在街上。狗追球。天空是蓝色的。她在看书。睡沙发。孩子们在公园玩。太阳从东方升起。喜欢听音乐。咖啡很烫。我饿了想吃东西。")


tokenizer.add_special_tokens({'additional_special_tokens': list(all_chinese_chars_in_corpus)})
model.resize_token_embeddings(len(tokenizer)) # 调整 embedding 层大小以适应新词汇表

print(f"✅ GPT-2 tokenizer 已扩展，新的词汇表大小: {len(tokenizer)}")
print(f"✅ GPT-2 模型 Embedding 层已调整。")

# 3. 验证 D_HIDDEN 与模型隐藏层维度
assert D_HIDDEN == model.config.hidden_size, "D_HIDDEN must match GPT-2's hidden_size for direct use."

# 4. 优化器
optimizer = AdamW(model.parameters(), lr=1e-5)

# --- 模拟多轮游戏 ---

# 5. 从模拟数据库加载所有游戏数据
try:
    with open(DATA_FILE, 'r', encoding='utf-8') as f:
        all_game_rounds = json.load(f)
except FileNotFoundError:
    print(f"❌ 错误: 数据库文件 '{DATA_FILE}' 未找到。请创建它。")
    exit()
except Exception as e:
    print(f"❌ 错误加载数据: {e}")
    exit()

print(f"\n--- 准备进行 {len(all_game_rounds)} 轮游戏 ---")

total_loss_sum = 0.0
correct_predictions_count = 0
total_rounds = len(all_game_rounds)

# 循环进行每一轮游戏
for i, game_round in enumerate(all_game_rounds):
    print(f"\n--- 游戏回合 {i + 1}/{total_rounds} ---")
    print(f"🎯 目标中文句子 (CPM '说'): {game_round['target_sentence_chinese_raw']}")
    print(f"📚 候选英文句子 (Agent B 选择): {game_round['candidate_english_sentences_raw']}")
    print(f"✅ 正确索引: {game_round['correct_candidate_index']}")

    # 6. Agent A (CPM 视角) '说' (提供中文句子作为乱码源)
    cpm_spoken_chinese_sentence = game_round['target_sentence_chinese_raw']

    # 7. Agent B (GPT-2) 处理中文乱码输入
    # 拿到 embedding 层，用于比较变化
    embedding_before = model.wte.weight.clone().detach() # wte 是 Word Token Embeddings

    # 将中文句子拆分为字符，并用 GPT-2 的 tokenizer 处理
    inputs_cn_symbolic = tokenizer(cpm_spoken_chinese_sentence, return_tensors="pt")

    # 获取中文乱码序列的语义表示
    outputs_cn_symbolic = model(**inputs_cn_symbolic)
    semantic_vector_B_from_A = outputs_cn_symbolic.last_hidden_state[:, 0, :] # 取第一个token的隐藏状态

    # 8. Agent B 处理英文候选句子
    semantic_vectors_B_candidates = []
    for eng_sentence in game_round['candidate_english_sentences_raw']:
        inputs_en = tokenizer(eng_sentence, return_tensors="pt")
        outputs_en = model(**inputs_en)
        vec_en = outputs_en.last_hidden_state[:, 0, :]
        semantic_vectors_B_candidates.append(vec_en)

    semantic_vectors_B_candidates = torch.cat(semantic_vectors_B_candidates, dim=0)

    # 9. Agent B 猜测 (计算相似度并预测)
    similarities = F.cosine_similarity(semantic_vector_B_from_A, semantic_vectors_B_candidates, dim=1)
    predicted_index = torch.argmax(similarities).item()

    print(f"🤔 相似度得分 (越高越相似): {similarities.tolist()}")
    print(f"🔮 Agent B 猜测的索引: {predicted_index}")

    # 10. 反馈与权重更新 (Agent B 学习)
    correct_index_tensor = torch.tensor([game_round['correct_candidate_index']], device=similarities.device)

    # 引入显式的奖励/惩罚
    # CrossEntropyLoss的reduction='mean' (默认) 或 'sum'
    # 为了保持损失量级可控，我们保持默认的'mean'，然后根据结果调整。
    base_loss = F.cross_entropy(similarities.unsqueeze(0), correct_index_tensor)

    if predicted_index == game_round['correct_candidate_index']:
        loss = base_loss * (1 - REWARD_CORRECT)
        outcome_message = f"🎉 Agent B 猜对啦！损失调整系数: {(1 - REWARD_CORRECT):.2f}"
        is_correct = True
    else:
        loss = base_loss * PENALTY_WRONG
        outcome_message = f"💔 Agent B 猜错了！损失调整系数: {PENALTY_WRONG:.2f}"
        is_correct = False

    print(outcome_message)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # 11. 比较 Embedding 变化 (并累加统计)
    embedding_after = model.wte.weight.detach()
    diff = torch.norm(embedding_after - embedding_before).item()

    total_loss_sum += loss.item()
    if is_correct:
        correct_predictions_count += 1
    if (i + 1) % 100 == 0 or i == 0 or (i + 1) == total_rounds:
        print(f"📉 本轮游戏最终损失: {loss.item():.4f}")
        print(f"🔍 Embedding (word token embeddings) 改变量: {diff:.6f}")
        print(f"✨ Agent B 最终猜测结果: {is_correct}")

# --- 10 轮游戏结束，汇总结果 ---
print("\n--- 10 轮游戏总结 ---")
print(f"总轮数: {total_rounds}")
print(f"平均损失: {total_loss_sum / total_rounds:.4f}")
print(f"猜对轮数: {correct_predictions_count}")
print(f"准确率: {(correct_predictions_count / total_rounds * 100):.2f}%")
