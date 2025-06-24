import torch
from transformers import AutoTokenizer, GPT2Model
import json
import torch.nn.functional as F
from torch.optim import AdamW
import os # 导入 os 模块用于路径操作

# --- 配置 ---
MODEL_PATH = "/ubsnhome/23063003r/refgame_project/models/gpt2"
DATA_FILE = "/ubsnhome/23063003r/refgame_project/data/generated_game_data.json"
OUTPUT_DIR = "/ubsnhome/23063003r/refgame_project/output/"
D_HIDDEN = 1600 
# D_HIDDEN = 768 # GPT-2的隐藏层维度

# 定义奖励和惩罚值（可以根据实验调整）
REWARD_CORRECT = 0.1 # 猜对时的“奖励”强度 (减小损失)
PENALTY_WRONG = 1.0  # 猜错时的“惩罚”强度 (增大损失)

# --- Agent B (Listener - GPT-2) 视角 ---

# 1. 加载 GPT-2 模型和 Tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = GPT2Model.from_pretrained(MODEL_PATH) # 使用 GPT2Model 获取句子嵌入
model.train() # 设置为训练模式

# --- IMPORTANT: Set pad_token for GPT-2 tokenizer ---
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# 2. 扩展 GPT-2 的 Tokenizer 以支持中文
all_chinese_chars_in_corpus = set()
try:
    with open(DATA_FILE, 'r', encoding='utf-8') as f:
        full_game_data = json.load(f)
    for entry in full_game_data:
        all_chinese_chars_in_corpus.update(list(entry['target_sentence_chinese_raw']))
except Exception as e:
    print(f"❌ 错误加载数据以收集中文符号: {e}")
    all_chinese_chars_in_corpus = set("一个苹果掉到了地上。猫跳到了桌子上。一辆红色的汽车开在街上。狗追球。天空是蓝色的。她在看书。睡沙发。孩子们在公园玩。太阳从东方升起。喜欢听音乐。咖啡很烫。我饿了想吃东西。")


tokenizer.add_special_tokens({'additional_special_tokens': list(all_chinese_chars_in_corpus)})
model.resize_token_embeddings(len(tokenizer)) # 调整 embedding 层大小以适应新词汇表

print(f"✅ GPT-2 tokenizer 已扩展，新的词汇表大小: {len(tokenizer)}")
print(f"✅ GPT-2 模型 Embedding 层已调整。")

# 3. 验证 D_HIDDEN 与模型隐藏层维度
# <<< --- IMPORTANT: 这里需要验证模型实际的 hidden_size ---
# model.config.hidden_size 应该就是 n_embd，也就是 1600
assert D_HIDDEN == model.config.hidden_size, f"D_HIDDEN ({D_HIDDEN}) must match GPT-2's hidden_size ({model.config.hidden_size}) for direct use."

# 4. 优化器
optimizer = AdamW(model.parameters(), lr=1e-5)

# --- 准备记录数据 ---
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)
    print(f"创建输出目录: {OUTPUT_DIR}")

per_round_metrics = [] # 存储每轮的详细指标

# --- 模拟多轮游戏 ---
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
print("total_rounds: ",total_rounds)

# 循环进行每一轮游戏
for i, game_round in enumerate(all_game_rounds):
    if (i + 1) % 100 == 0 or i == 0 or (i + 1) == total_rounds:
        print(f"\n--- 游戏回合 {i + 1}/{total_rounds} ---")
        print(f"🎯 目标中文句子 (CPM '说'): {game_round['target_sentence_chinese_raw']}")
        print(f"📚 候选英文句子 (Agent B 选择): {game_round['candidate_english_sentences_raw']}")
        print(f"✅ 正确索引: {game_round['correct_candidate_index']}")

    # 6. Agent A (CPM 视角) '说' (提供中文句子作为乱码源)
    cpm_spoken_chinese_sentence = game_round['target_sentence_chinese_raw']

    # 7. Agent B (GPT-2) 处理中文乱码输入
    embedding_before = model.wte.weight.clone().detach() # wte 是 Word Token Embeddings

    # --- 修正中文乱码输入 Tokenizer 调用 ---
    # `tokenizer()` 期望字符串列表进行批处理。对于单句，也要包装成列表。
    inputs_cn_symbolic = tokenizer([cpm_spoken_chinese_sentence], return_tensors="pt", padding=True, truncation=True)
    outputs_cn_symbolic = model(**inputs_cn_symbolic)
    semantic_vector_B_from_A = outputs_cn_symbolic.last_hidden_state[:, 0, :] # 取第一个token的隐藏状态

    # 8. Agent B 处理英文候选句子
    # --- 修正英文候选句子 Tokenizer 调用 ---
    # 确保 candidates 列表作为批次输入给 tokenizer
    inputs_en_candidates = tokenizer(game_round['candidate_english_sentences_raw'], return_tensors="pt", padding=True, truncation=True)

    semantic_vectors_B_candidates = [] # 列表清空，将直接从 outputs_en_candidates 获取

    # 将输入移动到模型设备
    inputs_cn_symbolic.to(model.device) # Move to device inside loop
    inputs_en_candidates.to(model.device) # Move to device inside loop

    outputs_en_candidates = model(**inputs_en_candidates)
    # 取每个候选句子的第一个token的隐藏状态作为句子表示
    semantic_vectors_B_candidates_batch = outputs_en_candidates.last_hidden_state[:, 0, :]

    # 此时 semantic_vectors_B_candidates_batch 的形状是 (num_candidates, D_HIDDEN)
    # 确保其维度符合 listener_mse_reciprocal_loss 的 (batch_size, num_candidates, D_HIDDEN)
    # 因为这里是单样本循环，batch_size=1，所以 unsqueeze(0)
    semantic_vectors_B_candidates = semantic_vectors_B_candidates_batch.unsqueeze(0)


    # 9. Agent B 猜测 (计算相似度并预测)
    # semantic_vector_B_from_A: (1, D_HIDDEN)
    # semantic_vectors_B_candidates: (1, num_candidates, D_HIDDEN)
    # 余弦相似度计算需要调整 dim 参数
    similarities = F.cosine_similarity(
        semantic_vector_B_from_A.unsqueeze(1), # (1, 1, D_HIDDEN) for broadcasting
        semantic_vectors_B_candidates,         # (1, num_candidates, D_HIDDEN)
        dim=2 # 沿着 D_HIDDEN 维度计算相似度
    ).squeeze(1) # 结果形状 (1, num_candidates) -> squeeze(1) to (num_candidates,)

    predicted_index = torch.argmax(similarities).item()

    if (i + 1) % 100 == 0 or i == 0 or (i + 1) == total_rounds:
        print(f"🤔 相似度得分 (越高越相似): {similarities.tolist()}")
        print(f"🔮 Agent B 猜测的索引: {predicted_index}")

    # 10. 反馈与权重更新 (Agent B 学习)
    # --- 修正 correct_index_tensor dtype 和 device ---
    correct_index_tensor = torch.tensor([game_round['correct_candidate_index']], device=model.device, dtype=torch.long)

    # --- Listener MSE Reciprocal Loss ---
    # semantic_vector_B_from_A: (1, D_HIDDEN)
    # semantic_vectors_B_candidates: (1, num_candidates, D_HIDDEN)
    # correct_index_tensor: (1,) (already long)

    # 注意：listener_mse_reciprocal_loss 期望的 input_A 和 candidates_B 形状是 (batch_size, D_HIDDEN) 和 (batch_size, num_candidates, D_HIDDEN)
    # 在这里，由于是单样本循环，它们已经符合这个批次形状
    base_loss = listener_mse_reciprocal_loss(
        semantic_vector_B_from_A,
        semantic_vectors_B_candidates,
        correct_index_tensor
    )

    is_correct = (predicted_index == game_round['correct_candidate_index']) # 比较Python int
    if is_correct:
        loss = base_loss * (1 - REWARD_CORRECT)
        outcome_message = f"🎉 Agent B 猜对啦！损失调整系数: {(1 - REWARD_CORRECT):.2f}"
    else:
        loss = base_loss * PENALTY_WRONG
        outcome_message = f"💔 Agent B 猜错了！损失调整系数: {PENALTY_WRONG:.2f}"

    print(outcome_message)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # 11. 比较 Embedding 变化
    embedding_after = model.wte.weight.detach()
    diff = torch.norm(embedding_after - embedding_before).item()

    total_loss_sum += loss.item()
    if is_correct:
        correct_predictions_count += 1
    if (i + 1) % 100 == 0 or i == 0 or (i + 1) == total_rounds:
        print(f"📉 本轮游戏最终损失: {loss.item():.4f}")
        print(f"🔍 Embedding (word token embeddings) 改变量: {diff:.6f}")
        print(f"✨ Agent B 最终猜测结果: {is_correct}")

    # --- 记录本轮数据 ---
    round_data = {
        "round_idx": i + 1,
        "chinese_sentence": game_round['target_sentence_chinese_raw'],
        "correct_english_sentence": game_round['correct_english_sentence_raw'],
        "candidate_english_sentences": game_round['candidate_english_sentences_raw'],
        "correct_candidate_idx": game_round['correct_candidate_index'],
        "predicted_index": predicted_index,
        "similarities": similarities.tolist(), # 转换为列表以便JSON序列化
        "is_correct_prediction": is_correct,
        "base_loss": base_loss.item(),
        "final_loss": loss.item(),
        "embedding_diff_norm": diff
    }
    per_round_metrics.append(round_data)

# --- 训练结束，汇总结果并保存 ---
print("\n--- 训练总结 ---")
final_accuracy_percentage = (correct_predictions_count / total_rounds * 100) if total_rounds > 0 else 0
print(f"总轮数: {total_rounds}")
print(f"平均损失: {total_loss_sum / total_rounds:.4f}")
print(f"猜对轮数: {correct_predictions_count}")
print(f"准确率: {final_accuracy_percentage:.2f}%")

# --- 保存结果到 JSON 文件 ---
summary_metrics = {
    "total_rounds": total_rounds,
    "final_average_loss": total_loss_sum / total_rounds,
    "final_correct_count": correct_predictions_count,
    "final_accuracy_percentage": final_accuracy_percentage
}

output_data = {
    "summary_metrics": summary_metrics,
    "per_round_metrics": per_round_metrics
}

output_file_path = os.path.join(OUTPUT_DIR, "training_results_15000.json")
with open(output_file_path, 'w', encoding='utf-8') as f:
    json.dump(output_data, f, ensure_ascii=False, indent=2)

print(f"\n🎉 训练结果已保存到: {output_file_path}")


# --- Listener MSE Reciprocal Loss 函数定义 (需要在主脚本中提前定义或导入) ---
def listener_mse_reciprocal_loss(
    semantic_vector_from_agent_A: torch.Tensor,
    semantic_vectors_candidates_B: torch.Tensor,
    correct_candidate_index: torch.Tensor, # 确保这里是 LongTensor
    epsilon: float = 1e-8 # 用于数值稳定的小常数
) -> torch.Tensor:
    """
    计算 Listener Loss，采用论文中 (EMERGENT TRANSLATION IN MULTI-AGENT COMMUNICATION)
    描述的 MSE 倒数对数形式。
    Args:
        semantic_vector_from_agent_A (torch.Tensor): Agent A（中文乱码）的语义向量。形状: (batch_size, D_HIDDEN)
        semantic_vectors_candidates_B (torch.Tensor): Agent B 候选英文句子的语义向量集合。形状: (batch_size, num_candidates, D_HIDDEN)
        correct_candidate_index (torch.Tensor): 正确候选句子的索引。形状: (batch_size,)
        epsilon (float): 用于数值稳定的小常数。
    Returns:
        torch.Tensor: 计算出的损失值。
    """
    # print(f"DEBUG: listener_mse_reciprocal_loss input shapes & dtypes:")
    # print(f"  semantic_vector_from_agent_A: {semantic_vector_from_agent_A.shape}, {semantic_vector_from_agent_A.dtype}")
    # print(f"  semantic_vectors_candidates_B: {semantic_vectors_candidates_B.shape}, {semantic_vectors_candidates_B.dtype}")
    # print(f"  correct_candidate_index: {correct_candidate_index.shape}, {correct_candidate_index.dtype}") # 确保是 torch.long

    expanded_vector_A = semantic_vector_from_agent_A.unsqueeze(1)
    # print(f"DEBUG: expanded_vector_A shape: {expanded_vector_A.shape}")

    squared_diff = (expanded_vector_A - semantic_vectors_candidates_B).pow(2)
    # print(f"DEBUG: squared_diff shape: {squared_diff.shape}")

    mse_distances = squared_diff.sum(dim=-1)
    # print(f"DEBUG: mse_distances shape: {mse_distances.shape}")

    logits = 1 / (mse_distances + epsilon)
    # print(f"DEBUG: logits shape: {logits.shape}")

    loss = F.cross_entropy(logits, correct_candidate_index) # correct_candidate_index 必须是 torch.long

    return loss
