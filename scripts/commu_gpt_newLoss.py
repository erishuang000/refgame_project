import torch
from transformers import AutoTokenizer, GPT2Model
import json
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader # 引入DataLoader
import os
import random # 引入random用于shuffle数据集

# --- 1. 配置管理 ---
class Config:
    MODEL_PATH = "/puhome/23063003r/refgame_project/models/gpt2"
    DATA_FILE = "/puhome/23063003r/refgame_project/data/generated_game_data.json"
    OUTPUT_DIR = "/puhome/23063003r/refgame_project/output/"
    D_HIDDEN = 768  # GPT-2的隐藏层维度

    # Loss 奖励/惩罚系数
    REWARD_CORRECT = 0.1
    PENALTY_WRONG = 1.0

    LEARNING_RATE = 1e-5
    BATCH_SIZE = 1 # 为了简化，目前仍按单样本处理，未来可改为更大
    # 注意：如果BATCH_SIZE > 1，则DataLoader的collate_fn需要处理padding

# --- 2. Listener Loss 函数 ---
def listener_mse_reciprocal_loss(
    semantic_vector_from_agent_A: torch.Tensor,
    semantic_vectors_candidates_B: torch.Tensor,
    correct_candidate_index: torch.Tensor,
    epsilon: float = 1e-8 # 用于数值稳定的小常数
) -> torch.Tensor:
    """
    计算 Listener Loss，采用论文中 (EMERGENT TRANSLATION IN MULTI-AGENT COMMUNICATION)
    描述的 MSE 倒数对数形式。
    Args:
        semantic_vector_from_agent_A (torch.Tensor): Agent A（中文乱码）的语义向量。
                                                     期望形状: (batch_size, D_HIDDEN)
        semantic_vectors_candidates_B (torch.Tensor): Agent B 候选英文句子的语义向量集合。
                                                      期望形状: (batch_size, num_candidates, D_HIDDEN)
        correct_candidate_index (torch.Tensor): 正确候选句子的索引。
                                                期望形状: (batch_size,)
        epsilon (float): 用于数值稳定的小常数，防止除以零。
    Returns:
        torch.Tensor: 计算出的损失值。
    """
    print(f"DEBUG: semantic_vector_from_agent_A shape: {semantic_vector_from_agent_A.shape}, dtype: {semantic_vector_from_agent_A.dtype}")
    print(f"DEBUG: semantic_vectors_candidates_B shape: {semantic_vectors_candidates_B.shape}, dtype: {semantic_vectors_candidates_B.dtype}")
    print(f"DEBUG: correct_candidate_index shape: {correct_candidate_index.shape}, dtype: {correct_candidate_index.dtype}")

    # 确保 Agent A 向量维度可以广播到候选向量
    # expanded_vector_A 形状变为 (batch_size, 1, D_HIDDEN)
    expanded_vector_A = semantic_vector_from_agent_A.unsqueeze(1)
    print(f"DEBUG: expanded_vector_A shape: {expanded_vector_A.shape}")

    # 计算 (E_EN^B(m_hat) - E_IMG^B(i_k))^2，即均方差的平方部分
    # 结果形状: (batch_size, num_candidates, D_HIDDEN)
    squared_diff = (expanded_vector_A - semantic_vectors_candidates_B).pow(2)
    print(f"DEBUG: squared_diff shape: {squared_diff.shape}")

    # 对特征维度求和，得到每个候选的 MSE 距离
    # 结果形状: (batch_size, num_candidates)
    mse_distances = squared_diff.sum(dim=-1)
    print(f"DEBUG: mse_distances shape: {mse_distances.shape}")

    # 论文中的 logits 是 MSE 的倒数，添加 epsilon 避免除零
    logits = 1 / (mse_distances + epsilon)
    print(f"DEBUG: logits shape: {logits.shape}")

    # 损失函数是 -log(softmax(logits))，F.cross_entropy 内部包含了 log_softmax
    # F.cross_entropy 期望 logits (N, C) 和 targets (N)
    # N 是 batch_size, C 是 num_candidates
    loss = F.cross_entropy(logits, correct_candidate_index)
    print(f"DEBUG: Final loss calculated.")

    return loss

# --- 3. Agent B (Listener) 模型类 ---
class AgentBListener(torch.nn.Module):
    def __init__(self, model_path, all_chinese_chars, hidden_dim):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        # --- 添加这一行来设置 pad_token ---
        if self.tokenizer.pad_token is None: # 检查是否已经有pad_token，避免重复设置
            self.tokenizer.pad_token = self.tokenizer.eos_token
        # --- 结束添加 ---
        self.model = GPT2Model.from_pretrained(model_path)

        # 扩展 tokenizer 并调整 embedding 层
        self.tokenizer.add_special_tokens({'additional_special_tokens': list(all_chinese_chars)})
        self.model.resize_token_embeddings(len(self.tokenizer))

        # 验证维度
        assert hidden_dim == self.model.config.hidden_size, \
            "D_HIDDEN must match GPT-2's hidden_size for direct use as semantic vector."

        print(f"✅ AgentB: GPT-2 tokenizer 已扩展，新的词汇表大小: {len(self.tokenizer)}")
        print(f"✅ AgentB: GPT-2 模型 Embedding 层已调整。")

    def forward(self, inputs_cn_symbolic_raw, inputs_en_candidates_raw, device):
        print(f"DEBUG_FORWARD_STEP2: inputs_cn_symbolic_raw (from DataLoader, before [0]): {inputs_cn_symbolic_raw}")
        print(f"DEBUG_FORWARD_STEP2: inputs_en_candidates_raw (from DataLoader, before [0]): {inputs_en_candidates_raw}")

        # 1. 处理中文乱码输入
        # inputs_cn_symbolic_raw 是 ['一个苹果掉到了地上。'] (list of 1 string)
        # 确保这里只取第一个元素，因为它仍然是 DataLoader 包装过的
        cpm_sentence_for_tokenizer = inputs_cn_symbolic_raw[0]
        inputs_cn_symbolic = self.tokenizer(cpm_sentence_for_tokenizer, return_tensors="pt", padding=True, truncation=True).to(device)
        outputs_cn_symbolic = self.model(**inputs_cn_symbolic)
        semantic_vector_B_from_A = outputs_cn_symbolic.last_hidden_state[:, 0, :]


        # 2. Agent B 处理英文候选句子 (再次仔细检查这里)
        # inputs_en_candidates_raw 是 [['候选1', '候选2', '候选3']] 或者 [('候选1',)] 这样的格式
        # 我们需要提取出 ['候选1', '候选2', '候选3']
        english_sentences_list = inputs_en_candidates_raw[0]
        print(f"DEBUG_FORWARD_STEP2: english_sentences_list (after [0] from inputs_en_candidates_raw): {english_sentences_list}, type: {type(english_sentences_list)}")
        # 期望这里是一个包含3个字符串的列表，如 ['An apple...', 'The cat...', 'A red...']

        # 检查是否是元组
        if isinstance(english_sentences_list, tuple):
            print(f"警告: english_sentences_list 是一个元组！将其转换为列表。")
            english_sentences_list = list(english_sentences_list)

        # 检查列表长度
        if not isinstance(english_sentences_list, list) or len(english_sentences_list) < 3: # 假设 num_candidates 是3
            print(f"严重错误: english_sentences_list 不是列表或长度不足3！实际: {english_sentences_list}")
            # 这里可以直接 raise TypeError 或 AssertionError 来提前报错
            raise ValueError(f"候选英文句子数量不足或格式错误: {english_sentences_list}")


        inputs_en_candidates_tokenized = self.tokenizer(
            english_sentences_list, # 传入字符串列表
            return_tensors="pt",
            padding=True,
            truncation=True
        ).to(device)

        print(f"DEBUG_TOKENIZER_AFTER_FIX_STEP2: inputs_en_candidates_tokenized input_ids shape: {inputs_en_candidates_tokenized['input_ids'].shape}")
        # 期望形状: (num_candidates, max_seq_len), 例如 (3, max_len)

        outputs_en_candidates = self.model(**inputs_en_candidates_tokenized)

        semantic_vectors_B_candidates = outputs_en_candidates.last_hidden_state[:, 0, :]

        semantic_vectors_B_candidates = semantic_vectors_B_candidates.unsqueeze(0) # 变为 (1, num_candidates, D_HIDDEN)

        return semantic_vector_B_from_A, semantic_vectors_B_candidates, embedding_before
        
    def get_embedding_after(self):
        return self.model.wte.weight.detach()

# --- 4. 游戏数据加载器 ---
class GameDataset(Dataset):
    def __init__(self, data_file):
        with open(data_file, 'r', encoding='utf-8') as f:
            self.data = json.load(f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

# --- 5. 游戏训练循环 ---
class GameTrainer:
    def __init__(self, config: Config, model: AgentBListener, optimizer: AdamW, device: torch.device):
        self.config = config
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.per_round_metrics = []
        self.total_loss_sum = 0.0
        self.correct_predictions_count = 0

    def train_one_round(self, game_round: dict, round_idx: int, total_rounds: int): # game_round 现在是原始字典
        self.optimizer.zero_grad()

        cpm_spoken_chinese_sentence = game_round['target_sentence_chinese_raw']

        # 1. Agent B (GPT-2) 处理中文乱码输入及英文候选句子
        # inputs_cn_symbolic_raw 应该是 ['句子'] (列表包含一个字符串)
        # inputs_en_candidates_raw 应该是 [['候选1', '候选2', '候选3']] (列表包含一个列表)

        # 将字符串包装成列表，因为 forward 期望 batch_size 个元素
        semantic_vector_B_from_A, semantic_vectors_B_candidates, embedding_before = \
            self.model(
                [cpm_spoken_chinese_sentence], # 传入 list of string
                [game_round['candidate_english_sentences_raw']], # 传入 list of list of string
                self.device
            )

        # 3. Agent B 猜测 (计算相似度并预测)
        # similarities: (1, num_candidates) -> squeeze(0) to (num_candidates,)
        similarities = F.cosine_similarity(
            semantic_vector_B_from_A,
            semantic_vectors_B_candidates.squeeze(0), # squeeze to match (num_candidates, D_HIDDEN) for comparison
            dim=1 # Compare along the D_HIDDEN dimension
        )
        predicted_index = torch.argmax(similarities).item()

        print(f"🤔 相似度得分 (越高越相似): {similarities.tolist()}")
        print(f"🔮 Agent B 猜测的索引: {predicted_index}")

        # 4. 反馈与权重更新 (Agent B 学习)
        correct_index_tensor = torch.tensor([game_round['correct_candidate_index']], device=self.device)

        # 使用 Listener Loss 函数
        base_loss = listener_mse_reciprocal_loss(
            semantic_vector_B_from_A,
            semantic_vectors_B_candidates, # 期望 (batch_size, num_candidates, D_HIDDEN)
            correct_index_tensor
        )

        is_correct = (predicted_index == game_round['correct_candidate_index'])
        if is_correct:
            loss = base_loss * (1 - self.config.REWARD_CORRECT)
            outcome_message = f"🎉 Agent B 猜对啦！损失调整系数: {(1 - self.config.REWARD_CORRECT):.2f}"
        else:
            loss = base_loss * self.config.PENALTY_WRONG
            outcome_message = f"💔 Agent B 猜错了！损失调整系数: {self.config.PENALTY_WRONG:.2f}"

        print(outcome_message)

        loss.backward()
        self.optimizer.step()

        # 5. 比较 Embedding 变化
        embedding_after = self.model.get_embedding_after()
        diff = torch.norm(embedding_after - embedding_before).item()

        self.total_loss_sum += loss.item()
        if is_correct:
            self.correct_predictions_count += 1

        print(f"📉 本轮游戏最终损失: {loss.item():.4f}")
        print(f"🔍 Embedding (word token embeddings) 改变量: {diff:.6f}")
        print(f"✨ Agent B 最终猜测结果: {is_correct}")

        # 记录本轮数据
        round_data = {
            "round_idx": round_idx,
            "chinese_sentence": game_round['target_sentence_chinese_raw'],
            "correct_english_sentence": game_round['correct_english_sentence_raw'],
            "candidate_english_sentences": game_round['candidate_english_sentences_raw'],
            "correct_candidate_idx": game_round['correct_candidate_index'],
            "predicted_index": predicted_index,
            "similarities": similarities.tolist(),
            "is_correct_prediction": is_correct,
            "base_loss": base_loss.item(),
            "final_loss": loss.item(),
            "embedding_diff_norm": diff
        }
        self.per_round_metrics.append(round_data)

    def train(self):
        game_dataset = GameDataset(self.config.DATA_FILE)
        data_loader = DataLoader(game_dataset, batch_size=self.config.BATCH_SIZE, shuffle=False) # shuffle=True for real training

        self.model.model.train() # Set model to training mode
        total_rounds = len(data_loader)
        print(f"\n--- 准备进行 {total_rounds} 轮游戏 ---")

        for i, raw_game_round_from_dataloader in enumerate(data_loader): # DataLoader 返回原始字典的批次
            # raw_game_round_from_dataloader['target_sentence_chinese_raw'] 会是一个 (batch_size,) 的元组或列表
            # raw_game_round_from_dataloader['candidate_english_sentences_raw'] 会是一个 (batch_size, num_candidates) 的元组的元组或列表的列表

            # 由于目前 batch_size=1, DataLoader 会将每个字段的值包装成一个元组或包含一个元素的列表
            # 例如，{'target_sentence_chinese_raw': ('你看起来像一个聪明人。',), ...}
            # 所以我们需要从这个包装中取出原始值
            single_game_round = {
                'target_sentence_chinese_raw': raw_game_round_from_dataloader['target_sentence_chinese_raw'][0],
                'correct_english_sentence_raw': raw_game_round_from_dataloader['correct_english_sentence_raw'][0],
                'candidate_english_sentences_raw': raw_game_round_from_dataloader['candidate_english_sentences_raw'][0], # 这是正确的，得到 ['S1', 'S2', 'S3']
                'correct_candidate_index': raw_game_round_from_dataloader['correct_candidate_index'][0].item() # 转换为Python int
            }

            print(f"\n--- 游戏回合 {i + 1}/{total_rounds} ---")
            print(f"🎯 目标中文句子 (CPM '说'): {single_game_round['target_sentence_chinese_raw']}")
            print(f"📚 候选英文句子 (Agent B 选择): {single_game_round['candidate_english_sentences_raw']}")
            print(f"✅ 正确索引: {single_game_round['correct_candidate_index']}")

            self.train_one_round(single_game_round, i + 1, total_rounds) # 传入解包后的字典


        # --- 训练结束，汇总结果并保存 ---
        print("\n--- 训练总结 ---")
        final_accuracy_percentage = (self.correct_predictions_count / total_rounds * 100) if total_rounds > 0 else 0
        print(f"总轮数: {total_rounds}")
        print(f"平均损失: {self.total_loss_sum / total_rounds:.4f}")
        print(f"猜对轮数: {self.correct_predictions_count}")
        print(f"准确率: {final_accuracy_percentage:.2f}%")

        # --- 保存结果到 JSON 文件 ---
        summary_metrics = {
            "total_rounds": total_rounds,
            "final_average_loss": self.total_loss_sum / total_rounds,
            "final_correct_count": self.correct_predictions_count,
            "final_accuracy_percentage": final_accuracy_percentage
        }

        output_data = {
            "summary_metrics": summary_metrics,
            "per_round_metrics": self.per_round_metrics
        }

        # 确保输出目录存在
        if not os.path.exists(self.config.OUTPUT_DIR):
            os.makedirs(self.config.OUTPUT_DIR)

        output_file_path = os.path.join(self.config.OUTPUT_DIR, "training_results_newLoss.json")
        with open(output_file_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)

        print(f"\n🎉 训练结果已保存到: {output_file_path}")


# --- 主程序入口 ---
if __name__ == "__main__":
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 使用设备: {device}")

    # 1. 收集所有中文字符以扩展tokenizer
    all_chinese_chars_in_corpus = set()
    try:
        with open(Config.DATA_FILE, 'r', encoding='utf-8') as f:
            temp_game_data = json.load(f)
        for entry in temp_game_data:
            all_chinese_chars_in_corpus.update(list(entry['target_sentence_chinese_raw']))
    except Exception as e:
        print(f"❌ 错误加载数据以收集中文符号，使用默认集: {e}")
        all_chinese_chars_in_corpus = set("一个苹果掉到了地上。猫跳到了桌子上。一辆红色的汽车开在街上。狗追球。天空是蓝色的。她在看书。睡沙发。孩子们在公园玩。太阳从东方升起。喜欢听音乐。咖啡很烫。我饿了想吃东西。")


    # 2. 初始化 Agent B 模型
    agent_b_model = AgentBListener(Config.MODEL_PATH, all_chinese_chars_in_corpus, Config.D_HIDDEN)
    agent_b_model.to(device) # 将模型移动到指定设备

    # 3. 初始化优化器
    optimizer = AdamW(agent_b_model.parameters(), lr=Config.LEARNING_RATE)

    # 4. 初始化训练器
    trainer = GameTrainer(Config, agent_b_model, optimizer, device)

    # 5. 开始训练
    trainer.train()
