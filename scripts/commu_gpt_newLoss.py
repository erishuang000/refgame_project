import torch
from transformers import AutoTokenizer, GPT2Model
import json
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
import os
import random # 引入random用于shuffle数据集

# --- 1. 配置管理 ---
class Config:
    MODEL_PATH = "/puhome/23063003r/refgame_project/models/gpt2"
    DATA_FILE = "/hpc2/puhome/23063003r/refgame_project/data/generated_game_data.json"
    OUTPUT_DIR = "/puhome/23063003r/refgame_project/output/"
    D_HIDDEN = 768  # GPT-2的隐藏层维度

    # Loss 奖励/惩罚系数
    REWARD_CORRECT = 0.1
    PENALTY_WRONG = 1.0

    LEARNING_RATE = 1e-5
    BATCH_SIZE = 1

# --- Listener Loss 函数 ---
def listener_mse_reciprocal_loss(
    semantic_vector_from_agent_A: torch.Tensor,
    semantic_vectors_candidates_B: torch.Tensor,
    correct_candidate_index: torch.Tensor, # 确保这里是 LongTensor
    epsilon: float = 1e-8
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
    print(f"DEBUG: listener_mse_reciprocal_loss input shapes & dtypes:")
    print(f"  semantic_vector_from_agent_A: {semantic_vector_from_agent_A.shape}, {semantic_vector_from_agent_A.dtype}")
    print(f"  semantic_vectors_candidates_B: {semantic_vectors_candidates_B.shape}, {semantic_vectors_candidates_B.dtype}")
    print(f"  correct_candidate_index: {correct_candidate_index.shape}, {correct_candidate_index.dtype}") # 确保是 torch.long

    expanded_vector_A = semantic_vector_from_agent_A.unsqueeze(1)
    print(f"DEBUG: expanded_vector_A shape: {expanded_vector_A.shape}")

    squared_diff = (expanded_vector_A - semantic_vectors_candidates_B).pow(2)
    print(f"DEBUG: squared_diff shape: {squared_diff.shape}")

    mse_distances = squared_diff.sum(dim=-1)
    print(f"DEBUG: mse_distances shape: {mse_distances.shape}")

    logits = 1 / (mse_distances + epsilon)
    print(f"DEBUG: logits shape: {logits.shape}")

    loss = F.cross_entropy(logits, correct_candidate_index) # correct_candidate_index 必须是 torch.long

    return loss

# --- Agent B (Listener) 模型类 ---
class AgentBListener(torch.nn.Module):
    def __init__(self, model_path, all_chinese_chars, hidden_dim):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = GPT2Model.from_pretrained(model_path)

        self.tokenizer.add_special_tokens({'additional_special_tokens': list(all_chinese_chars)})
        self.model.resize_token_embeddings(len(self.tokenizer))

        assert hidden_dim == self.model.config.hidden_size, \
            "D_HIDDEN must match GPT-2's hidden_size for direct use as semantic vector."

        print(f"✅ AgentB: GPT-2 tokenizer 已扩展，新的词汇表大小: {len(self.tokenizer)}")
        print(f"✅ AgentB: GPT-2 模型 Embedding 层已调整。")

    def forward(self, inputs_cn_raw, inputs_en_candidates_list_raw, device):
        # inputs_cn_raw: 形状 (batch_size,) 的字符串元组/列表 (e.g., ['你看起来像一个聪明人。'])
        # inputs_en_candidates_list_raw: 形状 (batch_size, num_candidates) 的元组的元组 (e.g., (('Tom', 'Sami', 'You'),))

        # 记录模型权重在计算前的状态
        embedding_before = self.model.wte.weight.clone().detach()

        # 处理中文乱码输入 (批处理)
        inputs_cn_symbolic = self.tokenizer(
            list(inputs_cn_raw), # DataLoader 可能返回元组，转成列表以便tokenizer处理
            return_tensors="pt",
            padding=True,
            truncation=True
        ).to(device)

        outputs_cn_symbolic = self.model(**inputs_cn_symbolic)
        # 取 CLS token 或平均池化，这里取第一个token的隐藏状态作为句子表示
        semantic_vector_B_from_A = outputs_cn_symbolic.last_hidden_state[:, 0, :]

        print(f"DEBUG_FORWARD: semantic_vector_B_from_A shape: {semantic_vector_B_from_A.shape}")

        # 处理英文候选句子 (批处理)
        # inputs_en_candidates_list_raw 可能是 (('候选1', '候选2', '候选3'),)
        # 需提取出 ['候选1', '候选2', '候选3']

        # 展平候选列表，因为tokenizer期望的是一个包含所有批次所有候选的单层列表
        # 例如，如果 batch_size=2，num_candidates=3，输入可能是 (('A1','A2','A3'), ('B1','B2','B3'))
        # 展平后为 ['A1','A2','A3','B1','B2','B3']
        flat_en_candidates = [sentence for sublist in inputs_en_candidates_list_raw for sentence in sublist]

        inputs_en_candidates_tokenized = self.tokenizer(
            flat_en_candidates,
            return_tensors="pt",
            padding=True,
            truncation=True
        ).to(device)

        print(f"DEBUG_FORWARD: inputs_en_candidates_tokenized input_ids shape: {inputs_en_candidates_tokenized['input_ids'].shape}")

        outputs_en_candidates = self.model(**inputs_en_candidates_tokenized)

        # 提取每个候选句子的语义向量，形状是 (total_candidates_in_batch, D_HIDDEN)
        flat_semantic_vectors_B_candidates = outputs_en_candidates.last_hidden_state[:, 0, :]

        # 将展平的语义向量重新塑形为 (batch_size, num_candidates, D_HIDDEN)
        # num_candidates = len(inputs_en_candidates_list_raw[0]) # 从第一个样本的候选数获取
        num_candidates = len(inputs_en_candidates_list_raw[0])
        print(f"DEBUG_FORWARD: Detected num_candidates per sample: {num_candidates}")

        semantic_vectors_B_candidates = flat_semantic_vectors_B_candidates.view(
            -1, num_candidates, self.model.config.hidden_size # -1 for batch_size
        )
        print(f"DEBUG_FORWARD: semantic_vectors_B_candidates shape (after reshape): {semantic_vectors_B_candidates.shape}")

        return semantic_vector_B_from_A, semantic_vectors_B_candidates, embedding_before

    def get_embedding_after(self):
        return self.model.wte.weight.detach()

# --- 游戏数据加载器 ---
class GameDataset(Dataset):
    def __init__(self, data_file):
        with open(data_file, 'r', encoding='utf-8') as f:
            self.data = json.load(f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # 直接返回原始字典
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

    def train_one_round(self, game_round: dict, round_idx: int, total_rounds: int):
        self.optimizer.zero_grad()

        # 从 DataLoader 得到的 game_round 已经是 PyTorch Tensor 或 List/Tuple 的批次形式
        # 这里不再进行解包，而是直接传递给 AgentBListener.forward
        # 因为 batch_size=1，所以这些批次维度是 1
        cpm_spoken_chinese_sentence_batch = game_round['target_sentence_chinese_raw']
        candidate_english_sentences_batch = game_round['candidate_english_sentences_raw']
        correct_candidate_index_batch = game_round['correct_candidate_index'] # 已经是 Tensor

        # 1. Agent B (GPT-2) 处理中文乱码输入及英文候选句子
        semantic_vector_B_from_A, semantic_vectors_B_candidates, embedding_before = \
            self.model(
                cpm_spoken_chinese_sentence_batch, # 直接传入 DataLoader 返回的批次
                candidate_english_sentences_batch, # 直接传入 DataLoader 返回的批次
                self.device
            )

        # 2. Agent B 猜测 (计算相似度并预测)
        # similarities: (batch_size, num_candidates)
        similarities = F.cosine_similarity(
            semantic_vector_B_from_A.unsqueeze(1), # (batch_size, 1, D_HIDDEN)
            semantic_vectors_B_candidates,         # (batch_size, num_candidates, D_HIDDEN)
            dim=2 # 沿着 D_HIDDEN 维度计算相似度
        ).squeeze(1) # 结果形状 (batch_size, num_candidates)

        # 对于 batch_size=1，predicted_index 依然取第一个
        predicted_index = torch.argmax(similarities[0]).item()

        print(f"🤔 相似度得分 (越高越相似): {similarities[0].tolist()}")
        print(f"🔮 Agent B 猜测的索引: {predicted_index}")

        # 3. 反馈与权重更新 (Agent B 学习)
        correct_index_tensor_batch = correct_candidate_index_batch.to(self.device) # 确保在正确设备上

        # 使用 Listener Loss 函数
        base_loss = listener_mse_reciprocal_loss(
            semantic_vector_B_from_A,
            semantic_vectors_B_candidates,
            correct_index_tensor_batch # 传入 Tensor
        )

        is_correct = (predicted_index == correct_index_tensor_batch.item()) # 对于batch_size=1，比较单个值
        if is_correct:
            loss = base_loss * (1 - self.config.REWARD_CORRECT)
            outcome_message = f"🎉 Agent B 猜对啦！损失调整系数: {(1 - self.config.REWARD_CORRECT):.2f}"
        else:
            loss = base_loss * self.config.PENALTY_WRONG
            outcome_message = f"💔 Agent B 猜错了！损失调整系数: {self.config.PENALTY_WRONG:.2f}"

        print(outcome_message)

        loss.backward()
        self.optimizer.step()

        # 4. 比较 Embedding 变化
        embedding_after = self.model.get_embedding_after()
        diff = torch.norm(embedding_after - embedding_before).item()

        self.total_loss_sum += loss.item()
        if is_correct:
            self.correct_predictions_count += 1

        print(f"📉 本轮游戏最终损失: {loss.item():.4f}")
        print(f"🔍 Embedding (word token embeddings) 改变量: {diff:.6f}")
        print(f"✨ Agent B 最终猜测结果: {is_correct}")

        # 记录本轮数据 (注意解包原始字符串以便保存JSON)
        round_data = {
            "round_idx": round_idx,
            "chinese_sentence": game_round['target_sentence_chinese_raw'][0], # 解包
            "correct_english_sentence": game_round['correct_english_sentence_raw'][0], # 解包
            "candidate_english_sentences": game_round['candidate_english_sentences_raw'][0], # 解包
            "correct_candidate_idx": game_round['correct_candidate_index'][0].item(), # 解包并转为int
            "predicted_index": predicted_index,
            "similarities": similarities[0].tolist(), # 解包并转为list
            "is_correct_prediction": is_correct,
            "base_loss": base_loss.item(),
            "final_loss": loss.item(),
            "embedding_diff_norm": diff
        }
        self.per_round_metrics.append(round_data)

    def train(self):
        game_dataset = GameDataset(self.config.DATA_FILE)
        # 使用 collate_fn=self._custom_collate_fn 确保数据以预期格式进入
        # 注意: 即使 batch_size=1, collate_fn 也会被调用
        data_loader = DataLoader(game_dataset, batch_size=self.config.BATCH_SIZE,
                                 shuffle=False, collate_fn=self._custom_collate_fn)

        self.model.model.train() # Set model to training mode
        total_rounds = len(data_loader)
        print(f"\n--- 准备进行 {total_rounds} 轮游戏 ---")

        for i, game_round_batch in enumerate(data_loader):
            print(f"\n--- 游戏回合 {i + 1}/{total_rounds} ---")

            # 由于我们有自定义的 collate_fn，game_round_batch 已经是批处理后的 Tensor 或 List
            # 但为了打印日志，我们可能需要从批次中取出第一个样本的数据
            # 注意：这里直接打印 batch_data 的 [0] 可能会因类型不同而失败
            # 最好从实际传入 train_one_round 的参数中获取
            print(f"🎯 目标中文句子 (CPM '说'): {game_round_batch['target_sentence_chinese_raw'][0]}")
            print(f"📚 候选英文句子 (Agent B 选择): {game_round_batch['candidate_english_sentences_raw'][0]}")
            print(f"✅ 正确索引: {game_round_batch['correct_candidate_index'][0].item()}")

            self.train_one_round(game_round_batch, i + 1, total_rounds) # 直接传入批次数据

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

        if not os.path.exists(self.config.OUTPUT_DIR):
            os.makedirs(self.config.OUTPUT_DIR)

        output_file_path = os.path.join(self.config.OUTPUT_DIR, "training_results.json")
        with open(output_file_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)

        print(f"\n🎉 训练结果已保存到: {output_file_path}")

    # 自定义 collate_fn
    def _custom_collate_fn(self, batch_list_of_dicts):
        # batch_list_of_dicts 是一个列表，每个元素是 GameDataset.__getitem__ 返回的一个字典
        # 例如：[{'target_cn': '你好', 'candidates_en': ['hi', 'hello']}, ...]

        collated_batch = {}
        # 初始化列表来收集所有字段
        chinese_sentences = []
        correct_english_sentences = []
        candidate_english_sentences_list = [] # 这是一个列表的列表 (batch_size, num_candidates)
        correct_candidate_indices = []

        for item in batch_list_of_dicts:
            chinese_sentences.append(item['target_sentence_chinese_raw'])
            correct_english_sentences.append(item['correct_english_sentence_raw'])
            candidate_english_sentences_list.append(item['candidate_english_sentences_raw'])
            correct_candidate_indices.append(item['correct_candidate_index'])

        collated_batch['target_sentence_chinese_raw'] = chinese_sentences # list of strings (batch_size,)
        collated_batch['correct_english_sentence_raw'] = correct_english_sentences # list of strings (batch_size,)
        collated_batch['candidate_english_sentences_raw'] = candidate_english_sentences_list # list of lists (batch_size, num_candidates)
        collated_batch['correct_candidate_index'] = torch.tensor(correct_candidate_indices, dtype=torch.long) # Tensor (batch_size,)

        return collated_batch


# --- 主程序入口 ---
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 使用设备: {device}")

    all_chinese_chars_in_corpus = set()
    try:
        with open(Config.DATA_FILE, 'r', encoding='utf-8') as f:
            temp_game_data = json.load(f)
        for entry in temp_game_data:
            all_chinese_chars_in_corpus.update(list(entry['target_sentence_chinese_raw']))
    except Exception as e:
        print(f"❌ 错误加载数据以收集中文符号，使用默认集: {e}")
        all_chinese_chars_in_corpus = set("一个苹果掉到了地上。猫跳到了桌子上。一辆红色的汽车开在街上。狗追球。天空是蓝色的。她在看书。睡沙发。孩子们在公园玩。太阳从东方升起。喜欢听音乐。咖啡很烫。我饿了想吃东西。")


    agent_b_model = AgentBListener(Config.MODEL_PATH, all_chinese_chars_in_corpus, Config.D_HIDDEN)
    agent_b_model.to(device)

    optimizer = AdamW(agent_b_model.parameters(), lr=Config.LEARNING_RATE)

    trainer = GameTrainer(Config, agent_b_model, optimizer, device)

    trainer.train()
