import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# --- 1. 配置模型路径 ---
# ❗️ 请确保这个路径和您主脚本中使用的路径完全一致
CPM_MODEL_PATH = "/ubsnhome/23063003r/refgame_project/models/CPM-Generate"

# --- 2. 加载模型和分词器 ---
print(f"--- 正在从 '{CPM_MODEL_PATH}' 加载 CPM 模型 ---")
try:
    # 使用 AutoModelForCausalLM 加载完整的、可用于生成的模型
    tokenizer = AutoTokenizer.from_pretrained(CPM_MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(CPM_MODEL_PATH)

    # 将模型移动到可用设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval() # 设置为评估模式
    print("✅ 模型和分词器加载成功！")
    print(f"   模型将被运行在: {device}")
except Exception as e:
    print(f"❌ 模型加载失败，请检查路径是否正确。错误: {e}")
    exit()

# --- 3. 准备测试用的 Prompts ---
prompts_to_test = [
    "清华大学位于中国",
    "人工智能的未来是",
    "飞机是",
    "猫是"
]

# --- 4. 开始测试 ---
print("\n--- 开始进行基础文本续写测试 ---")

for prompt in prompts_to_test:
    print(f"\n{'='*20}")
    print(f"测试输入 (Prompt): '{prompt}'")
    print(f"{'-'*20}")

    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

    # --- 使用两种不同的生成策略进行测试 ---

    # 策略 1: 贪心搜索 (Greedy Search) - 最稳定，没有随机性
    print("  ▶️  测试策略 1: 贪心搜索 (do_sample=False)")
    with torch.no_grad():
        greedy_output = model.generate(
            input_ids,
            max_new_tokens=50,
            pad_token_id=tokenizer.eos_token_id
        )
    decoded_greedy = tokenizer.decode(greedy_output[0], skip_special_tokens=True)
    print(f"  > 生成结果: {decoded_greedy}")

    # 策略 2: 采样搜索 (Sampling) - 带有随机性，与主脚本设置类似
    print("\n  ▶️  测试策略 2: 采样搜索 (do_sample=True)")
    with torch.no_grad():
        sampled_output = model.generate(
            input_ids,
            max_new_tokens=50,
            do_sample=True,
            top_k=50,
            temperature=0.7,
            pad_token_id=tokenizer.eos_token_id
        )
    decoded_sampled = tokenizer.decode(sampled_output[0], skip_special_tokens=True)
    print(f"  > 生成结果: {decoded_sampled}")

print(f"\n{'='*20}")
print("--- 测试结束 ---")
