# test_model.py

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

def test_model(model_path, text, device="cuda"):
    print(f"\n🔍 Loading model from: {model_path}")

    # 加载 tokenizer 和 model
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path, output_hidden_states=True).to(device)
    model.eval()

    # 编码文本
    inputs = tokenizer(text, return_tensors="pt").to(device)

    # 前向传播
    with torch.no_grad():
        outputs = model(**inputs)

    # 获取 embedding 层输出（hidden_states[0]）
    hidden_states = outputs.hidden_states  # Tuple of length (n_layers + 1)
    embedding_layer = hidden_states[0]

    print(f"✅ Model tested successfully: {model_path}")
    print(f"🔢 Input text: {text}")
    print(f"📐 Input token shape: {inputs['input_ids'].shape}")
    print(f"📊 Embedding shape: {embedding_layer.shape}")  # [batch_size, seq_len, hidden_size]
    print(f"📈 Number of layers (including embedding): {len(hidden_states)}")

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 🀄 中文模型（CPM）
    test_model("models/CPM-Generate", "一个红色的东西从高处落下。", device)

    # 🇬🇧 英文模型（GPT2）
    test_model("models/gpt2", "An apple fell on the ground.", device)
