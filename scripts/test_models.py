from transformers import BertTokenizer, AutoTokenizer, AutoModelForCausalLM
import torch
import os

def test_model(model_path, text, device="cuda"):
    print(f"\n🔍 Loading model from: {model_path}")

    if "CPM" in model_path:
        # 手动指定 CPM 使用 SentencePiece tokenizer
        from transformers import PreTrainedTokenizerFast
        tokenizer = PreTrainedTokenizerFast(
            tokenizer_file=os.path.join(model_path, "tokenizer.json"),
            model_max_length=1024
        )
        tokenizer.add_special_tokens({"pad_token": "<pad>"})  # CPM模型需要pad token
        model = AutoModelForCausalLM.from_pretrained(model_path, output_hidden_states=True).to(device)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(model_path, output_hidden_states=True).to(device)

    model.eval()

    inputs = tokenizer(text, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    hidden_states = outputs.hidden_states
    embedding_layer = hidden_states[0]

    print(f"✅ Model tested successfully: {model_path}")
    print(f"🔢 Input text: {text}")
    print(f"📐 Input token shape: {inputs['input_ids'].shape}")
    print(f"📊 Embedding shape: {embedding_layer.shape}")
    print(f"📈 Number of layers (including embedding): {len(hidden_states)}")

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    test_model("models/CPM-Generate", "一个红色的东西从高处落下。", device)
    test_model("models/gpt2", "An apple fell on the ground.", device)
