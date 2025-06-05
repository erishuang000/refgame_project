from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

def test_model_cpm(model_path, input_text, device):
    print(f"🔍 Loading model from: {model_path}")

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True).to(device)

    inputs = tokenizer(input_text, return_tensors="pt").to(device)
    outputs = model.generate(**inputs, max_length=50)
    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)

    print("🧠 Output:", decoded)

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    test_model_cpm("models/CPM-Generate", "一个红色的东西从高处落下。", device)
