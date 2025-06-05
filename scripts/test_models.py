from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
# from transformers import AutoTokenizer, AutoModelWithLMHead

model_id = "TsinghuaAI/CPM-Generate"
save_path = "./CPM-Generate"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelWithLMHead.from_pretrained(model_id)

tokenizer.save_pretrained(save_path)
model.save_pretrained(save_path)


# def test_model_cpm(model_path, input_text, device):
#     print(f"🔍 Loading model from: {model_path}")
#
#     tokenizer = AutoTokenizer.from_pretrained(
#         model_path,
#         trust_remote_code=True,
#         local_files_only=True
#     )
#     model = AutoModelForCausalLM.from_pretrained(
#         model_path,
#         trust_remote_code=True,
#         local_files_only=True
#     ).to(device)
#
#     inputs = tokenizer(input_text, return_tensors="pt").to(device)
#     outputs = model.generate(**inputs, max_length=50)
#     decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
#
#     print("🧠 Output:", decoded)


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    test_model_cpm("/puhome/23063003r/refgame_project/models/CPM-Generate", "一个红色的东西从高处落下。", device)
