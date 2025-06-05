# from transformers import AutoTokenizer, AutoModelForCausalLM
# import torch
# # from transformers import AutoTokenizer, AutoModelWithLMHead
#
# model_id = "TsinghuaAI/CPM-Generate"
# save_path = "./CPM-Generate"
#
# tokenizer = AutoTokenizer.from_pretrained(model_id)
# model = AutoModelWithLMHead.from_pretrained(model_id)
#
# tokenizer.save_pretrained(save_path)
# model.save_pretrained(save_path)


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


# if __name__ == "__main__":
#     device = "cuda" if torch.cuda.is_available() else "cpu"
#     test_model_cpm("/puhome/23063003r/refgame_project/models/CPM-Generate", "一个红色的东西从高处落下。", device)


from transformers import AutoTokenizer, AutoModelWithLMHead, AdamW
import torch

# 路径根据你的本地模型位置修改
MODEL_PATH = "/puhome/23063003r/refgame_project/models/CPM-Generate"

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelWithLMHead.from_pretrained(MODEL_PATH)
model.train()  # 设置为训练模式

# 假设对话样本
sample_text = "用户：你好\n模型：你好，请问有什么可以帮你？"
inputs = tokenizer(sample_text, return_tensors="pt")
inputs = {k: v for k, v in inputs.items() if k in ["input_ids", "attention_mask"]}
inputs["labels"] = inputs["input_ids"].clone()

# 拿到embedding层
embedding_before = model.transformer.wte.weight.clone().detach()

# 一步微调
optimizer = AdamW(model.parameters(), lr=1e-5)
outputs = model(**inputs)
loss = outputs.loss
loss.backward()
optimizer.step()

# 比较embedding改变
embedding_after = model.transformer.wte.weight.detach()
diff = torch.norm(embedding_after - embedding_before).item()

print(f"✅ 微调完成。Loss = {loss.item():.4f}")
print(f"🔍 Embedding 改变量: {diff:.6f}")
