# from transformers import AutoTokenizer, AutoModelForCausalLM
#
# model_id = "TsinghuaAI/CPM-Generate"
# save_path = "./CPM-Generate"
#
# tokenizer = AutoTokenizer.from_pretrained(model_id)
# model = AutoModelForCausalLM.from_pretrained(model_id)
#
# tokenizer.save_pretrained(save_path)
# model.save_pretrained(save_path)


from transformers import AutoTokenizer, AutoModelWithLMHead
from torch.optim import AdamW
import torch

# 路径根据你的本地模型位置修改
MODEL_PATH = "/ubsnhome/23063003r/refgame_project/models/CPM-Generate"

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
