from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.optim import AdamW
import torch
import json

# 设置模型路径
MODEL_PATH = "/ubsnhome/23063003r/refgame_project/models/gpt2"

# 初始化 tokenizer 和模型
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH)
model.train()

optimizer = AdamW(model.parameters(), lr=1e-5)

# 故事文本（按句子分割）
story = """
Could humans live on Mars some day? Scientists ask this question because Earth and Mars are similar. Similar to Earth's day, Mars's day is about 24 hours long. Also, both planets are near the Sun in our solar system. Earth is the 3rd planet and Mars the 4th planet from the Sun. Mars also has an axial tilt similar to Earth's axial tilt. An axial tilt gives both planets seasons with temperature changes. Just like Earth, Mars has cold winters and warmer summers. Like Earth, Mars has winds, weather, dust storms, and volcanoes. But in some ways, Earth and Mars are different. Differences include temperature, length of a year, and gravity. The average temperature is -81 deg F on Mars, but 57 deg F on Earth. A Martian year is almost twice as long as an Earth year. Earth's gravity is almost 3 times stronger than Martian gravity. Given the similarities, can humans go to Mars and live there? NASA scientists want to answer this question. NASA oversees U.S. research on space exploration. NASA scientists send devices called spacecraft to explore Mars. The spacecraft carry rovers that can rove or move around. These wheeled rovers can explore characteristics of the planet. They can also show land characteristics and weather on Mars. One of these NASA rovers is named Curiosity. Curiosity found evidence that soil on Mars contains 2% water. NASA has planned a new mission called Mars 2020. This mission will use a new car-sized rover to examine Mars. The new rover will contain additional instruments to study Mars. For example, one instrument will take images beneath Mars's surface. Another instrument will attempt to make oxygen from carbon dioxide. Mars 2020 will help scientists answer important questions. It will explore whether there has been life on Mars. It will also answer whether humans can live on Mars in the future.
"""

# 分句（你也可以替换为 nltk 或自定义规则）
sentences = [s.strip() for s in story.strip().split(".") if s.strip()]
results = []

for i, sentence in enumerate(sentences):
    sentence = sentence + "."  # 补上句号
    inputs = tokenizer(sentence, return_tensors="pt")
    inputs = {k: v for k, v in inputs.items() if k in ["input_ids", "attention_mask"]}
    inputs["labels"] = inputs["input_ids"].clone()

    with torch.no_grad():
        embedding_before = model.transformer.wte.weight.clone()

    # 单步训练
    outputs = model(**inputs)
    loss = outputs.loss
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    # 保存当前 embedding（flatten）
    embedding_after = model.transformer.wte.weight.detach().cpu()
    results.append({
        "sentence": sentence,
        "embedding": embedding_after.flatten().tolist()  # 注意数据量大时文件也大
    })

    print(f"✅ [{i+1}/{len(sentences)}] Sentence trained. Loss = {loss.item():.4f}")

# 保存为 JSON 文件
with open("embedding_after_each_sentence.json", "w") as f:
    json.dump(results, f)

print("✅ 所有句子学习完成，embedding 已保存到 embedding_after_each_sentence.json")
