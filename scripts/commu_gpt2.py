from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.optim import AdamW
import torch
import os

def interactive_finetune(model_path, save_path, rounds=10, lr=1e-5):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path)
    model.train()
    optimizer = AdamW(model.parameters(), lr=lr)

    history = ""

    for round_num in range(rounds):
        user_input = input(f"🧑 User ({round_num + 1}/{rounds}): ")
        history += f"用户：{user_input}\n"

        train_text = history + "模型："
        inputs = tokenizer(train_text, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v for k, v in inputs.items() if k in ["input_ids", "attention_mask"]}
        inputs["labels"] = inputs["input_ids"].clone()

        outputs = model(**inputs)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        with torch.no_grad():
            gen_ids = model.generate(inputs["input_ids"], max_length=inputs["input_ids"].shape[1] + 50, pad_token_id=tokenizer.eos_token_id)
            reply = tokenizer.decode(gen_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

        print(f"🤖 GPT: {reply.strip()}")
        history += f"模型：{reply.strip()}\n"

    os.makedirs(save_path, exist_ok=True)
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    print(f"✅ 模型已保存至: {save_path}")

if __name__ == "__main__":
    MODEL_PATH = "/puhome/23063003r/refgame_project/models/gpt2"
    SAVE_PATH = "/puhome/23063003r/refgame_project/models/gpt2_finetuned"
    interactive_finetune(MODEL_PATH, SAVE_PATH)
