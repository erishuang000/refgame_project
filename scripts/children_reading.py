import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# --- 配置 ---
MODEL_PATH = "/ubsnhome/23063003r/refgame_project/models/gpt2" # 确保这是您的本地GPT-2模型路径
STORY = """Could humans live on Mars some day? Scientists ask this question because Earth and Mars are similar. Similar to Earth's day, Mars's day is about 24 hours long. Also, both planets are near the Sun in our solar system. Earth is the 3rd planet and Mars the 4th planet from the Sun. Mars also has an axial tilt similar to Earth's axial tilt. An axial tilt gives both planets seasons with temperature changes. Just like Earth, Mars has cold winters and warmer summers. Like Earth, Mars has winds, weather, dust storms, and volcanoes. But in some ways, Earth and Mars are different. Differences include temperature, length of a year, and gravity. The average temperature is -81 deg F on Mars, but 57 deg F on Earth. A Martian year is almost twice as long as an Earth year. Earth's gravity is almost 3 times stronger than Martian gravity. Given the similarities, can humans go to Mars and live there? NASA scientists want to answer this question. NASA oversees U.S. research on space exploration. NASA scientists send devices called spacecraft to explore Mars. The spacecraft carry rovers that can rove or move around. These wheeled rovers can explore characteristics of the planet. They can also show land characteristics and weather on Mars. One of these NASA rovers is named Curiosity. Curiosity found evidence that soil on Mars contains 2% water. NASA has planned a new mission called Mars 2020. This mission will use a new car-sized rover to examine Mars. The new rover will contain additional instruments to study Mars. For example, one instrument will take images beneath Mars's surface. Another instrument will attempt to make oxygen from carbon dioxide. Mars 2020 will help scientists answer important questions. It will explore whether there has been life on Mars. It will also answer whether humans can live on Mars in the future."""

# --- 加载模型和分词器 ---
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH)
model.eval() # 设置为评估模式，因为我们是提取Embedding，不是训练

# GPT-2 的分词器没有 pad token，手动添加一个以确保批处理或需要 padding 的场景可以正常工作
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    model.resize_token_embeddings(len(tokenizer))


# --- 故事分割成句子 ---
# 一个简单的句子分割方法，您可以根据需要调整
sentences = [s.strip() for s in STORY.split('.') if s.strip()]
# 确保每个句子以标点符号结尾，方便处理
sentences = [s + '.' if not s.endswith(('.', '?', '!')) else s for s in sentences]


# --- 辅助函数：获取Embedding ---
def get_embeddings(model, tokenizer, text, return_input_embedding=False):
    """
    获取给定文本的输入和输出Embedding。
    输出Embedding这里我们取最后一个隐藏层的CLS token（如果有）或所有token的平均。
    对于GPT-2，通常直接使用最后一个隐藏层的所有token的平均。
    """
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=tokenizer.model_max_length)

    # 获取输入Embedding（词嵌入层在模型的前向传播之前）
    # wte 是 'word token embeddings'
    if return_input_embedding:
        # GPT-2 的输入 embedding 是通过查找 wte.weight 得到的
        input_ids = inputs['input_ids']
        input_embeddings = model.transformer.wte(input_ids).squeeze(0).mean(dim=0).tolist()
    else:
        input_embeddings = None # 默认不返回，节省计算

    # 获取输出Embedding
    with torch.no_grad(): # 在评估模式下，禁用梯度计算
        outputs = model(**inputs, output_hidden_states=True)

    # 取最后一个隐藏层的输出作为句子Embedding
    # 这里我们取所有token的平均，可以根据需求修改为 [CLS] token (如果模型有) 或其他池化策略
    # outputs.last_hidden_state 的形状是 (batch_size, sequence_length, hidden_size)
    output_embeddings = outputs.hidden_states[-1].squeeze(0).mean(dim=0).tolist()

    return input_embeddings, output_embeddings

# --- 方法一：逐句处理 (独立上下文) ---
print("--- 正在执行方法一：逐句处理 ---")
results_method1 = []
for i, sentence in enumerate(sentences):
    print(f"处理句子 {i+1}/{len(sentences)}: '{sentence}'")

    # 每次获取前都克隆一份初始的wte.weight作为“输入embedding层”
    # 这个是模型词汇表层面的embedding，而非特定句子的输入embedding
    # 如果要获取特定句子的 token embedding，需要在模型内部提取
    # 这里我们简化为每次调用前获取模型当前的 wte 权重作为“输入层状态”
    # 但更直接的，是获取当前句子的token在模型wte中的lookup

    # 获取句子本身的输入 embedding (通过查找wte表) 和输出 embedding
    sentence_input_embedding, sentence_output_embedding = get_embeddings(model, tokenizer, sentence, return_input_embedding=True)

    results_method1.append({
        "sentence_index": i,
        "text": sentence,
        "input_embedding": sentence_input_embedding, # 句子的平均输入embedding
        "output_embedding": sentence_output_embedding, # 句子的平均输出embedding
    })

# 保存结果
output_file_method1 = "output/story_embeddings_method1.json"
with open(output_file_method1, 'w', encoding='utf-8') as f:
    json.dump(results_method1, f, ensure_ascii=False, indent=4)
print(f"方法一结果已保存到：{output_file_method1}")

# --- 方法二：累积上下文处理 ---
print("\n--- 正在执行方法二：累积上下文处理 ---")
results_method2 = []
current_context = ""

for i, sentence in enumerate(sentences):
    print(f"处理句子 {i+1}/{len(sentences)} (累积上下文): '{sentence}'")

    # 构造当前完整的输入文本（包括所有历史上下文和当前句子）
    full_text_input = current_context + sentence

    # 获取当前完整文本的输入 embedding 和 输出 embedding
    # 我们关注的是模型对 *当前句子* 的理解，但这个理解是建立在完整上下文之上的。
    # 所以我们提取的是 `full_text_input` 的输出 embedding。
    # 这里为了简化，我们仍取 full_text_input 的整体平均输出 embedding。
    # 如果要精确到新加入句子的 embedding 变化，则需要更复杂的逻辑，
    # 比如找到新句子token在 full_text_input token序列中的位置，并提取其对应的 hidden_state。

    # 为了简化且仍能捕捉到上下文影响，我们获取整个 full_text_input 的平均 embedding。
    # input_embedding 对完整文本来说意义不大，因为我们主要看的是新加入句子在上下文中的输出变化。
    # 因此，我们只关注 full_text_input 的输出 embedding。
    _, full_text_output_embedding = get_embeddings(model, tokenizer, full_text_input, return_input_embedding=False)

    results_method2.append({
        "sentence_index": i,
        "text": sentence,
        "context_at_this_point": current_context,
        "full_text_input": full_text_input,
        "output_embedding_with_context": full_text_output_embedding, # 包含上下文信息的输出embedding
    })

    # 更新上下文
    current_context += sentence + " " # 添加一个空格以分隔下一个句子

# 保存结果
output_file_method2 = "output/story_embeddings_method2.json"
with open(output_file_method2, 'w', encoding='utf-8') as f:
    json.dump(results_method2, f, ensure_ascii=False, indent=4)
print(f"方法二结果已保存到：{output_file_method2}")
