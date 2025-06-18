import bz2
import tarfile
import os
import json
import random

# --- 配置 ---
DATA_DIR = "/puhome/23063003r/refgame_project/data/"
CMN_FILE = os.path.join(DATA_DIR, "cmn_sentences_detailed.tsv.bz2")
ENG_FILE = os.path.join(DATA_DIR, "eng_sentences.tsv.bz2")
LINKS_FILE = os.path.join(DATA_DIR, "links.tar.bz2")
OUTPUT_JSON_FILE = os.path.join(DATA_DIR, "generated_game_data.json")

NUM_CANDIDATES = 3  # 每个回合的英文候选句子数量 (1个正确 + N-1个干扰)
NUM_SAMPLES_TO_GENERATE = 15000 # 想要生成的游戏回合数。如果None，则生成所有可能的简单句对。
MIN_SENTENCE_LEN = 3 # 句子的最小词数（英文）或字符数（中文）
MAX_SENTENCE_LEN = 15 # 句子的最大词数（英文）或字符数（中文）


# --- 辅助函数：读取文件 ---

def read_bz2_tsv_to_dict(filepath):
    """读取.bz2的tsv文件，并构建 ID -> 文本 的字典"""
    id_to_text = {}
    try:
        with bz2.open(filepath, 'rt', encoding='utf8') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 3: # 确保至少有ID、语言、文本三列
                    sentence_id = parts[0]
                    text = parts[2]
                    id_to_text[sentence_id] = text
    except FileNotFoundError:
        print(f"错误: 文件未找到 - {filepath}")
        return None
    except Exception as e:
        print(f"读取文件时出错 {filepath}: {e}")
        return None
    return id_to_text

def read_tar_bz2_links(filepath):
    """读取.tar.bz2中的links.tsv文件，并返回链接对列表"""
    links = []
    try:
        with tarfile.open(filepath, 'r:bz2') as tar:
            members = tar.getmembers()
            if not members:
                print(f"警告: {filepath} 中没有成员文件。")
                return []

            # 查找 links.tsv 或类似的第一个 .tsv/.csv 文件
            links_member = None
            for member in members:
                if member.name.endswith('.tsv') or member.name.endswith('.csv'):
                    links_member = member
                    break

            if not links_member:
                print(f"警告: 在 {filepath} 中未找到 .tsv 或 .csv 成员文件。")
                return []

            extracted_file = tar.extractfile(links_member)
            if extracted_file:
                for line in extracted_file:
                    parts = line.decode('utf8').strip().split('\t')
                    if len(parts) == 2: # 确保是 ID \t ID 格式
                        links.append((parts[0], parts[1]))
            else:
                print(f"错误: 无法提取 {links_member.name} 从 {filepath}。")

    except FileNotFoundError:
        print(f"错误: 文件未找到 - {filepath}")
        return []
    except tarfile.ReadError:
        print(f"错误: 无法解压.tar.bz2文件，可能已损坏或格式不正确 - {filepath}")
        return []
    except Exception as e:
        print(f"读取文件时出错 {filepath}: {e}")
        return []
    return links

def is_simple_sentence(text, lang_code, min_len, max_len):
    """判断句子是否符合简单句子标准（基于长度）"""
    if not text:
        return False
    if lang_code == 'cmn': # 中文按字符数
        length = len(text.replace(" ", "")) # 移除空格计算字符数
    else: # 英文按词数
        length = len(text.split())
    return min_len <= length <= max_len

# --- 主要构建函数 ---

def build_tatoeba_game_data(
    cmn_filepath, eng_filepath, links_filepath,
    num_candidates=3, num_samples=None,
    min_len=3, max_len=15
):
    print("1. 读取中文句子数据...")
    cmn_id_to_text = read_bz2_tsv_to_dict(cmn_filepath)
    if cmn_id_to_text is None: return []
    print(f"   - 读取到 {len(cmn_id_to_text)} 条中文句子。")

    print("2. 读取英文句子数据...")
    eng_id_to_text = read_bz2_tsv_to_dict(eng_filepath)
    if eng_id_to_text is None: return []
    print(f"   - 读取到 {len(eng_id_to_text)} 条英文句子。")

    print("3. 读取链接数据...")
    links = read_tar_bz2_links(links_filepath)
    if not links: return []
    print(f"   - 读取到 {len(links)} 条链接。")

    print("4. 构建平行句对并筛选简单句子...")
    all_parallel_pairs = [] # 存储 (中文文本, 英文文本) 对

    # 建立一个英文句子列表，用于随机抽取干扰项
    all_english_sentences_list = list(eng_id_to_text.values())

    for id1, id2 in links:
        text1 = cmn_id_to_text.get(id1)
        text2 = eng_id_to_text.get(id2)

        # 检查是否是 中文 -> 英文 的链接
        if text1 and text2 and is_simple_sentence(text1, 'cmn', min_len, max_len) and is_simple_sentence(text2, 'eng', min_len, max_len):
            all_parallel_pairs.append((text1, text2))

        # 检查是否是 英文 -> 中文 的链接（Tatoeba链接是双向的，但保险起见都检查）
        text1_eng = eng_id_to_text.get(id1)
        text2_cmn = cmn_id_to_text.get(id2)
        if text1_eng and text2_cmn and is_simple_sentence(text2_cmn, 'cmn', min_len, max_len) and is_simple_sentence(text1_eng, 'eng', min_len, max_len):
             # 避免重复添加，因为links文件可能已经包含双向链接
            if (text2_cmn, text1_eng) not in all_parallel_pairs:
                all_parallel_pairs.append((text2_cmn, text1_eng))

    print(f"   - 找到 {len(all_parallel_pairs)} 条符合简单句子标准的平行句对。")

    print("5. 生成游戏回合数据...")
    game_data = []

    # 确定要处理的样本数量
    actual_num_samples = num_samples if num_samples is not None and num_samples <= len(all_parallel_pairs) else len(all_parallel_pairs)

    # 随机选择要处理的平行句对，以避免每次都处理相同的前N个，特别是当数据量很大时
    selected_parallel_pairs = random.sample(all_parallel_pairs, actual_num_samples)

    for i, (target_cn, correct_en) in enumerate(selected_parallel_pairs):
        candidates = [correct_en] # 添加正确答案

        # 收集干扰项
        distractors = []
        # 从所有英文句子中随机选择，直到达到所需数量，且不与正确答案重复
        while len(distractors) < (num_candidates - 1):
            random_en = random.choice(all_english_sentences_list)
            if random_en != correct_en and random_en not in distractors:
                distractors.append(random_en)

        candidates.extend(distractors)

        # 打乱候选句子，并找到正确答案的新索引
        random.shuffle(candidates)
        correct_index = candidates.index(correct_en)

        game_round = {
            "target_sentence_chinese_raw": target_cn,
            "correct_english_sentence_raw": correct_en,
            "candidate_english_sentences_raw": candidates,
            "correct_candidate_index": correct_index
        }
        game_data.append(game_round)

        if (i + 1) % 100 == 0:
            print(f"   - 已生成 {i + 1}/{actual_num_samples} 轮游戏数据。")

    print(f"   - 最终生成 {len(game_data)} 轮游戏数据。")
    return game_data

# --- 执行构建并保存 ---
if __name__ == "__main__":
    print("--- 开始构建游戏数据库 ---")

    # 构建游戏数据
    generated_game_data = build_tatoeba_game_data(
        CMN_FILE, ENG_FILE, LINKS_FILE,
        num_candidates=NUM_CANDIDATES,
        num_samples=NUM_SAMPLES_TO_GENERATE, # 根据需要调整生成的样本数
        min_len=MIN_SENTENCE_LEN,
        max_len=MAX_SENTENCE_LEN
    )

    if generated_game_data:
        # 保存到 JSON 文件
        with open(OUTPUT_JSON_FILE, 'w', encoding='utf-8') as f:
            json.dump(generated_game_data, f, ensure_ascii=False, indent=2)
        print(f"\n🎉 成功将游戏数据保存到: {OUTPUT_JSON_FILE}")
    else:
        print("\n❌ 未能生成游戏数据。请检查文件路径和内容。")
