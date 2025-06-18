import json
import os

# --- 配置 ---
DATA_FILE = "/hpc2/puhome/23063003r/refgame_project/data/generated_game_data.json"
NUM_EXPECTED_CANDIDATES = 3 # 你在生成数据时设置的 NUM_CANDIDATES

def check_game_data(data_filepath, expected_num_candidates):
    """
    检查游戏数据文件中每个回合的候选句子数量和正确索引的有效性。

    Args:
        data_filepath (str): 游戏数据JSON文件的路径。
        expected_num_candidates (int): 每个回合期望的候选句子数量。

    Returns:
        tuple: (total_rounds, issues_found_count, problematic_rounds_details)
    """
    print(f"--- 正在检查数据文件: {data_filepath} ---")
    print(f"--- 期望每轮候选句子数量: {expected_num_candidates} ---")

    try:
        with open(data_filepath, 'r', encoding='utf-8') as f:
            game_data_list = json.load(f)
    except FileNotFoundError:
        print(f"❌ 错误: 数据文件 '{data_filepath}' 未找到。请检查路径。")
        return 0, 0, []
    except json.JSONDecodeError:
        print(f"❌ 错误: 无法解析JSON文件 '{data_filepath}'。请检查文件内容是否为有效的JSON。")
        return 0, 0, []
    except Exception as e:
        print(f"❌ 错误加载数据: {e}")
        return 0, 0, []

    total_rounds = len(game_data_list)
    issues_found_count = 0
    problematic_rounds_details = []

    for i, game_round in enumerate(game_data_list):
        round_idx = i + 1 # 1-based indexing for reporting

        candidates = game_round.get('candidate_english_sentences_raw')
        correct_idx = game_round.get('correct_candidate_index')

        # 检查 candidates 字段是否存在且为列表
        if not isinstance(candidates, list):
            issues_found_count += 1
            problem_detail = {
                "round_idx": round_idx,
                "type": "Missing or Invalid Candidates List",
                "detail": f"'candidate_english_sentences_raw' 字段缺失或不是列表: {candidates}"
            }
            problematic_rounds_details.append(problem_detail)
            continue # 跳过当前回合的进一步检查

        # 检查候选句子数量
        if len(candidates) != expected_num_candidates:
            issues_found_count += 1
            problem_detail = {
                "round_idx": round_idx,
                "type": "Incorrect Candidate Count",
                "expected": expected_num_candidates,
                "found": len(candidates),
                "chinese_sentence": game_round.get('target_sentence_chinese_raw', 'N/A'),
                "candidates": candidates
            }
            problematic_rounds_details.append(problem_detail)

        # 检查 correct_candidate_index 的有效性
        if not isinstance(correct_idx, int):
            issues_found_count += 1
            problem_detail = {
                "round_idx": round_idx,
                "type": "Invalid Correct Index Type",
                "detail": f"'correct_candidate_index' 字段缺失或不是整数: {correct_idx}",
                "chinese_sentence": game_round.get('target_sentence_chinese_raw', 'N/A')
            }
            problematic_rounds_details.append(problem_detail)
        elif not (0 <= correct_idx < len(candidates)):
            issues_found_count += 1
            problem_detail = {
                "round_idx": round_idx,
                "type": "Index Out of Bounds",
                "correct_index": correct_idx,
                "candidate_list_length": len(candidates),
                "chinese_sentence": game_round.get('target_sentence_chinese_raw', 'N/A'),
                "candidates": candidates
            }
            problematic_rounds_details.append(problem_detail)
        else:
            # 额外检查：correct_english_sentence_raw 是否真的在 candidates 列表中正确的位置
            correct_en_sentence = game_round.get('correct_english_sentence_raw', 'N/A')
            if candidates[correct_idx] != correct_en_sentence:
                issues_found_count += 1
                problem_detail = {
                    "round_idx": round_idx,
                    "type": "Mismatch Correct Sentence at Index",
                    "correct_index": correct_idx,
                    "expected_sentence": correct_en_sentence,
                    "found_sentence_at_index": candidates[correct_idx],
                    "chinese_sentence": game_round.get('target_sentence_chinese_raw', 'N/A'),
                    "candidates": candidates
                }
                problematic_rounds_details.append(problem_detail)


    print(f"\n--- 检查结果 ---")
    print(f"总游戏回合数: {total_rounds}")
    print(f"发现问题回合数: {len(problematic_rounds_details)} (共 {issues_found_count} 个问题)")

    if problematic_rounds_details:
        print("\n详细问题列表:")
        for detail in problematic_rounds_details:
            print(f"- 回合 {detail['round_idx']} ({detail['type']}):")
            if 'chinese_sentence' in detail:
                print(f"  中文句: {detail['chinese_sentence']}")
            if 'detail' in detail:
                print(f"  详情: {detail['detail']}")
            if 'expected' in detail:
                print(f"  期望数量: {detail['expected']}, 实际数量: {detail['found']}")
            if 'correct_index' in detail:
                print(f"  正确索引: {detail['correct_index']}, 列表长度: {detail['candidate_list_length']}")
            if 'expected_sentence' in detail:
                print(f"  期望句子: '{detail['expected_sentence']}'")
                print(f"  索引处句子: '{detail['found_sentence_at_index']}'")
            if 'candidates' in detail:
                print(f"  候选列表: {detail['candidates']}")
            print("-" * 30)
    else:
        print("🎉 太棒了！所有游戏回合数据都符合预期的格式。")

    return total_rounds, issues_found_count, problematic_rounds_details

if __name__ == "__main__":
    total_rounds, issues_count, details = check_game_data(DATA_FILE, NUM_EXPECTED_CANDIDATES)
