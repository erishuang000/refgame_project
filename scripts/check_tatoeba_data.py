import bz2
import tarfile
import os

# 定义文件路径
data_dir = "/puhome/23063003r/refgame_project/data/" # 确保这个路径是正确的，指向你的data文件夹
cmn_file = os.path.join(data_dir, "cmn_sentences_detailed.tsv.bz2")
eng_file = os.path.join(data_dir, "eng_sentences.tsv.bz2")
links_file = os.path.join(data_dir, "links.tar.bz2")

def read_bz2_file(filepath, num_lines=5):
    """读取.bz2文件的前N行"""
    print(f"\n--- {filepath} (前 {num_lines} 行) ---")
    try:
        with bz2.open(filepath, 'rt', encoding='utf8') as f:
            for i, line in enumerate(f):
                if i >= num_lines:
                    break
                print(f"  {line.strip()}")
    except FileNotFoundError:
        print(f"文件未找到: {filepath}")
    except Exception as e:
        print(f"读取文件时出错 {filepath}: {e}")

def read_tar_bz2_file(filepath, num_lines=5):
    """读取.tar.bz2文件内部的第一个文件的前N行"""
    print(f"\n--- {filepath} (内部文件前 {num_lines} 行) ---")
    try:
        with tarfile.open(filepath, 'r:bz2') as tar:
            # 假设只有一个内部文件或者我们只关心第一个
            members = tar.getmembers()
            if not members:
                print("  Tar文件中没有成员。")
                return

            first_member = members[0]
            print(f"  内部文件: {first_member.name}")
            extracted_file = tar.extractfile(first_member)

            if extracted_file:
                for i, line in enumerate(extracted_file):
                    if i >= num_lines:
                        break
                    print(f"  {line.decode('utf8').strip()}") # tarfile读取的是bytes
            else:
                print(f"  无法提取内部文件: {first_member.name}")

    except FileNotFoundError:
        print(f"文件未找到: {filepath}")
    except tarfile.ReadError:
        print(f"无法解压.tar.bz2文件，可能已损坏或格式不正确: {filepath}")
    except Exception as e:
        print(f"读取文件时出错 {filepath}: {e}")

# 执行读取
read_bz2_file(cmn_file)
read_bz2_file(eng_file)
read_tar_bz2_file(links_file)
