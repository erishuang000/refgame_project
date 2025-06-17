#!/bin/bash
#SBATCH --job-name=make_dataset           # 任务名
#SBATCH --partition=h07q1               # 分区名（如果有别的分区请替换）
#SBATCH --nodes=1                         # 节点数
#SBATCH --ntasks=1                        # 任务数
#SBATCH --cpus-per-task=4                 # 每个任务使用4个CPU核心
#SBATCH --mem=8G                          # 分配内存
#SBATCH --time=01:00:00                   # 最长运行时间
#SBATCH --output=logs/make_dataset_%j.out # 标准输出
#SBATCH --error=logs/make_dataset_%j.err  # 错误日志

echo "🚀 作业开始于: $(date)"
echo "📌 节点: $(hostname)"
echo "🔧 使用 CPU 核数: $SLURM_CPUS_PER_TASK"

# 加载环境
source ~/.bashrc
conda activate refgame

# 运行数据处理脚本，推荐在脚本中使用多线程（如 joblib, multiprocessing, torch 等）
python scripts/make_eng_cmn_dataset.py

echo "✅ 作业结束于: $(date)"
