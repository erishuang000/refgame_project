#!/bin/bash
#SBATCH --job-name=commu_gpt          # 任务名
#SBATCH --partition=h07q1               # 分区名（如果有别的分区请替换）
#SBATCH --nodes=1                         # 节点数
#SBATCH --ntasks=1                        # 任务数
#SBATCH --cpus-per-task=8                 # 每个任务使用4个CPU核心
#SBATCH --mem=16G                          # 分配内存
#SBATCH --time=08:00:00                   # 最长运行时间
#SBATCH --output=logs/commu_%j.out # 标准输出
#SBATCH --error=logs/commu_%j.err  # 错误日志

echo "🚀 作业开始于: $(date)"
echo "📌 节点: $(hostname)"
echo "🔧 使用 CPU 核数: $SLURM_CPUS_PER_TASK"

# 加载环境
source ~/.bashrc
conda activate refgame

# 运行数据处理脚本，推荐在脚本中使用多线程（如 joblib, multiprocessing, torch 等）
python scripts/commu_gpt_newLoss.py

echo "✅ 作业结束于: $(date)"
