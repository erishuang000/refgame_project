#!/bin/bash
#SBATCH --job-name=commu_gptXL         # 任务名
#SBATCH --partition=normal             # 分区名（如果有别的分区请替换）
#SBATCH --nodes=1                      # 节点数
#SBATCH --ntasks=1                     # 任务数
#SBATCH --cpus-per-task=8              # 每个任务使用8个CPU核心
#SBATCH --mem=16G                      # 分配内存
#SBATCH --time=08:00:00                # 最长运行时间
#SBATCH --output=logs/commu_%j.out # 标准输出
#SBATCH --error=logs/commu_%j.err  # 错误日志

echo "🚀 作业开始于: $(date)"
echo "📌 节点: $(hostname)"
echo "🔧 使用 CPU 核数: $SLURM_CPUS_PER_TASK"

# 加载 miniconda3 模块 (如果你的HPC系统需要此步骤)
module load miniconda3/24.3.0-gcc-11.5.0-linux-rocky9-cascadelake-nwowwvl

# 这是在 SLURM 脚本中正确激活 Conda 环境的关键步骤
# 确保 /ubsn/home/23063003r/miniconda3 是你的 Miniconda3 实际安装路径
source activate refgame

# 运行数据处理脚本
python scripts/children_reading.py

echo "✅ 作业结束于: $(date)"
