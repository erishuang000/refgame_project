#!/bin/bash
#SBATCH --job-name=test_job
#SBATCH --output=../logs/test_%j.out
echo "Hello from SLURM!"
