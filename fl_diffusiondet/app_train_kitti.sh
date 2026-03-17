#!/bin/bash
#SBATCH -A NAISS2025-5-233
#SBATCH -p alvis
#SBATCH --gres=gpu:A100fat:1           # 
#SBATCH -t 0-55:00:00
#SBATCH --output=output_temp_iid/logs/job_stdout.log
#SBATCH --error=output_temp_iid/logs/job_stderr.log
#SBATCH --job-name=iid-DiffDet-kitti

# Safety: Stop if anything fails
set -e

# Load modules
module load CUDA/12.1.1
module load Python/3.11.3-GCCcore-12.3.0

source /mimer/NOBACKUP/groups/naiss2024-5-153/old_projects/Berhane/labelled_kitti/vyolo/bin/activate

cd /mimer/NOBACKUP/groups/naiss2024-5-153/old_projects/Berhane/labelled_kitti/fl_diffusiondet
# Create logs directory
# mkdir -p output_temp/logs
mkdir -p output_temp_iid/logs


export RAY_DEDUP_LOGS=0
# export RAY_verbose=1
flwr run . local-simulation-gpu



