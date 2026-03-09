#!/bin/bash
#SBATCH -A NAISS2025-5-233
#SBATCH -p alvis
#SBATCH --gres=gpu:A100:1           # 
#SBATCH -t 99:00:00
#SBATCH --output=output_yolo_temp/logs/job_stdout.log
#SBATCH --error=output_yolo_temp/logs/job_stderr.log
#SBATCH --job-name=DiffDet-FL-Batched


# Load modules
module load CUDA/12.1.1
module load Python/3.11.3-GCCcore-12.3.0

source /mimer/NOBACKUP/groups/naiss2024-5-153/old_projects/Berhane/labelled_kitti/vyolo/bin/activate

cd /mimer/NOBACKUP/groups/naiss2024-5-153/old_projects/Berhane/labelled_kitti/fl_yolo
# Create logs directory
mkdir -p output_yolo_temp/logs


export RAY_DEDUP_LOGS=0
# export RAY_verbose=1
flwr run . local-simulation-gpu


# after moved main dirctory to old_projects/Berhane..


# source /mimer/NOBACKUP/groups/naiss2024-5-153/old_projects/Berhane/labelled_kitti/setenv.sh

# cd /mimer/NOBACKUP/groups/naiss2024-5-153/old_projects/Berhane/labelled_kitti/fl_yolo
# mkdir -p output_yolo_temp/logs

# export RAY_DEDUP_LOGS=0
# flwr run . local-simulation-gpu