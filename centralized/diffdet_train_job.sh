#!/bin/bash
#SBATCH -A NAISS2025-5-233             # Project name
#SBATCH -p alvis                       # Alvis GPU partition
#SBATCH --gres=gpu:A100fat:2               # Request  GPUs
#SBATCH -t 99:00:00                  # Max runtime
#SBATCH --output=diffdet_temp_kitti/logs/job.out
#SBATCH --error=diffdet_temp_kitti/logs/job.err

mkdir -p diffdet_temp_kitti/logs

# Run DiffusionDet centralized training
# python train_net.py   --config-file configs/diffdet.kitti.res50.yaml

python train_net.py \
    --config-file configs2/diffdet.kitti.res50.yaml \
    --num-gpus 2 \
    SOLVER.IMS_PER_BATCH 32 \
    

# python -u train_net.py --config-file configs2/diffdet.zod.res50.yaml # -u for unbuffered output

# python -u train2.py --config-file configs2/diffdet.zod.res50.yaml
# python -u train2.py --config-file configs2/diffdet.zod.res50_copy.yaml
