#!/bin/bash
#SBATCH -A NAISS2025-5-233
#SBATCH -p alvis
#SBATCH --gpus-per-node=A40:1
#SBATCH --time=99:00:00 
#SBATCH --job-name=yolo1080p_zod
#SBATCH --output=yolo_kitti_temp/job.out
#SBATCH --error=yolo_kitti_temp/job.err


# run this command after setting up the environment:

# training
# yolo detect train \
#   model=yolov8n.pt \
#   data=configs/kitti_yolo.yaml \
#   epochs=100 \
#   patience=30 \
#   batch=64 \
#   imgsz=640 \
#   workers=16 \
#   amp=True \
#   cache=True \
#   device=0, \
#   project=yolo_kitti_temp \
#   name=train \
#   exist_ok=True
# # python -u train_yolo.py 
# # torchrun --nproc_per_node=4 train_yolo.py

#inference 
yolo detect predict \
  model=final_outputs/yolo_kitti_100ep/train/weights/latest.pt \
  source='samples_per_class/*/*.png' \
  imgsz=640 \
  conf=0.25 \
  save=True \
  save_txt=False \
  save_conf=True \
  project=final_outputs/yolo_kitti_100ep \
  name=preds100e5 \
  exist_ok=True \
  device=0



