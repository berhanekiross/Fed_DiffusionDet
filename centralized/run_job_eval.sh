#!/bin/bash
#SBATCH -A NAISS2025-5-233             # Project name
#SBATCH -p alvis                       # Alvis GPU partition
#SBATCH --gres=gpu:A40:1               # Request  GPUs
#SBATCH -t 0-1:10:00                  # Max runtime
#SBATCH --output=inferences/zod_inference_results/job.out
#SBATCH --error=inferences/zod_inference_results/err.err
#SBATCH --job-name=diffdet_eval        # Job name for easy identification


mkdir -p inferences/zod_inference_results


# ml load scikit-learn/1.3.1-gfbf-2023a

# Run evaluation
# echo "Running DiffusionDet evaluation..."
# python diffdet_evaluation_script.py
# python loss_plot.py
# python metrics_0_cm.py
# python metrics_1.py
# python metrics_2.py
python infer_zod.py 
# python kitti_dists.py




