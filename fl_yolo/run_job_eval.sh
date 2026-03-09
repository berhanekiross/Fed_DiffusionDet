#!/bin/bash
#SBATCH -A NAISS2025-5-233             # Project name
#SBATCH -p alvis                       # Alvis GPU partition
#SBATCH --gres=gpu:A40:1               # Request  GPUs
#SBATCH -t 0-1:10:00                  # Max runtime
#SBATCH --output=temp/job_stdout.log
#SBATCH --error=temp/job_stderr.log


mkdir -p temp/


# Run evaluation
# echo "Running DiffusionDet evaluation..."
# python diffdet_evaluation_script.py
# python loss_plot.py
# python metrics_0_cm.py
# python metrics_1.py
# python metrics_2.py
# python create_iid_kitti.py 
# python analyze_iid.py
# python fast_map.py
# python plot_metrics.py --data-dir Outputs/diffdet_kitti_FedAvg_r100 --output-dir comprehensive_analysis
# python symlink.py 
# python partition_real.py
# python partition_iid_yolo.py
python global_eval.py
# python plot_fl.py


