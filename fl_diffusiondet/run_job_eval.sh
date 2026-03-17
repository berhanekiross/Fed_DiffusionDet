#!/bin/bash
#SBATCH -A NAISS2025-5-233             # Project name
#SBATCH -p alvis                       # Alvis GPU partition
#SBATCH --gres=gpu:A100:1               # Request  GPUs
#SBATCH -t 0-1:10:00                  # Max runtime
#SBATCH --output=temp/job_stdout.log
#SBATCH --error=temp/job_stderr.log


# mkdir -p output/kitti_res50_90kIter/logs


# ml load scikit-learn/1.3.1-gfbf-2023a

# python diffdet_preds.py


# python diffdet_preds.py --samples-dir /mimer/NOBACKUP/groups/naiss2024-5-153/old_projects/Berhane/labelled_kitti/difdet/samples_per_class


# python diffdet_preds.py \
#   --config pyproject.toml \
#   --samples-dir /mimer/NOBACKUP/groups/naiss2024-5-153/old_projects/Berhane/labelled_kitti/difdet/samples_per_class \
#   --confidence 0.25 \
#   --max-samples 10


# python diffdet_preds.py --config pyproject.toml --confidence 0.01
# python diffdet_preds.py --output-dir ./output_fedprox --confidence 0.01 --max-samples 3



# Run evaluation
# echo "Running DiffusionDet evaluation..."
# python diffdet_evaluation_script.py
# python loss_plot.py
# python metrics_0_cm.py
# python metrics_1.py
# python metrics_2.py
# python create_iid_kitti.py 
# python analyze_iid.py
python fast_map.py
# python plot_metrics.py --data-dir Outputs/diffdet_kitti_FedAvg_r100 --output-dir comprehensive_analysis
# python symlink.py 
# python infer_diffdet2.py


