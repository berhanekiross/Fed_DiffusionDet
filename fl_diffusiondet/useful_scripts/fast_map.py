"""Quick mAP tracking from saved models"""

import torch
import matplotlib.pyplot as plt
import os
import glob
from fl.task import Net, coco_test
from register_kitti_fldata import register_kitti_splits, register_kitti_iid_splits

def quick_evaluate_models():
    # Auto-discover models from output_temp_iid directory
    model_dir = "output_temp_iid"
    model_pattern = os.path.join(model_dir, "FedAvg_round_*.pth")
    model_files = glob.glob(model_pattern)
    
    # Extract round numbers and create models dict
    models = {}
    for model_file in model_files:
        filename = os.path.basename(model_file)
        round_num = int(filename.split('_')[2].split('.')[0])
        models[round_num] = model_file
    
    # Sort by round number
    models = dict(sorted(models.items()))
    
    print(f"Found {len(models)} models: {list(models.keys())}")
    
    register_kitti_splits()
    register_kitti_iid_splits()
    net = Net()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    results = {}
    
    for round_num, model_path in models.items():
        print(f"[EVAL] Round {round_num}...")
        
        try:
            # Load model
            checkpoint = torch.load(model_path, map_location='cpu')
            if 'model_state_dict' in checkpoint:
                net.model.load_state_dict(checkpoint['model_state_dict'])
            else:
                net.model.load_state_dict(checkpoint)
            
            dataset_options = [
                "global_val",
                "fl_kitti_val_Car",
                "fl_kitti_iid_val_client_0"
            ]
            
            coco_results = None
            for dataset_name in dataset_options:
                try:
                    print(f"    Trying dataset: {dataset_name}")
                    coco_results = coco_test(net, dataset_name, device)
                    print(f"    Success with dataset: {dataset_name}")
                    break
                except Exception as e:
                    print(f"    Failed with {dataset_name}: {str(e)[:100]}...")
                    continue
            
            if coco_results is None:
                print(f"    All datasets failed, skipping round {round_num}")
                continue
            
            bbox_metrics = coco_results.get("bbox", {})
            
            if coco_results is None:
                print(f"    All datasets failed, skipping round {round_num}")
                continue

            bbox_metrics = coco_results.get("bbox", {})

            # DEBUG: Print available keys
            print(f"    Available metric keys: {list(bbox_metrics.keys())}")  
            
            results[round_num] = {
                "map": bbox_metrics.get("AP", 0.0),
                "map50": bbox_metrics.get("AP50", 0.0),
                # "map75": bbox_metrics.get("AP75", 0.0),
                "recall": bbox_metrics.get("AR@100", 0.0)  # AR @ maxDets=100
            }
            
            print(f"  mAP: {results[round_num]['map']:.4f} | mAP50: {results[round_num]['map50']:.4f} | Recall: {results[round_num]['recall']:.4f}")
            
        except Exception as e:
            print(f"  ERROR loading round {round_num}: {e}")
            continue
    
    if not results:
        print("No models successfully evaluated!")
        return {}
    
    # Extract metrics for plotting
    rounds = sorted(results.keys())
    maps = [results[r]["map"] for r in rounds]
    map50s = [results[r]["map50"] for r in rounds]
    # map75s = [results[r]["map75"] for r in rounds]
    recalls = [results[r]["recall"] for r in rounds]
    
    # Single plot with all metrics
    plt.figure(figsize=(10, 6))
    
    plt.plot(rounds, map50s, 'b-o', label='mAP@0.5', linewidth=2, markersize=6)
    plt.plot(rounds, maps, 'g-s', label='mAP@[0.5:0.95]', linewidth=2, markersize=6)
    plt.plot(rounds, recalls, 'r-^', label='Recall', linewidth=2, markersize=6)
    
    plt.xlabel('FL Round', fontsize=12)
    plt.ylabel('Score (%)', fontsize=12)
    plt.title('DiffusionDet FL Training Metrics', fontsize=14, fontweight='bold')
    plt.legend(loc='best', fontsize=11)
    plt.ylim(20,100)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plt.savefig('fedavg_metrics_combined.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"\nEvaluation Summary:")
    print(f"Rounds evaluated: {rounds}")
    print(f"Final mAP50: {map50s[-1]:.4f}")
    print(f"Final mAP@[0.5:0.95]: {maps[-1]:.4f}")
    print(f"Final Recall: {recalls[-1]:.4f}")
    print(f"Best mAP50: {max(map50s):.4f} at round {rounds[map50s.index(max(map50s))]}")
    
    return results

if __name__ == "__main__":
    results = quick_evaluate_models()