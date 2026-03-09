
#!/usr/bin/env python3
"""
Comprehensive YOLO inference and evaluation script for global models
Evaluates all rounds in parallel on A100 GPUs
"""

import os
import glob
import argparse
import multiprocessing as mp
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import pandas as pd
from ultralytics import YOLO
import torch
import numpy as np
import json
from datetime import datetime

class YOLOEvaluator:
    def __init__(self, weights_dir, val_data_yaml, output_dir, device='cuda'):
        self.weights_dir = Path(weights_dir)
        self.val_data_yaml = val_data_yaml
        self.output_dir = Path(output_dir)
        self.device = device
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Get all model files
        self.model_files = self._get_model_files()
        
        print(f"Found {len(self.model_files)} model files to evaluate")
        print(f"Using device: {device}")
        
    def _get_model_files(self):
        """Get all .pt model files sorted by round number"""
        model_files = list(self.weights_dir.glob("*.pt"))
        
        # Sort by round number
        def get_round_number(filename):
            stem = filename.stem
            if stem == 'latest':
                return float('inf')  # Put latest at the end
            elif stem.startswith('round_'):
                try:
                    return int(stem.split('_')[1])
                except:
                    return 0
            else:
                return 0
        
        model_files.sort(key=get_round_number)
        return model_files
    
    def evaluate_single_model(self, model_path, gpu_id=0):
        """Evaluate a single model and generate comprehensive metrics"""
        
        model_name = model_path.stem
        print(f"[GPU {gpu_id}] Evaluating {model_name}...")
        
        # Set device for this process
        device = f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu'
        
        try:
            # Load model
            print(f"[GPU {gpu_id}] Loading model from {model_path}")
            model = YOLO(str(model_path))
            
            # Debug model info
            print(f"[GPU {gpu_id}] Model loaded successfully")
            
            # Create output directory for this model
            model_output_dir = self.output_dir / model_name
            model_output_dir.mkdir(parents=True, exist_ok=True)
            
            # Run validation with comprehensive metrics
            print(f"[GPU {gpu_id}] Starting validation...")
            results = model.val(
                data=self.val_data_yaml,
                device=device,
                save_json=True,
                save_hybrid=True,
                conf=0.001,  # Low confidence threshold for comprehensive evaluation
                iou=0.6,
                max_det=300,
                half=False,  # Disable FP16 to avoid potential issues
                plots=True,
                save_txt=False,
                save_conf=True,
                project=str(model_output_dir),
                name='evaluation',
                exist_ok=True,
                verbose=False,  # Reduce verbosity
                split='val'
            )
            
            print(f"[GPU {gpu_id}] Validation completed, extracting metrics...")
            
            # Debug: Print results structure
            print(f"[GPU {gpu_id}] Results type: {type(results)}")
            if hasattr(results, 'box'):
                print(f"[GPU {gpu_id}] Box results available")
            else:
                print(f"[GPU {gpu_id}] No box results found")
                
            # Extract key metrics with safe attribute access
            metrics = {
                'model': model_name,
                'mAP50': getattr(results.box, 'map50', 0.0) if hasattr(results, 'box') else 0.0,
                'mAP50-95': getattr(results.box, 'map', 0.0) if hasattr(results, 'box') else 0.0,
                'precision': getattr(results.box, 'mp', 0.0) if hasattr(results, 'box') else 0.0,
                'recall': getattr(results.box, 'mr', 0.0) if hasattr(results, 'box') else 0.0,
                'fitness': getattr(results, 'fitness', 0.0),
            }
            
            # Add timing metrics if available
            if hasattr(results, 'speed'):
                metrics.update({
                    'inference_time': getattr(results.speed, 'inference', 0.0),
                    'nms_time': getattr(results.speed, 'nms', 0.0),
                })
                metrics['total_time'] = metrics['inference_time'] + metrics['nms_time']
            else:
                metrics.update({
                    'inference_time': 0.0,
                    'nms_time': 0.0,
                    'total_time': 0.0
                })
            
            # Add per-class metrics if available
            if hasattr(results, 'box') and hasattr(results.box, 'ap') and results.box.ap is not None:
                class_names = ['Car', 'Van', 'Truck', 'Pedestrian', 'Person_sitting', 'Cyclist', 'Tram']
                try:
                    ap_array = results.box.ap
                    for i, class_name in enumerate(class_names):
                        if i < len(ap_array):
                            if len(ap_array[i]) > 0:
                                metrics[f'AP50_{class_name}'] = float(ap_array[i][0])
                                metrics[f'AP50-95_{class_name}'] = float(np.mean(ap_array[i]))
                            else:
                                metrics[f'AP50_{class_name}'] = 0.0
                                metrics[f'AP50-95_{class_name}'] = 0.0
                except Exception as e:
                    print(f"[GPU {gpu_id}] Warning: Could not extract per-class metrics: {e}")
            
            # Convert all metrics to float to ensure JSON serialization
            for key, value in metrics.items():
                if isinstance(value, (np.ndarray, np.float32, np.float64)):
                    metrics[key] = float(value)
                elif hasattr(value, 'item'):  # torch tensors
                    metrics[key] = float(value.item())
            
            # Save metrics to JSON
            metrics_file = model_output_dir / 'evaluation' / 'metrics.json'
            metrics_file.parent.mkdir(parents=True, exist_ok=True)
            with open(metrics_file, 'w') as f:
                json.dump(metrics, f, indent=2, default=float)
            
            print(f"[GPU {gpu_id}] ✅ {model_name}: mAP50={metrics['mAP50']:.3f}, mAP50-95={metrics['mAP50-95']:.3f}")
            
            return metrics
            
        except Exception as e:
            import traceback
            error_msg = f"Error evaluating {model_name}: {e}"
            print(f"[GPU {gpu_id}] ❌ {error_msg}")
            print(f"[GPU {gpu_id}] Traceback: {traceback.format_exc()}")
            
            return {
                'model': model_name,
                'error': str(e),
                'mAP50': 0.0,
                'mAP50-95': 0.0,
                'precision': 0.0,
                'recall': 0.0,
                'fitness': 0.0,
                'inference_time': 0.0,
                'nms_time': 0.0,
                'total_time': 0.0
            }

def evaluate_model_wrapper(args):
    """Wrapper function for multiprocessing"""
    evaluator, model_path, gpu_id = args
    return evaluator.evaluate_single_model(model_path, gpu_id)

def run_parallel_evaluation(weights_dir, val_data_yaml, output_dir, max_workers=None):
    """Run evaluation in parallel across multiple GPUs"""
    
    # Initialize evaluator
    evaluator = YOLOEvaluator(weights_dir, val_data_yaml, output_dir)
    
    # Determine number of workers
    if max_workers is None:
        num_gpus = torch.cuda.device_count()
        max_workers = min(num_gpus, len(evaluator.model_files))
        if max_workers == 0:
            max_workers = 1  # Fallback to CPU
    
    print(f"Running evaluation with {max_workers} parallel workers")
    print(f"Available GPUs: {torch.cuda.device_count()}")
    
    # Prepare arguments for parallel processing
    args_list = []
    for i, model_path in enumerate(evaluator.model_files):
        gpu_id = i % torch.cuda.device_count() if torch.cuda.is_available() else 0
        args_list.append((evaluator, model_path, gpu_id))
    
    # Run evaluation in parallel
    all_metrics = []
    
    if torch.cuda.is_available() and max_workers > 1:
        # Use ProcessPoolExecutor for GPU parallelization
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            results = executor.map(evaluate_model_wrapper, args_list)
            all_metrics = list(results)
    else:
        # Sequential execution for CPU or single GPU
        for args in args_list:
            result = evaluate_model_wrapper(args)
            all_metrics.append(result)
    
    return all_metrics, evaluator

def generate_summary_report(all_metrics, output_dir):
    """Generate comprehensive summary report"""
    
    output_dir = Path(output_dir)
    
    # Convert to DataFrame
    df = pd.DataFrame(all_metrics)
    
    # Sort by round number
    def extract_round_number(model_name):
        if model_name == 'latest':
            return float('inf')
        elif model_name.startswith('round_'):
            try:
                return int(model_name.split('_')[1])
            except:
                return 0
        return 0
    
    df['round_num'] = df['model'].apply(extract_round_number)
    df = df.sort_values('round_num').reset_index(drop=True)
    
    # Save detailed results
    results_file = output_dir / 'comprehensive_results.csv'
    df.to_csv(results_file, index=False)
    print(f"📊 Detailed results saved to: {results_file}")
    
    # Generate summary statistics
    successful_df = df[~df['error'].isna()] if 'error' in df.columns else df
    
    summary_stats = {
        'total_models': len(df),
        'successful_evaluations': len(successful_df),
    }
    
    # Only calculate best models if we have successful evaluations
    if len(successful_df) > 0:
        summary_stats.update({
            'best_mAP50': {
                'value': float(successful_df['mAP50'].max()),
                'model': successful_df.loc[successful_df['mAP50'].idxmax(), 'model']
            },
            'best_mAP50_95': {
                'value': float(successful_df['mAP50-95'].max()),
                'model': successful_df.loc[successful_df['mAP50-95'].idxmax(), 'model']
            },
        })
        
        # Check for latest model
        latest_models = df[df['model'] == 'latest']
        if len(latest_models) > 0:
            summary_stats['final_round_mAP50'] = float(latest_models['mAP50'].iloc[0])
        else:
            summary_stats['final_round_mAP50'] = None
    else:
        summary_stats.update({
            'best_mAP50': {'value': 0.0, 'model': 'None'},
            'best_mAP50_95': {'value': 0.0, 'model': 'None'},
            'final_round_mAP50': None
        })
    
    # Save summary
    summary_file = output_dir / 'evaluation_summary.json'
    with open(summary_file, 'w') as f:
        json.dump(summary_stats, f, indent=2, default=float)
    
    # Print summary
    print("\n" + "="*60)
    print("📈 EVALUATION SUMMARY")
    print("="*60)
    print(f"Total models evaluated: {summary_stats['total_models']}")
    print(f"Successful evaluations: {summary_stats['successful_evaluations']}")
    print(f"Best mAP@50: {summary_stats['best_mAP50']['value']:.3f} ({summary_stats['best_mAP50']['model']})")
    print(f"Best mAP@50-95: {summary_stats['best_mAP50_95']['value']:.3f} ({summary_stats['best_mAP50_95']['model']})")
    if summary_stats['final_round_mAP50']:
        print(f"Final round mAP@50: {summary_stats['final_round_mAP50']:.3f}")
    print("="*60)
    
    return df, summary_stats

def main():
    parser = argparse.ArgumentParser(description='Comprehensive YOLO Model Evaluation')
    parser.add_argument('--weights_dir', type=str, 
                       default='output_yolo_temp/global/weights',
                       help='Directory containing model weights')
    parser.add_argument('--val_data', type=str,
                       default='/mimer/NOBACKUP/groups/naiss2024-5-153/old_projects/Berhane/labelled_kitti/global_val/data.yaml',
                       help='Path to validation data YAML')
    parser.add_argument('--output_dir', type=str,
                       default='output_yolo_temp/global/evaluation_metrics',
                       help='Output directory for evaluation results')
    parser.add_argument('--max_workers', type=int, default=None,
                       help='Maximum number of parallel workers (default: auto-detect)')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use for evaluation (cuda/cpu)')
    
    args = parser.parse_args()
    
    print("🚀 Starting YOLO Model Evaluation")
    print(f"Weights directory: {args.weights_dir}")
    print(f"Validation data: {args.val_data}")
    print(f"Output directory: {args.output_dir}")
    print(f"Device: {args.device}")
    
    # Verify paths exist
    if not os.path.exists(args.weights_dir):
        print(f"❌ Error: Weights directory not found: {args.weights_dir}")
        return
    
    if not os.path.exists(args.val_data):
        print(f"❌ Error: Validation data file not found: {args.val_data}")
        return
    
    start_time = datetime.now()
    
    # Run parallel evaluation
    all_metrics, evaluator = run_parallel_evaluation(
        args.weights_dir, 
        args.val_data, 
        args.output_dir, 
        args.max_workers
    )
    
    # Generate summary report
    df, summary_stats = generate_summary_report(all_metrics, args.output_dir)
    
    end_time = datetime.now()
    duration = end_time - start_time
    
    print(f"⏱️  Total evaluation time: {duration}")
    print(f"📁 Results saved to: {args.output_dir}")
    print("✅ Evaluation completed successfully!")

if __name__ == "__main__":
    main()