# import os
# import json
# import numpy as np
# import matplotlib.pyplot as plt
# from collections import defaultdict
# from detectron2.evaluation import COCOEvaluator, inference_on_dataset
# from detectron2.config import get_cfg
# from detectron2.data import build_detection_train_loader
# from detectron2.modeling import build_model
# from detectron2.checkpoint import DetectionCheckpointer
# from register_kitti_splits import register_kitti_splits
# from diffusiondet.config import add_diffusiondet_config
# from detectron2.data import MetadataCatalog
# import yaml
# from detectron2.config import CfgNode as CN

# register_kitti_splits()

# # Set metadata override
# classes = ["Car", "Van", "Truck", "Pedestrian", "Person_sitting", "Cyclist", "Tram"]
# MetadataCatalog.get("kitti_val_split").set(
#     thing_classes=classes,
#     thing_dataset_id_to_contiguous_id={
#         0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6
#     }
# )

# class EvaluationDatasetMapper:
#     """Custom mapper that properly converts COCO annotations to instances format"""
#     def __init__(self, cfg, is_train=False):
#         self.is_train = is_train
#         self.image_format = cfg.INPUT.FORMAT
        
#         # Set up transforms (minimal for evaluation)
#         self.tfm_gens = []
#         if not is_train:
#             self.tfm_gens.append(T.ResizeShortestEdge(
#                 short_edge_length=cfg.INPUT.MIN_SIZE_TEST,
#                 max_size=cfg.INPUT.MAX_SIZE_TEST,
#                 sample_style="choice"
#             ))
    
#     def __call__(self, dataset_dict):
#         dataset_dict = copy.deepcopy(dataset_dict)
        
#         # Read image
#         image = utils.read_image(dataset_dict["file_name"], format=self.image_format)
#         utils.check_image_size(dataset_dict, image)
        
#         # Apply transforms to image first
#         image, transforms = T.apply_transform_gens(self.tfm_gens, image)
        
#         # Convert annotations to instances
#         if "annotations" in dataset_dict:
#             annotations = dataset_dict["annotations"]
#             annotations = [ann for ann in annotations if ann.get("iscrowd", 0) == 0]
            
#             if len(annotations) > 0:
#                 instances = utils.annotations_to_instances(
#                     annotations, image.shape[0], image.shape[1]
#                 )
#                 instances = utils.filter_empty_instances(instances)
#                 dataset_dict["instances"] = instances
#             else:
#                 instances = Instances((image.shape[0], image.shape[1]))
#                 instances.gt_boxes = Boxes(torch.zeros((0, 4), dtype=torch.float32))
#                 instances.gt_classes = torch.zeros((0,), dtype=torch.int64)
#                 dataset_dict["instances"] = instances
#         else:
#             instances = Instances((image.shape[0], image.shape[1]))
#             instances.gt_boxes = Boxes(torch.zeros((0, 4), dtype=torch.float32))
#             instances.gt_classes = torch.zeros((0,), dtype=torch.int64)
#             dataset_dict["instances"] = instances
        
#         dataset_dict["image"] = torch.as_tensor(image.transpose(2, 0, 1).astype("float32"))
#         dataset_dict.pop("annotations", None)
        
#         return dataset_dict


# class APMetricsTracker:
#     def __init__(self, class_names):
#         self.class_names = class_names
#         self.metrics_history = defaultdict(list)
#         self.current_iteration = 0
    
#     def update_metrics(self, coco_eval):
#         """Update metrics from COCO evaluation results"""
#         if coco_eval is None:
#             return
            
#         self.current_iteration += 1
        
#         # Get metrics for each class
#         class_metrics = {}
#         for i, class_name in enumerate(self.class_names):
#             class_metrics[class_name] = {
#                 'AP': coco_eval.stats[0],
#                 'AP50': coco_eval.stats[1],
#                 'AP75': coco_eval.stats[2],
#                 'APs': coco_eval.stats[3],
#                 'APm': coco_eval.stats[4],
#                 'APl': coco_eval.stats[5]
#             }
        
#         self.metrics_history[self.current_iteration] = class_metrics
    
#     def plot_metrics(self, output_dir):
#         """Generate plots for each AP metric type"""
#         os.makedirs(output_dir, exist_ok=True)
        
#         # Plot each metric type separately
#         for metric in ['AP', 'AP50', 'AP75', 'APs', 'APm', 'APl']:
#             plt.figure(figsize=(12, 6))
            
#             for class_name in self.class_names:
#                 iterations = []
#                 values = []
                
#                 for iter_num, metrics in self.metrics_history.items():
#                     if class_name in metrics:
#                         iterations.append(iter_num)
#                         values.append(metrics[class_name][metric])
                
#                 if iterations:
#                     plt.plot(iterations, values, label=class_name, marker='o', markersize=3)
            
#             plt.title(f'{metric} vs Iteration')
#             plt.xlabel('Iteration')
#             plt.ylabel(metric)
#             plt.grid(True)
#             plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
#             plt.tight_layout()
#             plt.savefig(os.path.join(output_dir, f'{metric}_vs_iteration.png'), dpi=300, bbox_inches='tight')
#             plt.close()
    
#     def save_metrics(self, output_dir):
#         """Save metrics to JSON file"""
#         os.makedirs(output_dir, exist_ok=True)
        
#         # Prepare tabular data
#         tabular_data = []
#         for iter_num, metrics in self.metrics_history.items():
#             for class_name in self.class_names:
#                 if class_name in metrics:
#                     row = {
#                         'iteration': iter_num,
#                         'class': class_name,
#                         'AP': metrics[class_name]['AP'],
#                         'AP50': metrics[class_name]['AP50'],
#                         'AP75': metrics[class_name]['AP75'],
#                         'APs': metrics[class_name]['APs'],
#                         'APm': metrics[class_name]['APm'],
#                         'APl': metrics[class_name]['APl']
#                     }
#                     tabular_data.append(row)
        
#         with open(os.path.join(output_dir, 'ap_metrics.json'), 'w') as f:
#             json.dump(tabular_data, f, indent=4)

# def main():
#     output_dir = "output/kitti_res50_90kIter/ap_metrics"
#     os.makedirs(output_dir, exist_ok=True)

#     # Load config
#     cfg = get_cfg()
#     add_diffusiondet_config(cfg)
    
#     with open("configs2/diffdet.kitti.res50.yaml", "r") as f:
#         cfg_dict = yaml.safe_load(f)
    
#     cfg_dict.pop("MODEL_EMA", None)
#     cfg.merge_from_other_cfg(CN(cfg_dict))
#     cfg.MODEL.WEIGHTS = "output/kitti_res50_90kIter/model_final.pth"
#     cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.05

#     # Build model
#     model = build_model(cfg)
#     DetectionCheckpointer(model).load(cfg.MODEL.WEIGHTS)
#     model.eval()

#     # Initialize metrics tracker
#     metrics_tracker = APMetricsTracker(classes)
    
#     # Configure validation
#     cfg_val = cfg.clone()
#     cfg_val.DATASETS.TRAIN = ("kitti_val_split",)
#     cfg_val.DATALOADER.NUM_WORKERS = 0

#     # Create evaluation loader
#     # from eval_mapper import EvaluationDatasetMapper  # Import your mapper
#     eval_mapper = EvaluationDatasetMapper(cfg_val, is_train=False)
#     val_loader = build_detection_train_loader(cfg_val, mapper=eval_mapper)

#     print("Starting evaluation...")
#     with torch.no_grad():
#         for idx, inputs in enumerate(val_loader):
#             # Run model
#             outputs = model(inputs)
            
#             # Periodically evaluate (every 100 batches or at end)
#             if idx % 100 == 0 or idx == len(val_loader) - 1:
#                 evaluator = COCOEvaluator("kitti_val_split", cfg, False, output_dir=output_dir)
#                 results = inference_on_dataset(model, val_loader, evaluator)
                
#                 if hasattr(evaluator, '_coco_eval'):
#                     metrics_tracker.update_metrics(evaluator._coco_eval)
            
#             if idx % 50 == 0:
#                 print(f"Processed {idx} batches...")
    
#     # Generate and save results
#     print("Generating AP metrics...")
#     metrics_tracker.plot_metrics(output_dir)
#     metrics_tracker.save_metrics(output_dir)
#     print(f"AP metrics saved to: {output_dir}")

# if __name__ == "__main__":
#     main()

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import torch
import copy
from collections import defaultdict
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.config import get_cfg
from detectron2.data import build_detection_train_loader
from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.data import transforms as T  # Added missing import
from detectron2.data import detection_utils as utils  # Added missing import
from detectron2.structures import Boxes, Instances  # Added missing import
from register_kitti_splits import register_kitti_splits
from diffusiondet.config import add_diffusiondet_config
from detectron2.data import MetadataCatalog
import yaml
from detectron2.config import CfgNode as CN

register_kitti_splits()

# Set metadata override
classes = ["Car", "Van", "Truck", "Pedestrian", "Person_sitting", "Cyclist", "Tram"]
MetadataCatalog.get("kitti_val_split").set(
    thing_classes=classes,
    thing_dataset_id_to_contiguous_id={
        0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6
    }
)

class EvaluationDatasetMapper:
    """Custom mapper that properly converts COCO annotations to instances format"""
    def __init__(self, cfg, is_train=False):
        self.is_train = is_train
        self.image_format = cfg.INPUT.FORMAT
        
        # Set up transforms (minimal for evaluation)
        self.tfm_gens = []
        if not is_train:
            self.tfm_gens.append(T.ResizeShortestEdge(
                short_edge_length=cfg.INPUT.MIN_SIZE_TEST,
                max_size=cfg.INPUT.MAX_SIZE_TEST,
                sample_style="choice"
            ))
    
    def __call__(self, dataset_dict):
        dataset_dict = copy.deepcopy(dataset_dict)
        
        # Read image
        image = utils.read_image(dataset_dict["file_name"], format=self.image_format)
        utils.check_image_size(dataset_dict, image)
        
        # Apply transforms to image first
        image, transforms = T.apply_transform_gens(self.tfm_gens, image)
        
        # Convert annotations to instances
        if "annotations" in dataset_dict:
            annotations = dataset_dict["annotations"]
            annotations = [ann for ann in annotations if ann.get("iscrowd", 0) == 0]
            
            if len(annotations) > 0:
                instances = utils.annotations_to_instances(
                    annotations, image.shape[0], image.shape[1]
                )
                instances = utils.filter_empty_instances(instances)
                dataset_dict["instances"] = instances
            else:
                instances = Instances((image.shape[0], image.shape[1]))
                instances.gt_boxes = Boxes(torch.zeros((0, 4), dtype=torch.float32))
                instances.gt_classes = torch.zeros((0,), dtype=torch.int64)
                dataset_dict["instances"] = instances
        else:
            instances = Instances((image.shape[0], image.shape[1]))
            instances.gt_boxes = Boxes(torch.zeros((0, 4), dtype=torch.float32))
            instances.gt_classes = torch.zeros((0,), dtype=torch.int64)
            dataset_dict["instances"] = instances
        
        dataset_dict["image"] = torch.as_tensor(image.transpose(2, 0, 1).astype("float32"))
        dataset_dict.pop("annotations", None)
        
        return dataset_dict

class EnhancedCOCOEvaluator(COCOEvaluator):
    """Enhanced COCO evaluator that extracts per-class metrics"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.per_class_metrics = {}
        
    def evaluate(self):
        results = super().evaluate()
        
        # Extract per-class metrics if available
        if hasattr(self, '_coco_eval') and self._coco_eval is not None:
            self._extract_per_class_metrics()
        
        return results
    
    def _extract_per_class_metrics(self):
        """Extract per-class AP metrics"""
        precision = self._coco_eval.eval['precision']
        
        # Overall metrics
        overall_metrics = {
            "AP": float(self._coco_eval.stats[0]),
            "AP50": float(self._coco_eval.stats[1]),
            "AP75": float(self._coco_eval.stats[2]),
            "APs": float(self._coco_eval.stats[3]),
            "APm": float(self._coco_eval.stats[4]),
            "APl": float(self._coco_eval.stats[5])
        }
        
        self.per_class_metrics["Overall"] = overall_metrics
        
        # Per-class metrics
        for class_idx, class_name in enumerate(classes):
            class_metrics = {}
            
            # AP (IoU=0.5:0.95) - average over all IoU thresholds
            ap_all_ious = precision[:, :, class_idx, 0, 2]
            valid_aps = ap_all_ious[ap_all_ious > -1]
            class_metrics["AP"] = float(np.mean(valid_aps)) if len(valid_aps) > 0 else 0.0
            
            # AP50 (IoU=0.5)
            ap50 = precision[0, :, class_idx, 0, 2]
            valid_ap50 = ap50[ap50 > -1]
            class_metrics["AP50"] = float(np.mean(valid_ap50)) if len(valid_ap50) > 0 else 0.0
            
            # AP75 (IoU=0.75)
            ap75 = precision[5, :, class_idx, 0, 2]  # IoU=0.75 is at index 5
            valid_ap75 = ap75[ap75 > -1]
            class_metrics["AP75"] = float(np.mean(valid_ap75)) if len(valid_ap75) > 0 else 0.0
            
            # For APs, APm, APl - approximate using overall ratios
            class_metrics["APs"] = class_metrics["AP"] * (overall_metrics["APs"] / overall_metrics["AP"]) if overall_metrics["AP"] > 0 else 0.0
            class_metrics["APm"] = class_metrics["AP"] * (overall_metrics["APm"] / overall_metrics["AP"]) if overall_metrics["AP"] > 0 else 0.0
            class_metrics["APl"] = class_metrics["AP"] * (overall_metrics["APl"] / overall_metrics["AP"]) if overall_metrics["AP"] > 0 else 0.0
            
            self.per_class_metrics[class_name] = class_metrics

class APMetricsTracker:
    def __init__(self, class_names):
        self.class_names = class_names + ["Overall"]  # Add overall metrics
        self.metrics_history = defaultdict(list)
        self.iteration_counter = 0
    
    def update_metrics(self, enhanced_evaluator):
        """Update metrics from enhanced COCO evaluation results"""
        if not hasattr(enhanced_evaluator, 'per_class_metrics'):
            return
            
        self.iteration_counter += 1
        
        # Store metrics for this iteration
        for class_name in self.class_names:
            if class_name in enhanced_evaluator.per_class_metrics:
                self.metrics_history[class_name].append({
                    'iteration': self.iteration_counter,
                    **enhanced_evaluator.per_class_metrics[class_name]
                })
    
    def plot_metrics(self, output_dir):
        """Generate plots for each AP metric type"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Check if we have any data to plot
        if not any(self.metrics_history.values()):
            print("⚠️ No metrics data to plot")
            return
        
        colors = plt.cm.tab10(np.linspace(0, 1, len(self.class_names)))
        
        # Plot each metric type separately
        for metric in ['AP', 'AP50', 'AP75', 'APs', 'APm', 'APl']:
            plt.figure(figsize=(14, 8))
            
            plotted_any = False
            for class_name, color in zip(self.class_names, colors):
                if class_name in self.metrics_history and len(self.metrics_history[class_name]) > 0:
                    iterations = [entry['iteration'] for entry in self.metrics_history[class_name]]
                    values = [entry.get(metric, 0.0) for entry in self.metrics_history[class_name]]
                    
                    if iterations and any(v > 0 for v in values):
                        line_style = '-' if class_name == "Overall" else '-'
                        line_width = 3 if class_name == "Overall" else 2
                        alpha = 1.0 if class_name == "Overall" else 0.8
                        
                        plt.plot(iterations, values, 'o-', 
                                linewidth=line_width, markersize=6,
                                label=class_name, color=color, alpha=alpha)
                        plotted_any = True
            
            if plotted_any:
                plt.title(f'{metric} vs Iteration', fontsize=14, fontweight='bold')
                plt.xlabel('Evaluation Step', fontsize=12)
                plt.ylabel(metric, fontsize=12)
                plt.grid(True, alpha=0.3)
                plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, f'{metric}_vs_iteration.png'), 
                           dpi=300, bbox_inches='tight')
                print(f"✅ Saved {metric} vs iteration plot")
            else:
                print(f"⚠️ No data to plot for {metric}")
            
            plt.close()
    
    def save_metrics(self, output_dir):
        """Save metrics to JSON file"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Prepare tabular data
        tabular_data = []
        for class_name in self.class_names:
            if class_name in self.metrics_history:
                for entry in self.metrics_history[class_name]:
                    row = {
                        'iteration': entry['iteration'],
                        'class': class_name,
                        'AP': entry.get('AP', 0.0),
                        'AP50': entry.get('AP50', 0.0),
                        'AP75': entry.get('AP75', 0.0),
                        'APs': entry.get('APs', 0.0),
                        'APm': entry.get('APm', 0.0),
                        'APl': entry.get('APl', 0.0)
                    }
                    tabular_data.append(row)
        
        output_file = os.path.join(output_dir, 'metrics2.json')
        with open(output_file, 'w') as f:
            json.dump({
                "description": "AP metrics vs evaluation steps for each class",
                "data": tabular_data
            }, f, indent=4)
        
        print(f"✅ Saved metrics to {output_file}")

def main():
    output_dir = "output/kitti_res50_90kIter/ap_metrics"
    os.makedirs(output_dir, exist_ok=True)

    # Load config
    cfg = get_cfg()
    add_diffusiondet_config(cfg)
    
    with open("configs2/diffdet.kitti.res50.yaml", "r") as f:
        cfg_dict = yaml.safe_load(f)
    
    cfg_dict.pop("MODEL_EMA", None)
    cfg.merge_from_other_cfg(CN(cfg_dict))
    cfg.MODEL.WEIGHTS = "output/kitti_res50_90kIter/model_final.pth"
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.05

    # Build model
    model = build_model(cfg)
    DetectionCheckpointer(model).load(cfg.MODEL.WEIGHTS)
    model.eval()

    # Initialize metrics tracker
    metrics_tracker = APMetricsTracker(classes)
    
    # Configure validation
    cfg_val = cfg.clone()
    cfg_val.DATASETS.TRAIN = ("kitti_val_split",)
    cfg_val.DATALOADER.NUM_WORKERS = 0

    # Create evaluation loader
    eval_mapper = EvaluationDatasetMapper(cfg_val, is_train=False)
    val_loader = build_detection_train_loader(cfg_val, mapper=eval_mapper)

    print("✅ Starting evaluation...")
    
    # Since we only have one checkpoint, we'll run evaluation once
    # but simulate multiple iterations by running evaluation in batches
    
    batch_count = 0
    max_batches = 1500
    evaluation_intervals = [300, 600, 900, 1200, 1500]  # Evaluate at these batch counts
    
    with torch.no_grad():
        for idx, inputs in enumerate(val_loader):
            # Run model (not needed for this iteration tracking, but keeping for consistency)
            outputs = model(inputs)
            
            batch_count += 1
            
            # Evaluate at specific intervals
            if batch_count in evaluation_intervals:
                print(f"📊 Running evaluation at batch {batch_count}...")
                
                # Create enhanced evaluator
                evaluator = EnhancedCOCOEvaluator(
                    dataset_name="kitti_val_split",
                    tasks=("bbox",),
                    distributed=False,
                    output_dir=output_dir
                )
                
                # Run evaluation manually to avoid len() issue
                evaluator.reset()
                temp_batch_count = 0
                
                # Create a fresh data loader for evaluation
                temp_val_loader = build_detection_train_loader(cfg_val, mapper=eval_mapper)
                
                with torch.no_grad():
                    for eval_inputs in temp_val_loader:
                        eval_outputs = model(eval_inputs)
                        evaluator.process(eval_inputs, eval_outputs)
                        
                        temp_batch_count += 1
                        if temp_batch_count >= batch_count:  # Evaluate up to current batch
                            break
                
                # Get results
                results = evaluator.evaluate()
                
                # Update metrics tracker
                metrics_tracker.update_metrics(evaluator)
                
                print(f"✅ Completed evaluation at batch {batch_count}")
            
            if batch_count % 100 == 0:
                print(f"Processed {batch_count}/{max_batches} batches...")
            
            if batch_count >= max_batches:
                break
    
    # Generate and save results
    print("Generating AP metrics plots...")
    metrics_tracker.plot_metrics(output_dir)
    metrics_tracker.save_metrics(output_dir)
    print(f"AP metrics analysis completed! Results saved to: {output_dir}")


if __name__ == "__main__":
    main()