
"""
metrics1.py - Generate F1, Precision, Recall vs Confidence plots and P-R curve
"""

import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import json
from sklearn.metrics import precision_recall_curve, average_precision_score
from collections import defaultdict

from detectron2.config import get_cfg
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.data import build_detection_train_loader
from detectron2.modeling import build_model
import yaml
from detectron2.config import CfgNode as CN
from register_kitti_splits import register_kitti_splits
from detectron2.data import MetadataCatalog
from detectron2.structures import BoxMode, Boxes, Instances
from detectron2.data import transforms as T
from detectron2.data import detection_utils as utils
import copy

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

class MetricsCollector:
    def __init__(self, num_classes, iou_threshold=0.5):
        self.num_classes = num_classes
        self.iou_threshold = iou_threshold
        self.all_predictions = []
        self.all_targets = []
        self.all_scores = []
        self.class_predictions = {i: [] for i in range(num_classes)}
        self.class_targets = {i: [] for i in range(num_classes)}
        self.class_scores = {i: [] for i in range(num_classes)}
        
    def compute_iou(self, box1, box2):
        """Compute IoU between two boxes [x1, y1, x2, y2]"""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        if x2 <= x1 or y2 <= y1:
            return 0.0
            
        intersection = (x2 - x1) * (y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def process_batch(self, inputs, outputs):
        """Process a batch and collect predictions with scores"""
        for input_data, output in zip(inputs, outputs):
            # Get ground truth
            gt_boxes = None
            gt_classes = None
            
            if "instances" in input_data:
                gt_instances = input_data["instances"]
                if hasattr(gt_instances, "gt_boxes") and hasattr(gt_instances, "gt_classes"):
                    gt_boxes = gt_instances.gt_boxes.tensor.cpu().numpy()
                    gt_classes = gt_instances.gt_classes.cpu().numpy()
            
            # Get predictions
            pred_boxes = None
            pred_classes = None
            pred_scores = None
            
            if "instances" in output and len(output["instances"]) > 0:
                pred_instances = output["instances"]
                if hasattr(pred_instances, "pred_boxes"):
                    pred_boxes = pred_instances.pred_boxes.tensor.cpu().numpy()
                    pred_classes = pred_instances.pred_classes.cpu().numpy()
                    pred_scores = pred_instances.scores.cpu().numpy()
            
            if gt_boxes is not None and gt_classes is not None:
                # For each ground truth box
                for gt_box, gt_class in zip(gt_boxes, gt_classes):
                    best_score = 0.0
                    best_match = False
                    
                    # Find best matching prediction
                    if pred_boxes is not None:
                        for pred_box, pred_class, pred_score in zip(pred_boxes, pred_classes, pred_scores):
                            if pred_class == gt_class:
                                iou = self.compute_iou(gt_box, pred_box)
                                if iou >= self.iou_threshold and pred_score > best_score:
                                    best_score = pred_score
                                    best_match = True
                    
                    # Store results for this ground truth
                    self.class_targets[gt_class].append(1)  # Positive ground truth
                    self.class_scores[gt_class].append(best_score)
                    
                # For each prediction, check if it's a false positive
                if pred_boxes is not None:
                    for pred_box, pred_class, pred_score in zip(pred_boxes, pred_classes, pred_scores):
                        matched = False
                        
                        # Check if this prediction matches any ground truth
                        for gt_box, gt_class in zip(gt_boxes, gt_classes):
                            if pred_class == gt_class:
                                iou = self.compute_iou(gt_box, pred_box)
                                if iou >= self.iou_threshold:
                                    matched = True
                                    break
                        
                        if not matched:
                            # False positive
                            self.class_targets[pred_class].append(0)
                            self.class_scores[pred_class].append(pred_score)
    
    def compute_metrics_at_confidence(self, confidence_thresholds):
        """Compute P, R, F1 for each class at different confidence thresholds"""
        results = {
            "confidence_thresholds": confidence_thresholds.tolist(),
            "per_class_metrics": {},
            "overall_metrics": {
                "precision": [],
                "recall": [],
                "f1": []
            }
        }
        
        for class_idx, class_name in enumerate(classes):
            if len(self.class_targets[class_idx]) == 0:
                continue
                
            y_true = np.array(self.class_targets[class_idx])
            y_scores = np.array(self.class_scores[class_idx])
            
            class_precision = []
            class_recall = []
            class_f1 = []
            
            for conf_thresh in confidence_thresholds:
                y_pred = (y_scores >= conf_thresh).astype(int)
                
                # Calculate metrics
                tp = np.sum((y_true == 1) & (y_pred == 1))
                fp = np.sum((y_true == 0) & (y_pred == 1))
                fn = np.sum((y_true == 1) & (y_pred == 0))
                
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
                
                class_precision.append(precision)
                class_recall.append(recall)
                class_f1.append(f1)
            
            results["per_class_metrics"][class_name] = {
                "precision": class_precision,
                "recall": class_recall,
                "f1": class_f1
            }
        
        # Compute overall metrics (macro average)
        all_precision = []
        all_recall = []
        all_f1 = []
        
        for i, conf_thresh in enumerate(confidence_thresholds):
            precisions = [results["per_class_metrics"][class_name]["precision"][i] 
                         for class_name in results["per_class_metrics"]]
            recalls = [results["per_class_metrics"][class_name]["recall"][i] 
                      for class_name in results["per_class_metrics"]]
            f1s = [results["per_class_metrics"][class_name]["f1"][i] 
                   for class_name in results["per_class_metrics"]]
            
            all_precision.append(np.mean(precisions))
            all_recall.append(np.mean(recalls))
            all_f1.append(np.mean(f1s))
        
        results["overall_metrics"]["precision"] = all_precision
        results["overall_metrics"]["recall"] = all_recall
        results["overall_metrics"]["f1"] = all_f1
        
        return results
    
    def compute_pr_curves(self):
        """Compute precision-recall curves for each class"""
        pr_results = {}
        
        for class_idx, class_name in enumerate(classes):
            if len(self.class_targets[class_idx]) == 0:
                continue
                
            y_true = np.array(self.class_targets[class_idx])
            y_scores = np.array(self.class_scores[class_idx])
            
            precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
            ap = average_precision_score(y_true, y_scores)
            
            pr_results[class_name] = {
                "precision": precision.tolist(),
                "recall": recall.tolist(),
                "thresholds": thresholds.tolist(),
                "average_precision": float(ap)
            }
        
        return pr_results

def plot_metrics_vs_confidence(results, output_dir):
    """Plot P, R, F1 vs confidence threshold"""
    confidence_thresholds = results["confidence_thresholds"]
    colors = plt.cm.tab10(np.linspace(0, 1, len(classes)))
    
    # 1. Precision vs Confidence
    plt.figure(figsize=(12, 8))
    for class_name, color in zip(classes, colors):
        if class_name in results["per_class_metrics"]:
            precision = results["per_class_metrics"][class_name]["precision"]
            plt.plot(confidence_thresholds, precision, label=class_name, color=color, linewidth=2)
    
    plt.plot(confidence_thresholds, results["overall_metrics"]["precision"], 
             label="Overall (macro avg)", color='black', linewidth=3, linestyle='--')
    
    plt.xlabel("Confidence Threshold", fontsize=12)
    plt.ylabel("Precision", fontsize=12)
    plt.title("Precision vs Confidence Threshold", fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "precision_vs_confidence.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Recall vs Confidence
    plt.figure(figsize=(12, 8))
    for class_name, color in zip(classes, colors):
        if class_name in results["per_class_metrics"]:
            recall = results["per_class_metrics"][class_name]["recall"]
            plt.plot(confidence_thresholds, recall, label=class_name, color=color, linewidth=2)
    
    plt.plot(confidence_thresholds, results["overall_metrics"]["recall"], 
             label="Overall (macro avg)", color='black', linewidth=3, linestyle='--')
    
    plt.xlabel("Confidence Threshold", fontsize=12)
    plt.ylabel("Recall", fontsize=12)
    plt.title("Recall vs Confidence Threshold", fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "recall_vs_confidence.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. F1 vs Confidence
    plt.figure(figsize=(12, 8))
    for class_name, color in zip(classes, colors):
        if class_name in results["per_class_metrics"]:
            f1 = results["per_class_metrics"][class_name]["f1"]
            plt.plot(confidence_thresholds, f1, label=class_name, color=color, linewidth=2)
    
    plt.plot(confidence_thresholds, results["overall_metrics"]["f1"], 
             label="Overall (macro avg)", color='black', linewidth=3, linestyle='--')
    
    plt.xlabel("Confidence Threshold", fontsize=12)
    plt.ylabel("F1 Score", fontsize=12)
    plt.title("F1 Score vs Confidence Threshold", fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "f1_vs_confidence.png"), dpi=300, bbox_inches='tight')
    plt.close()

def plot_pr_curves(pr_results, output_dir):
    """Plot Precision-Recall curves"""
    plt.figure(figsize=(12, 8))
    colors = plt.cm.tab10(np.linspace(0, 1, len(classes)))
    
    for class_name, color in zip(classes, colors):
        if class_name in pr_results:
            precision = pr_results[class_name]["precision"]
            recall = pr_results[class_name]["recall"]
            ap = pr_results[class_name]["average_precision"]
            
            plt.plot(recall, precision, label=f"{class_name} (AP={ap:.3f})", 
                    color=color, linewidth=2)
    
    plt.xlabel("Recall", fontsize=12)
    plt.ylabel("Precision", fontsize=12)
    plt.title("Precision-Recall Curves", fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "precision_recall_curve.png"), dpi=300, bbox_inches='tight')
    plt.close()

def main():
    # Configuration
    output_dir = "output/kitti_res50_90kIter/evaluation"
    os.makedirs(output_dir, exist_ok=True)
    
    from diffusiondet.config import add_diffusiondet_config
    cfg = get_cfg()
    add_diffusiondet_config(cfg)
    
    with open("configs2/diffdet.kitti.res50.yaml", "r") as f:
        cfg_dict = yaml.safe_load(f)
    
    cfg_dict.pop("MODEL_EMA", None)
    cfg.merge_from_other_cfg(CN(cfg_dict))
    cfg.MODEL.WEIGHTS = "output/kitti_res50_90kIter/model_final.pth"
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.0  # Set to 0 to get all predictions
    
    # Build model
    model = build_model(cfg)
    DetectionCheckpointer(model).load(cfg.MODEL.WEIGHTS)
    model.eval()
    
    # Create metrics collector
    metrics_collector = MetricsCollector(num_classes=len(classes), iou_threshold=0.5)
    
    # Setup data loader
    cfg_val = cfg.clone()
    cfg_val.DATASETS.TRAIN = ("kitti_val_split",)
    cfg_val.DATALOADER.NUM_WORKERS = 0
    
    eval_mapper = EvaluationDatasetMapper(cfg_val, is_train=False)
    val_loader = build_detection_train_loader(cfg_val, mapper=eval_mapper)
    
    print("✅ Starting metrics collection...")
    
    # Process validation data
    batch_count = 0
    max_batches = 1500
    
    with torch.no_grad():
        for idx, inputs in enumerate(val_loader):
            outputs = model(inputs)
            metrics_collector.process_batch(inputs, outputs)
            
            batch_count += 1
            if batch_count % 100 == 0:
                print(f"Processed {batch_count} batches...")
            
            if batch_count >= max_batches:
                break
    
    print("✅ Computing metrics...")
    
    # Define confidence thresholds
    confidence_thresholds = np.linspace(0.0, 1.0, 101)
    
    # Compute metrics
    metrics_results = metrics_collector.compute_metrics_at_confidence(confidence_thresholds)
    pr_results = metrics_collector.compute_pr_curves()
    
    # Combine results
    final_results = {
        "metrics_vs_confidence": metrics_results,
        "precision_recall_curves": pr_results
    }
    
    # Save results
    with open(os.path.join(output_dir, "metrics1.json"), "w") as f:
        json.dump(final_results, f, indent=4)
    
    # Generate plots
    plot_metrics_vs_confidence(metrics_results, output_dir)
    plot_pr_curves(pr_results, output_dir)
    
    print(f"Metrics1 analysis completed! Results saved to: {output_dir}")


if __name__ == "__main__":
    main()