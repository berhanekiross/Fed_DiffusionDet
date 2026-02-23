
# import os
# import numpy as np
# import torch
# import matplotlib.pyplot as plt
# import json
# from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
# from collections import defaultdict

# from detectron2.config import get_cfg
# from detectron2.checkpoint import DetectionCheckpointer
# from detectron2.data import build_detection_test_loader, build_detection_train_loader
# from detectron2.modeling import build_model
# import yaml
# from detectron2.config import CfgNode as CN
# from register_kitti_splits import register_kitti_splits
# from detectron2.data import MetadataCatalog
# from detectron2.structures import BoxMode, Boxes, Instances
# from detectron2.data import transforms as T
# from detectron2.data import detection_utils as utils
# import copy

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
#     """
#     Custom mapper that properly converts COCO annotations to instances format
#     with fixed transform handling
#     """
#     def __init__(self, cfg, is_train=False):
#         self.is_train = is_train
#         self.image_format = cfg.INPUT.FORMAT
        
#         # Set up transforms (minimal for evaluation)
#         self.tfm_gens = []
#         if not is_train:
#             # Only resize for evaluation, no augmentation
#             self.tfm_gens.append(T.ResizeShortestEdge(
#                 short_edge_length=cfg.INPUT.MIN_SIZE_TEST,
#                 max_size=cfg.INPUT.MAX_SIZE_TEST,
#                 sample_style="choice"
#             ))
    
#     def __call__(self, dataset_dict):
#         """
#         Convert dataset_dict with 'annotations' to format with 'instances'
#         with proper transform handling
#         """
#         dataset_dict = copy.deepcopy(dataset_dict)
        
#         # Read image
#         image = utils.read_image(dataset_dict["file_name"], format=self.image_format)
#         utils.check_image_size(dataset_dict, image)
        
#         # Apply transforms to image first
#         image, transforms = T.apply_transform_gens(self.tfm_gens, image)
        
#         # Convert annotations to instances AFTER applying transforms
#         if "annotations" in dataset_dict:
#             annotations = dataset_dict["annotations"]
            
#             # Filter out annotations with iscrowd=1 if any
#             annotations = [ann for ann in annotations if ann.get("iscrowd", 0) == 0]
            
#             if len(annotations) > 0:
#                 # Create instances object from TRANSFORMED image dimensions
#                 instances = utils.annotations_to_instances(
#                     annotations, image.shape[0], image.shape[1]
#                 )
                
#                 # No need to transform instances since we created them after image transform
#                 # Filter out empty instances
#                 instances = utils.filter_empty_instances(instances)
#                 dataset_dict["instances"] = instances
#             else:
#                 # Create empty instances if no annotations
#                 instances = Instances((image.shape[0], image.shape[1]))
#                 instances.gt_boxes = Boxes(torch.zeros((0, 4), dtype=torch.float32))
#                 instances.gt_classes = torch.zeros((0,), dtype=torch.int64)
#                 dataset_dict["instances"] = instances
#         else:
#             # Create empty instances if no annotations key
#             instances = Instances((image.shape[0], image.shape[1]))
#             instances.gt_boxes = Boxes(torch.zeros((0, 4), dtype=torch.float32))
#             instances.gt_classes = torch.zeros((0,), dtype=torch.int64)
#             dataset_dict["instances"] = instances
        
#         # Set the transformed image
#         dataset_dict["image"] = torch.as_tensor(image.transpose(2, 0, 1).astype("float32"))
        
#         # Remove annotations as we now have instances
#         dataset_dict.pop("annotations", None)
        
#         return dataset_dict

# class ConfusionMatrixCollector:
#     def __init__(self, num_classes, conf_threshold=0.05, iou_threshold=0.5):
#         self.num_classes = num_classes
#         self.conf_threshold = conf_threshold
#         self.iou_threshold = iou_threshold
#         self.all_predictions = []
#         self.all_targets = []
#         self.debug_counts = {
#             'total_batches': 0,
#             'batches_with_gt': 0,
#             'batches_with_pred': 0,
#             'total_gt_boxes': 0,
#             'total_pred_boxes': 0,
#             'valid_pred_boxes': 0,
#             'matched_pairs': 0
#         }
        
#     def compute_iou(self, box1, box2):
#         """Compute IoU between two boxes [x1, y1, x2, y2]"""
#         # Get intersection coordinates
#         x1 = max(box1[0], box2[0])
#         y1 = max(box1[1], box2[1])
#         x2 = min(box1[2], box2[2])
#         y2 = min(box1[3], box2[3])
        
#         # Compute intersection area
#         if x2 <= x1 or y2 <= y1:
#             return 0.0
            
#         intersection = (x2 - x1) * (y2 - y1)
        
#         # Compute union area
#         area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
#         area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
#         union = area1 + area2 - intersection
        
#         return intersection / union if union > 0 else 0.0
    
#     def process_batch(self, inputs, outputs, debug=False):
#         """Process a batch of inputs and outputs"""
#         self.debug_counts['total_batches'] += 1
        
#         if debug and self.debug_counts['total_batches'] <= 3:
#             print(f"\n🔍 DEBUG: Batch {self.debug_counts['total_batches']}")
#             print("-" * 50)
            
#             if len(inputs) > 0:
#                 sample_input = inputs[0]
#                 print(f"Input keys: {list(sample_input.keys())}")
                
#                 if "instances" in sample_input:
#                     gt_instances = sample_input["instances"]
#                     print(f"✅ Found GT instances!")
#                     print(f"GT instances type: {type(gt_instances)}")
#                     print(f"GT boxes shape: {gt_instances.gt_boxes.tensor.shape}")
#                     print(f"GT classes shape: {gt_instances.gt_classes.shape}")
#                     print(f"GT classes sample: {gt_instances.gt_classes[:10]}")
#                     print(f"GT classes unique: {torch.unique(gt_instances.gt_classes)}")
#                 else:
#                     print("❌ No 'instances' in input")
                    
#             if len(outputs) > 0:
#                 sample_output = outputs[0]
#                 print(f"Output keys: {list(sample_output.keys())}")
#                 if "instances" in sample_output:
#                     pred_instances = sample_output["instances"]
#                     print(f"Pred instances length: {len(pred_instances)}")
#                     if len(pred_instances) > 0:
#                         print(f"Pred scores > 0.05: {(pred_instances.scores > 0.05).sum()}")
        
#         for input_data, output in zip(inputs, outputs):
#             # Check for ground truth
#             gt_boxes = None
#             gt_classes = None
            
#             if "instances" in input_data:
#                 gt_instances = input_data["instances"]
#                 if hasattr(gt_instances, "gt_boxes") and hasattr(gt_instances, "gt_classes"):
#                     gt_boxes = gt_instances.gt_boxes.tensor
#                     gt_classes = gt_instances.gt_classes
#                     self.debug_counts['batches_with_gt'] += 1
#                     self.debug_counts['total_gt_boxes'] += len(gt_boxes)
            
#             # Check for predictions
#             pred_boxes = None
#             pred_classes = None
#             pred_scores = None
            
#             if "instances" in output:
#                 pred_instances = output["instances"]
#                 if len(pred_instances) > 0:
#                     self.debug_counts['batches_with_pred'] += 1
#                     self.debug_counts['total_pred_boxes'] += len(pred_instances)
                    
#                     if hasattr(pred_instances, "pred_boxes") and hasattr(pred_instances, "pred_classes") and hasattr(pred_instances, "scores"):
#                         pred_boxes = pred_instances.pred_boxes.tensor
#                         pred_classes = pred_instances.pred_classes
#                         pred_scores = pred_instances.scores
                        
#                         # Count valid predictions
#                         valid_preds = pred_scores >= self.conf_threshold
#                         self.debug_counts['valid_pred_boxes'] += valid_preds.sum().item()
            
#             # If we have both GT and predictions, process them
#             if gt_boxes is not None and gt_classes is not None:
#                 if pred_boxes is not None and pred_classes is not None and pred_scores is not None:
#                     # Filter predictions by confidence
#                     valid_preds = pred_scores >= self.conf_threshold
#                     if valid_preds.sum() > 0:
#                         valid_pred_boxes = pred_boxes[valid_preds]
#                         valid_pred_classes = pred_classes[valid_preds]
#                         valid_pred_scores = pred_scores[valid_preds]
                        
#                         # Simple matching: for each GT, find best pred
#                         for gt_box, gt_class in zip(gt_boxes.cpu().numpy(), gt_classes.cpu().numpy()):
#                             best_iou = 0
#                             best_pred_class = -1  # Background
                            
#                             for pred_box, pred_class in zip(valid_pred_boxes.cpu().numpy(), valid_pred_classes.cpu().numpy()):
#                                 iou = self.compute_iou(gt_box, pred_box)
#                                 if iou > best_iou and iou >= self.iou_threshold:
#                                     best_iou = iou
#                                     best_pred_class = pred_class
                            
#                             self.all_targets.append(gt_class)
#                             self.all_predictions.append(best_pred_class)
                            
#                             if best_pred_class != -1:
#                                 self.debug_counts['matched_pairs'] += 1
#                     else:
#                         # No valid predictions - all GT are false negatives
#                         for gt_class in gt_classes.cpu().numpy():
#                             self.all_targets.append(gt_class)
#                             self.all_predictions.append(-1)
#                 else:
#                     # No predictions - all GT are false negatives
#                     for gt_class in gt_classes.cpu().numpy():
#                         self.all_targets.append(gt_class)
#                         self.all_predictions.append(-1)
    
#     def print_debug_stats(self):
#         """Print debugging statistics"""
#         print(f"\n📊 DEBUG STATISTICS:")
#         print(f"Total batches processed: {self.debug_counts['total_batches']}")
#         print(f"Batches with GT: {self.debug_counts['batches_with_gt']}")
#         print(f"Batches with predictions: {self.debug_counts['batches_with_pred']}")
#         print(f"Total GT boxes: {self.debug_counts['total_gt_boxes']}")
#         print(f"Total pred boxes: {self.debug_counts['total_pred_boxes']}")
#         print(f"Valid pred boxes (conf > {self.conf_threshold}): {self.debug_counts['valid_pred_boxes']}")
#         print(f"Matched pairs: {self.debug_counts['matched_pairs']}")
#         print(f"Collected predictions: {len(self.all_predictions)}")
#         print(f"Collected targets: {len(self.all_targets)}")
    
#     def generate_confusion_matrix(self, output_dir, class_names):
#         """Generate and save confusion matrix"""
#         print(f"Collected {len(self.all_predictions)} predictions and {len(self.all_targets)} targets")
        
#         if len(self.all_predictions) == 0 or len(self.all_targets) == 0:
#             print("⚠️ No data collected for confusion matrix")
#             return
        
#         # Clean data - remove background predictions for valid comparison
#         valid_pairs = [(t, p) for t, p in zip(self.all_targets, self.all_predictions) 
#                       if t >= 0 and p >= 0]
        
#         if len(valid_pairs) == 0:
#             print("⚠️ No valid prediction-target pairs found")
#             print(f"   - Total pairs: {len(self.all_targets)}")
#             print(f"   - Background predictions: {sum(1 for p in self.all_predictions if p == -1)}")
#             return
        
#         clean_targets, clean_predictions = zip(*valid_pairs)
        
#         # Generate confusion matrix
#         cm = confusion_matrix(clean_targets, clean_predictions, 
#                             labels=list(range(self.num_classes)))
        
#         # Normalized confusion matrix
#         cm_normalized = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-8)
        
#         # Save matrices
#         np.save(os.path.join(output_dir, "confusion_matrix.npy"), cm)
#         np.save(os.path.join(output_dir, "confusion_matrix_normalized.npy"), cm_normalized)
        
#         # Plot confusion matrices
#         # 1. Raw counts
#         plt.figure(figsize=(10, 8))
#         disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
#         disp.plot(cmap='Blues', values_format='d')
#         plt.title("Confusion Matrix (Actual Counts)", fontsize=14, fontweight='bold')
#         plt.xticks(rotation=45, ha='right')
#         plt.yticks(rotation=0)
#         plt.tight_layout()
#         plt.savefig(os.path.join(output_dir, "confusion_matrix.png"), dpi=300, bbox_inches='tight')
#         plt.close()
        
#         # 2. Normalized
#         plt.figure(figsize=(10, 8))
#         disp = ConfusionMatrixDisplay(confusion_matrix=cm_normalized, display_labels=class_names)
#         disp.plot(cmap='Blues', values_format='.3f')
#         plt.title("Normalized Confusion Matrix", fontsize=14, fontweight='bold')
#         plt.xticks(rotation=45, ha='right')
#         plt.yticks(rotation=0)
#         plt.tight_layout()
#         plt.savefig(os.path.join(output_dir, "confusion_matrix_normalized.png"), dpi=300, bbox_inches='tight')
#         plt.close()
        
#         print(f"✅ Generated confusion matrix with {len(clean_targets)} valid samples")
        
#         # Print some statistics
#         print("\nConfusion Matrix Statistics:")
#         print(f"Total valid predictions: {len(clean_targets)}")
#         print(f"Accuracy: {np.trace(cm) / np.sum(cm):.3f}")
        
#         # Per-class accuracy
#         class_accuracies = np.diag(cm) / (np.sum(cm, axis=1) + 1e-8)
#         for i, (class_name, acc) in enumerate(zip(class_names, class_accuracies)):
#             print(f"{class_name}: {acc:.3f} ({np.sum(cm[i, :])} samples)")

# # Configuration and evaluation
# output_dir = "output/kitti_res50_90kIter/evaluation"
# os.makedirs(output_dir, exist_ok=True)

# from diffusiondet.config import add_diffusiondet_config
# cfg = get_cfg()
# add_diffusiondet_config(cfg)

# with open("configs2/diffdet.kitti.res50.yaml", "r") as f:
#     cfg_dict = yaml.safe_load(f)

# cfg_dict.pop("MODEL_EMA", None)
# cfg.merge_from_other_cfg(CN(cfg_dict))
# cfg.MODEL.WEIGHTS = "output/kitti_res50_90kIter/model_final.pth"
# cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.05

# # Build model
# model = build_model(cfg)
# DetectionCheckpointer(model).load(cfg.MODEL.WEIGHTS)
# model.eval()

# # Create confusion matrix collector
# cm_collector = ConfusionMatrixCollector(
#     num_classes=len(classes),
#     conf_threshold=0.05,  # Same as model threshold
#     iou_threshold=0.5     # Standard IoU threshold
# )

# print(f"✅ Starting confusion matrix generation...")
# print(f"Using confidence threshold: {cm_collector.conf_threshold}")
# print(f"Using IoU threshold: {cm_collector.iou_threshold}")

# # 🔧 FIXED: Proper transform handling in custom mapper
# cfg_val = cfg.clone()
# cfg_val.DATASETS.TRAIN = ("kitti_val_split",)
# cfg_val.DATALOADER.NUM_WORKERS = 0

# # Use our custom mapper with fixed transform handling
# eval_mapper = EvaluationDatasetMapper(cfg_val, is_train=False)
# val_loader = build_detection_train_loader(cfg_val, mapper=eval_mapper)

# print("✅ Created validation data loader with fixed transform handling")

# # Process validation data
# batch_count = 0
# max_batches = 1500  # Approximate validation set size

# print(f"📊 Starting evaluation loop...")

# with torch.no_grad():
#     for idx, inputs in enumerate(val_loader):
#         outputs = model(inputs)
        
#         # Debug first few batches
#         debug_this_batch = idx < 3
#         cm_collector.process_batch(inputs, outputs, debug=debug_this_batch)
        
#         batch_count += 1
        
#         if (batch_count) % 100 == 0:
#             print(f"Processed {batch_count} batches...")
            
#         # Stop after processing validation set
#         if batch_count >= max_batches:
#             print(f"Reached maximum batches ({max_batches}), stopping...")
#             break

# print(f"✅ Evaluation completed after {batch_count} batches!")

# # Print debug statistics
# cm_collector.print_debug_stats()

# # Generate confusion matrix
# cm_collector.generate_confusion_matrix(output_dir, classes)

# print(f"\n✅ Confusion matrix saved to: {output_dir}")

import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import json
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from collections import defaultdict

from detectron2.config import get_cfg
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.data import build_detection_test_loader, build_detection_train_loader
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
    """
    Custom mapper that properly converts COCO annotations to instances format
    with fixed transform handling
    """
    def __init__(self, cfg, is_train=False):
        self.is_train = is_train
        self.image_format = cfg.INPUT.FORMAT
        
        # Set up transforms (minimal for evaluation)
        self.tfm_gens = []
        if not is_train:
            # Only resize for evaluation, no augmentation
            self.tfm_gens.append(T.ResizeShortestEdge(
                short_edge_length=cfg.INPUT.MIN_SIZE_TEST,
                max_size=cfg.INPUT.MAX_SIZE_TEST,
                sample_style="choice"
            ))
    
    def __call__(self, dataset_dict):
        """
        Convert dataset_dict with 'annotations' to format with 'instances'
        with proper transform handling
        """
        dataset_dict = copy.deepcopy(dataset_dict)
        
        # Read image
        image = utils.read_image(dataset_dict["file_name"], format=self.image_format)
        utils.check_image_size(dataset_dict, image)
        
        # Apply transforms to image first
        image, transforms = T.apply_transform_gens(self.tfm_gens, image)
        
        # Convert annotations to instances AFTER applying transforms
        if "annotations" in dataset_dict:
            annotations = dataset_dict["annotations"]
            
            # Filter out annotations with iscrowd=1 if any
            annotations = [ann for ann in annotations if ann.get("iscrowd", 0) == 0]
            
            if len(annotations) > 0:
                # Create instances object from TRANSFORMED image dimensions
                instances = utils.annotations_to_instances(
                    annotations, image.shape[0], image.shape[1]
                )
                
                # No need to transform instances since we created them after image transform
                # Filter out empty instances
                instances = utils.filter_empty_instances(instances)
                dataset_dict["instances"] = instances
            else:
                # Create empty instances if no annotations
                instances = Instances((image.shape[0], image.shape[1]))
                instances.gt_boxes = Boxes(torch.zeros((0, 4), dtype=torch.float32))
                instances.gt_classes = torch.zeros((0,), dtype=torch.int64)
                dataset_dict["instances"] = instances
        else:
            # Create empty instances if no annotations key
            instances = Instances((image.shape[0], image.shape[1]))
            instances.gt_boxes = Boxes(torch.zeros((0, 4), dtype=torch.float32))
            instances.gt_classes = torch.zeros((0,), dtype=torch.int64)
            dataset_dict["instances"] = instances
        
        # Set the transformed image
        dataset_dict["image"] = torch.as_tensor(image.transpose(2, 0, 1).astype("float32"))
        
        # Remove annotations as we now have instances
        dataset_dict.pop("annotations", None)
        
        return dataset_dict

class FairConfusionMatrixCollector:
    """
    Fair confusion matrix collector that includes background class
    to make DiffusionDet comparable to YOLO evaluation.
    """
    def __init__(self, num_classes, conf_threshold=0.05, iou_threshold=0.5):
        self.num_classes = num_classes
        self.background_class = num_classes  # Background is class N
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        
        # Store all GT-Pred pairs including background
        self.all_targets = []
        self.all_predictions = []
        
        self.debug_counts = {
            'total_batches': 0,
            'total_gt_boxes': 0,
            'total_pred_boxes': 0,
            'valid_pred_boxes': 0,
            'matched_pairs': 0,
            'false_positives': 0,
            'false_negatives': 0
        }
        
    def compute_iou(self, box1, box2):
        """Compute IoU between two boxes [x1, y1, x2, y2]"""
        # Get intersection coordinates
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        # Compute intersection area
        if x2 <= x1 or y2 <= y1:
            return 0.0
            
        intersection = (x2 - x1) * (y2 - y1)
        
        # Compute union area
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def process_batch(self, inputs, outputs, debug=False):
        """
        Process a batch with fair GT-Pred matching including background handling.
        This makes DiffusionDet evaluation comparable to YOLO.
        """
        self.debug_counts['total_batches'] += 1
        
        for input_data, output in zip(inputs, outputs):
            # Extract ground truth
            gt_boxes = None
            gt_classes = None
            
            if "instances" in input_data:
                gt_instances = input_data["instances"]
                if hasattr(gt_instances, "gt_boxes") and hasattr(gt_instances, "gt_classes"):
                    gt_boxes = gt_instances.gt_boxes.tensor.cpu().numpy()
                    gt_classes = gt_instances.gt_classes.cpu().numpy()
                    self.debug_counts['total_gt_boxes'] += len(gt_boxes)
            
            # Extract predictions
            pred_boxes = None
            pred_classes = None
            pred_scores = None
            
            if "instances" in output:
                pred_instances = output["instances"]
                if len(pred_instances) > 0:
                    self.debug_counts['total_pred_boxes'] += len(pred_instances)
                    
                    if hasattr(pred_instances, "pred_boxes") and hasattr(pred_instances, "pred_classes") and hasattr(pred_instances, "scores"):
                        # Filter by confidence threshold
                        valid_mask = pred_instances.scores >= self.conf_threshold
                        
                        if valid_mask.sum() > 0:
                            pred_boxes = pred_instances.pred_boxes.tensor[valid_mask].cpu().numpy()
                            pred_classes = pred_instances.pred_classes[valid_mask].cpu().numpy()
                            pred_scores = pred_instances.scores[valid_mask].cpu().numpy()
                            self.debug_counts['valid_pred_boxes'] += len(pred_boxes)
            
            # Process this image
            self._process_image_detections(gt_boxes, gt_classes, pred_boxes, pred_classes, debug)
    
    def _process_image_detections(self, gt_boxes, gt_classes, pred_boxes, pred_classes, debug=False):
        """
        Process detections for a single image with fair GT-Pred matching.
        
        This implements the same logic as YOLO confusion matrix:
        - Matched GT → Predicted class
        - Unmatched GT → Background (missed detection)
        - Unmatched Pred → Background → Predicted class (false positive)
        """
        
        # Track matched predictions to avoid double counting
        matched_pred_indices = set()
        
        # Process ground truth objects
        if gt_boxes is not None and gt_classes is not None and len(gt_boxes) > 0:
            for gt_box, gt_class in zip(gt_boxes, gt_classes):
                best_iou = 0
                best_pred_idx = -1
                best_pred_class = self.background_class  # Default to background (missed)
                
                # Find best matching prediction
                if pred_boxes is not None and pred_classes is not None and len(pred_boxes) > 0:
                    for pred_idx, (pred_box, pred_class) in enumerate(zip(pred_boxes, pred_classes)):
                        if pred_idx in matched_pred_indices:
                            continue  # Already matched
                            
                        iou = self.compute_iou(gt_box, pred_box)
                        
                        if iou >= self.iou_threshold and iou > best_iou:
                            best_iou = iou
                            best_pred_idx = pred_idx
                            best_pred_class = pred_class
                
                # Record GT → Prediction mapping
                self.all_targets.append(gt_class)
                self.all_predictions.append(best_pred_class)
                
                if best_pred_idx != -1:
                    matched_pred_indices.add(best_pred_idx)
                    self.debug_counts['matched_pairs'] += 1
                else:
                    self.debug_counts['false_negatives'] += 1
                    
                if debug and self.debug_counts['total_batches'] <= 2:
                    status = "MATCHED" if best_pred_class != self.background_class else "MISSED"
                    print(f"  GT {gt_class} → Pred {best_pred_class} ({status})")
        
        # Process unmatched predictions (false positives)
        if pred_boxes is not None and pred_classes is not None and len(pred_boxes) > 0:
            for pred_idx, pred_class in enumerate(pred_classes):
                if pred_idx not in matched_pred_indices:
                    # False positive: Background → Predicted class
                    self.all_targets.append(self.background_class)
                    self.all_predictions.append(pred_class)
                    self.debug_counts['false_positives'] += 1
                    
                    if debug and self.debug_counts['total_batches'] <= 2:
                        print(f"  Background → Pred {pred_class} (FALSE POS)")
    
    def print_debug_stats(self):
        """Print comprehensive debugging statistics"""
        print(f"\n📊 FAIR CONFUSION MATRIX STATISTICS:")
        print(f"Total batches processed: {self.debug_counts['total_batches']}")
        print(f"Total GT boxes: {self.debug_counts['total_gt_boxes']}")
        print(f"Total pred boxes: {self.debug_counts['total_pred_boxes']}")
        print(f"Valid pred boxes (conf > {self.conf_threshold}): {self.debug_counts['valid_pred_boxes']}")
        print(f"Matched GT-Pred pairs: {self.debug_counts['matched_pairs']}")
        print(f"False negatives (missed): {self.debug_counts['false_negatives']}")
        print(f"False positives: {self.debug_counts['false_positives']}")
        print(f"Total matrix entries: {len(self.all_predictions)}")
        
        # Class distribution
        target_counts = {}
        pred_counts = {}
        
        for target in self.all_targets:
            target_counts[target] = target_counts.get(target, 0) + 1
        for pred in self.all_predictions:
            pred_counts[pred] = pred_counts.get(pred, 0) + 1
            
        print(f"\nTarget class distribution: {target_counts}")
        print(f"Prediction class distribution: {pred_counts}")
    
    def generate_confusion_matrix(self, output_dir, class_names):
        """Generate fair confusion matrix including background class"""
        print(f"Generating fair confusion matrix with background class...")
        
        if len(self.all_predictions) == 0 or len(self.all_targets) == 0:
            print("⚠️ No data collected for confusion matrix")
            return
        
        # Include background in class names
        extended_class_names = class_names + ["Background"]
        
        # Generate confusion matrix (including background)
        all_labels = list(range(self.num_classes + 1))  # 0-6 + background (7)
        
        cm = confusion_matrix(
            self.all_targets, 
            self.all_predictions, 
            labels=all_labels
        )
        
        # Normalized confusion matrix
        cm_normalized = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-8)
        
        # Save matrices
        np.save(os.path.join(output_dir, "confusion_matrix_fair.npy"), cm)
        np.save(os.path.join(output_dir, "confusion_matrix_fair_normalized.npy"), cm_normalized)
        
        # Plot confusion matrices
        # 1. Raw counts (with background)
        plt.figure(figsize=(12, 10))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=extended_class_names)
        disp.plot(cmap='Blues', values_format='d')
        plt.title("Confusion Matrix-DIffdetCL", fontsize=11, fontweight='bold')
        disp.ax_.tick_params(axis='x', labelsize=8)  
        disp.ax_.tick_params(axis='y', labelsize=8)  
        for text in disp.text_.ravel():
            if text.get_text():  
                text.set_fontsize(8)  

        plt.xticks(rotation=45, ha='right', fontsize=8)  
        plt.yticks(rotation=0, fontsize=8)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "confusion_matrix_diffdetCL.png"), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Normalized (with background)
        plt.figure(figsize=(12, 10))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm_normalized, display_labels=extended_class_names)
        disp.plot(cmap='Blues', values_format='.3f')
        plt.title("Normalized Confusion Matrix DIffdetCL", fontsize=11, fontweight='bold')
        disp.ax_.tick_params(axis='x', labelsize=8)  
        disp.ax_.tick_params(axis='y', labelsize=8)  
        for text in disp.text_.ravel():
            if text.get_text():  
                text.set_fontsize(8)  

        plt.xticks(rotation=45, ha='right', fontsize=8)  
        plt.yticks(rotation=0, fontsize=8)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "confusion_matrix_diffdetCL_normalized.png"), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. ALSO create object-only matrix (like original DiffusionDet)
        object_pairs = [(t, p) for t, p in zip(self.all_targets, self.all_predictions) 
                       if t < self.background_class and p < self.background_class]
        
        if len(object_pairs) > 0:
            obj_targets, obj_predictions = zip(*object_pairs)
            
            cm_obj = confusion_matrix(obj_targets, obj_predictions, labels=list(range(self.num_classes)))
            cm_obj_norm = cm_obj.astype('float') / (cm_obj.sum(axis=1)[:, np.newaxis] + 1e-8)
            
            plt.figure(figsize=(10, 8))
            disp = ConfusionMatrixDisplay(confusion_matrix=cm_obj_norm, display_labels=class_names)
            disp.plot(cmap='Blues', values_format='.3f')
            plt.title("Object-Only Confusion Matrix (No Background)\nOriginal DiffusionDet Style", fontsize=11, fontweight='bold')
            disp.ax_.tick_params(axis='x', labelsize=8)  
            disp.ax_.tick_params(axis='y', labelsize=8)  
            for text in disp.text_.ravel():
                if text.get_text():  
                    text.set_fontsize(8)  

            plt.xticks(rotation=45, ha='right', fontsize=8)  
            plt.yticks(rotation=0, fontsize=8)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "confusion_matrix_objects_only.png"), dpi=300, bbox_inches='tight')
            plt.close()
        
        print(f"✅ Generated fair confusion matrix with {len(self.all_predictions)} total samples")
        print(f"   - Matched detections: {self.debug_counts['matched_pairs']}")
        print(f"   - Missed detections: {self.debug_counts['false_negatives']}")
        print(f"   - False positives: {self.debug_counts['false_positives']}")
        
        # Print comparison statistics
        total_gt = self.debug_counts['total_gt_boxes']
        detection_rate = self.debug_counts['matched_pairs'] / total_gt if total_gt > 0 else 0
        false_pos_rate = self.debug_counts['false_positives'] / self.debug_counts['valid_pred_boxes'] if self.debug_counts['valid_pred_boxes'] > 0 else 0
        
        print(f"\nDetection Performance:")
        print(f"Detection rate: {detection_rate:.3f} ({self.debug_counts['matched_pairs']}/{total_gt})")
        print(f"False positive rate: {false_pos_rate:.3f} ({self.debug_counts['false_positives']}/{self.debug_counts['valid_pred_boxes']})")
        
        # Overall accuracy (including background)
        overall_accuracy = np.trace(cm) / np.sum(cm)
        print(f"Overall accuracy (with background): {overall_accuracy:.3f}")
        
        # Object-only accuracy
        if len(object_pairs) > 0:
            obj_accuracy = np.trace(cm_obj) / np.sum(cm_obj)
            print(f"Object-only accuracy (no background): {obj_accuracy:.3f}")

# Configuration and evaluation (UNCHANGED - same as your original)
output_dir = "output/kitti_res50_90kIter"
os.makedirs(output_dir, exist_ok=True)

from diffusiondet.config import add_diffusiondet_config
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

# Create FAIR confusion matrix collector
cm_collector = FairConfusionMatrixCollector(
    num_classes=len(classes),
    conf_threshold=0.05,  # Same as model threshold
    iou_threshold=0.5     # Standard IoU threshold
)

print(f"✅ Starting FAIR confusion matrix generation (comparable to YOLO)...")
print(f"Using confidence threshold: {cm_collector.conf_threshold}")
print(f"Using IoU threshold: {cm_collector.iou_threshold}")
print(f"Including background class for fair comparison")

# Data loader setup (UNCHANGED)
cfg_val = cfg.clone()
cfg_val.DATASETS.TRAIN = ("kitti_val_split",)
cfg_val.DATALOADER.NUM_WORKERS = 0

eval_mapper = EvaluationDatasetMapper(cfg_val, is_train=False)
val_loader = build_detection_train_loader(cfg_val, mapper=eval_mapper)

print("✅ Created validation data loader with fixed transform handling")

# Process validation data
batch_count = 0
max_batches = 1500

print(f"📊 Starting evaluation loop...")

with torch.no_grad():
    for idx, inputs in enumerate(val_loader):
        outputs = model(inputs)
        
        # Debug first few batches
        debug_this_batch = idx < 2
        cm_collector.process_batch(inputs, outputs, debug=debug_this_batch)
        
        batch_count += 1
        
        if (batch_count) % 100 == 0:
            print(f"Processed {batch_count} batches...")
            
        if batch_count >= max_batches:
            print(f"Reached maximum batches ({max_batches}), stopping...")
            break

print(f"✅ Evaluation completed after {batch_count} batches!")

# Print debug statistics
cm_collector.print_debug_stats()

# Generate FAIR confusion matrix
cm_collector.generate_confusion_matrix(output_dir, classes)

print(f"\n✅ Fair confusion matrix saved to: {output_dir}")
print(f"Files generated:")
print(f"  - confusion_matrix_fair.png (with background - comparable to YOLO)")
print(f"  - confusion_matrix_fair_normalized.png (normalized with background)")
print(f"  - confusion_matrix_objects_only.png (original DiffusionDet style)")
print(f"\nNow you can fairly compare DiffusionDet with YOLO results!")