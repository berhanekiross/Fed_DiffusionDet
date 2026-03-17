"""Enhanced evaluation module with confusion matrix and AP calculation."""

import torch
import numpy as np
import json
import os
from collections import defaultdict


# bacground aware detection for fl.evaluation

class FastDetectionMetrics:
    """Enhanced evaluator with background-aware confusion matrix."""

    def __init__(self, num_classes=7, iou_thresholds=[0.5, 0.75], ap_iou_range=(0.5, 0.95, 0.05), confidence_threshold=0.3):
        self.num_classes = num_classes
        self.num_classes_with_bg = num_classes + 1  # Add background class
        self.background_class_id = num_classes  # Background is class 7
        self.iou_thresholds = iou_thresholds
        self.confidence_threshold = confidence_threshold  # Add this line
        self.class_names = ["Car", "Van", "Truck", "Pedestrian", "Person_sitting", "Cyclist", "Tram", "Background"]
        
        # AP calculation IoU range
        self.ap_iou_thresholds = np.arange(ap_iou_range[0], ap_iou_range[1] + ap_iou_range[2], ap_iou_range[2])



    def _compute_confusion_matrix(self, predictions, ground_truth):
        """Compute background-aware confusion matrix."""
        confusion_matrix = np.zeros((self.num_classes_with_bg, self.num_classes_with_bg), dtype=int)
        
        for pred, gt in zip(predictions, ground_truth):
            pred_boxes = pred.get('boxes', [])
            pred_classes = pred.get('classes', [])
            pred_scores = pred.get('scores', [])
            gt_boxes = gt.get('boxes', [])
            gt_classes = gt.get('classes', [])
            
            # Handle empty predictions/ground truth
            if not pred_boxes and not gt_boxes:
                # True negative (background correctly predicted as background)
                confusion_matrix[self.background_class_id, self.background_class_id] += 1
                continue
            elif not pred_boxes and gt_boxes:
                # All ground truth missed (confused with background)
                for gt_class in gt_classes:
                    confusion_matrix[gt_class, self.background_class_id] += 1
                continue
            elif pred_boxes and not gt_boxes:
                # All predictions are false positives (background confused with objects)
                for pred_class in pred_classes:
                    confusion_matrix[self.background_class_id, pred_class] += 1
                continue
            
            # Match predictions to ground truth
            matched_gt = set()
            matched_pred = set()
            
            # Sort predictions by confidence
            if pred_scores:
                sorted_indices = np.argsort(pred_scores)[::-1]
                pred_boxes = [pred_boxes[i] for i in sorted_indices]
                pred_classes = [pred_classes[i] for i in sorted_indices]
            
            # Find matches
            for pred_idx, (pred_box, pred_class) in enumerate(zip(pred_boxes, pred_classes)):
                best_iou = 0
                best_gt_idx = -1
                
                for gt_idx, (gt_box, gt_class) in enumerate(zip(gt_boxes, gt_classes)):
                    if gt_idx in matched_gt:
                        continue
                    
                    iou = self.compute_iou(pred_box, gt_box)
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = gt_idx
                
                if best_iou >= self.iou_thresholds[0] and best_gt_idx != -1:
                    # True positive - correct class prediction
                    actual_class = gt_classes[best_gt_idx]
                    confusion_matrix[actual_class, pred_class] += 1
                    matched_gt.add(best_gt_idx)
                    matched_pred.add(pred_idx)
                else:
                    # False positive - background confused with object
                    confusion_matrix[self.background_class_id, pred_class] += 1
                    matched_pred.add(pred_idx)
            
            # Unmatched ground truth - object confused with background
            for gt_idx, gt_class in enumerate(gt_classes):
                if gt_idx not in matched_gt:
                    confusion_matrix[gt_class, self.background_class_id] += 1
        
        return confusion_matrix
    
    def _compute_ap_metrics(self, predictions, ground_truth):
        """Compute Average Precision metrics across IoU thresholds."""
        ap_results = {}
        
        # Collect all detections with confidence scores
        all_detections = []
        all_ground_truths = []
        
        for img_idx, (pred, gt) in enumerate(zip(predictions, ground_truth)):
            pred_boxes = pred.get('boxes', [])
            pred_classes = pred.get('classes', [])
            pred_scores = pred.get('scores', [])
            
            for box, cls, score in zip(pred_boxes, pred_classes, pred_scores):
                all_detections.append({
                    'image_id': img_idx,
                    'class': cls,
                    'confidence': score,
                    'bbox': box
                })
            
            gt_boxes = gt.get('boxes', [])
            gt_classes = gt.get('classes', [])
            
            for box, cls in zip(gt_boxes, gt_classes):
                all_ground_truths.append({
                    'image_id': img_idx,
                    'class': cls,
                    'bbox': box,
                    'matched': False
                })
        
        # Compute AP for each class and IoU threshold
        for class_id in range(self.num_classes):
            class_detections = [d for d in all_detections if d['class'] == class_id]
            class_ground_truths = [gt for gt in all_ground_truths if gt['class'] == class_id]
            
            if not class_ground_truths:
                ap_results[f"ap_class_{class_id}"] = 0.0
                continue
            
            # Sort detections by confidence
            class_detections.sort(key=lambda x: x['confidence'], reverse=True)
            
            # Compute AP at different IoU thresholds
            aps_for_class = []
            for iou_thresh in self.ap_iou_thresholds:
                ap = self._compute_single_ap(class_detections, class_ground_truths, iou_thresh)
                aps_for_class.append(ap)
            
            ap_results[f"ap_class_{class_id}"] = np.mean(aps_for_class)
            ap_results[f"ap50_class_{class_id}"] = aps_for_class[0] if aps_for_class else 0.0
        
        # Overall metrics
        class_aps = [ap_results.get(f"ap_class_{i}", 0.0) for i in range(self.num_classes)]
        ap_results["map"] = np.mean(class_aps)  # Mean AP across all classes
        ap_results["map50"] = np.mean([ap_results.get(f"ap50_class_{i}", 0.0) for i in range(self.num_classes)])
        
        return ap_results

    def _compute_basic_metrics(self, predictions, ground_truth, iou_threshold):
        """Compute basic P, R, F1 metrics (excluding background from averages)."""
        tp = np.zeros(self.num_classes)  # Only for object classes
        fp = np.zeros(self.num_classes)
        fn = np.zeros(self.num_classes)
        
        for pred, gt in zip(predictions, ground_truth):
            matches = self._match_predictions_single_iou(pred, gt, iou_threshold)
            
            for class_id in range(self.num_classes):  # Only object classes
                tp[class_id] += matches[class_id]['tp']
                fp[class_id] += matches[class_id]['fp']
                fn[class_id] += matches[class_id]['fn']
        
        # precision = tp / (tp + fp + 1e-8)
        # recall = tp / (tp + fn + 1e-8)
        # COCO-style class validation
        valid_classes = [i for i in range(self.num_classes) if (tp[i] + fn[i]) > 0]
    
        precision = np.array([tp[i] / (tp[i] + fp[i] + 1e-8) for i in valid_classes])
        recall = np.array([tp[i] / (tp[i] + fn[i] + 1e-8) for i in valid_classes])    
        f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
        
        return {
            "precision": precision.tolist(),
            "recall": recall.tolist(),
            "f1": f1.tolist(),
            "precision_mean": float(np.mean(precision)),
            "recall_mean": float(np.mean(recall)),
            "f1_mean": float(np.mean(f1)),
            "tp": tp.tolist(),
            "fp": fp.tolist(),
            "fn": fn.tolist()
        }

    def evaluate(self, predictions, ground_truth):
        """Compute comprehensive detection metrics with background awareness and confidence filtering."""
        
        # Apply confidence filtering to predictions
        filtered_predictions = filter_by_confidence(predictions, self.confidence_threshold)
        
        # Basic metrics at primary IoU threshold (object classes only)
        basic_metrics = self._compute_basic_metrics(filtered_predictions, ground_truth, self.iou_thresholds[0])
        confusion_matrix = self._compute_confusion_matrix(filtered_predictions, ground_truth)
        
        # AP metrics across IoU thresholds (object classes only) - use original predictions
        ap_metrics = self._compute_ap_metrics(predictions, ground_truth)  # AP uses all confidences
        
        # Background statistics
        bg_stats = self._compute_background_stats(confusion_matrix)
        
        # Combine all metrics
        return {
            **basic_metrics,
            "confusion_matrix": confusion_matrix.tolist(),
            "background_stats": bg_stats,
            "ap_metrics": ap_metrics,
            "class_names": self.class_names,
            "iou_thresholds": self.iou_thresholds,
            "ap_iou_thresholds": self.ap_iou_thresholds.tolist(),
            "confidence_threshold": self.confidence_threshold
        }

    def _compute_background_stats(self, confusion_matrix):
        """Compute background-specific statistics."""
        bg_id = self.background_class_id
        
        # True negatives (background correctly predicted as background)
        true_negatives = confusion_matrix[bg_id, bg_id]
        
        # False negatives (objects missed, predicted as background)
        objects_missed = sum(confusion_matrix[i, bg_id] for i in range(self.num_classes))
        
        # False positives (background predicted as objects)
        background_false_positives = sum(confusion_matrix[bg_id, i] for i in range(self.num_classes))
        
        return {
            "true_negatives": int(true_negatives),
            "objects_missed": int(objects_missed),
            "background_false_positives": int(background_false_positives),
            "background_precision": float(true_negatives / (true_negatives + background_false_positives + 1e-8)),
            "background_recall": float(true_negatives / (true_negatives + objects_missed + 1e-8))
        }
    
    # Keep all other methods unchanged (compute_iou, _compute_ap_metrics, etc.)
    def compute_iou(self, box1, box2):
        """Compute IoU between two boxes."""
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
    
    def _compute_single_ap(self, detections, ground_truths, iou_threshold):
        """Compute AP for single class at single IoU threshold."""
        if not ground_truths:
            return 0.0
        
        # Reset matched status
        for gt in ground_truths:
            gt['matched'] = False
        
        tp = []
        fp = []
        
        for detection in detections:
            # Find best matching ground truth
            best_iou = 0
            best_gt_idx = -1
            
            for gt_idx, gt in enumerate(ground_truths):
                if gt['image_id'] != detection['image_id'] or gt['matched']:
                    continue
                
                iou = self.compute_iou(detection['bbox'], gt['bbox'])
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gt_idx
            
            if best_iou >= iou_threshold and best_gt_idx != -1:
                tp.append(1)
                fp.append(0)
                ground_truths[best_gt_idx]['matched'] = True
            else:
                tp.append(0)
                fp.append(1)
        
        if not tp:
            return 0.0
        
        # Compute precision and recall
        tp_cumsum = np.cumsum(tp)
        fp_cumsum = np.cumsum(fp)
        recalls = tp_cumsum / len(ground_truths)
        precisions = tp_cumsum / (tp_cumsum + fp_cumsum)
        
        # Compute AP using 11-point interpolation
        ap = 0.0
        for t in np.arange(0, 1.1, 0.1):
            if np.sum(recalls >= t) == 0:
                p = 0
            else:
                p = np.max(precisions[recalls >= t])
            ap += p / 11
        
        return ap
    def _match_predictions_single_iou(self, predictions, ground_truth, iou_threshold):
        """Match predictions to ground truth at single IoU threshold."""
                
        matches = defaultdict(lambda: {'tp': 0, 'fp': 0, 'fn': 0})
        
        pred_boxes = predictions.get('boxes', [])
        pred_classes = predictions.get('classes', [])
        gt_boxes = ground_truth.get('boxes', [])
        gt_classes = ground_truth.get('classes', [])
        
        # Count ground truth per class
        for gt_class in gt_classes:
            matches[gt_class]['fn'] += 1
        
        # Match predictions
        matched_gt = set()
        for pred_box, pred_class in zip(pred_boxes, pred_classes):
            best_iou = 0
            best_gt_idx = -1
            
            for gt_idx, (gt_box, gt_class) in enumerate(zip(gt_boxes, gt_classes)):
                if gt_idx in matched_gt or pred_class != gt_class:
                    continue
                
                iou = self.compute_iou(pred_box, gt_box)
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gt_idx
            
            if best_iou >= iou_threshold and best_gt_idx != -1:
                matches[pred_class]['tp'] += 1
                matches[gt_classes[best_gt_idx]]['fn'] -= 1
                matched_gt.add(best_gt_idx)
            else:
                matches[pred_class]['fp'] += 1
        
        return matches

def filter_by_confidence(predictions_list, confidence_threshold):
    """Filter predictions by confidence score threshold."""
    filtered_predictions = []
    
    for pred_dict in predictions_list:
        boxes = pred_dict.get('boxes', [])
        classes = pred_dict.get('classes', [])
        scores = pred_dict.get('scores', [])
        
        # Filter based on confidence threshold
        filtered_boxes = []
        filtered_classes = []
        filtered_scores = []
        
        for box, cls, score in zip(boxes, classes, scores):
            if score >= confidence_threshold:
                filtered_boxes.append(box)
                filtered_classes.append(cls)
                filtered_scores.append(score)
        
        filtered_predictions.append({
            'boxes': filtered_boxes,
            'classes': filtered_classes,
            'scores': filtered_scores
        })
    
    return filtered_predictions


def extract_predictions(detectron_output):
    """Extract predictions from detectron2 output with confidence filtering."""
    instances = detectron_output["instances"]
    
    return {
        'boxes': instances.pred_boxes.tensor.cpu().numpy().tolist(),
        'classes': instances.pred_classes.cpu().numpy().tolist(),
        'scores': instances.scores.cpu().numpy().tolist()
    }

def extract_ground_truth(batch_item):
    """Extract ground truth from batch item."""
    instances = batch_item["instances"]
    
    return {
        'boxes': instances.gt_boxes.tensor.cpu().numpy().tolist(),
        'classes': instances.gt_classes.cpu().numpy().tolist()
    }