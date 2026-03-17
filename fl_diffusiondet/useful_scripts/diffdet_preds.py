
"""
Fixed FL Predictions Script
Uses your exact DiffusionDet configuration and setup.
"""

import os
import sys
import json
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import argparse
import toml
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from collections import OrderedDict

# Try to import cv2, provide fallback if not available
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    print("Warning: OpenCV not available. Using PIL as fallback for image loading.")

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

# Import your project modules
from register_kitti_fldata import register_kitti_splits

# Detectron2 imports
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.modeling import build_model
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog

# Try to import DiffusionDet modules (with fallback)
try:
    from diffusiondet import add_diffusiondet_config
    from diffusiondet.util.model_ema import add_model_ema_configs
    DIFFUSIONDET_AVAILABLE = True
    print("✅ DiffusionDet modules imported successfully")
except ImportError as e:
    print(f"⚠️  DiffusionDet modules not available: {e}")
    print("Will use manual config setup")
    DIFFUSIONDET_AVAILABLE = False

# KITTI class mapping
KITTI_CLASSES = {
    0: "Car",
    1: "Van", 
    2: "Truck",
    3: "Pedestrian",
    4: "Person_sitting",
    5: "Cyclist",
    6: "Tram"
}

# KITTI colors for visualization
KITTI_COLORS = [
    [255, 69, 58],    # Car - Red
    [255, 159, 159],  # Van - Light Red/Pink
    [255, 149, 0],    # Truck - Orange
    [255, 193, 7],    # Pedestrian - Yellow/Orange
    [255, 235, 59],   # Person_sitting - Yellow
    [76, 175, 80],    # Cyclist - Green
    [139, 195, 74]    # Tram - Light Green
]

# Convert to matplotlib format (0-1 range)
KITTI_COLORS_MPL = [[c/255.0 for c in color] for color in KITTI_COLORS]

def setup_config_with_diffusiondet(config_file: str):
    """Setup config exactly like your FL training does."""
    try:
        cfg = get_cfg()
        
        if DIFFUSIONDET_AVAILABLE:
            # Use your exact setup from task.py
            add_diffusiondet_config(cfg)
            add_model_ema_configs(cfg)
            print("✅ Added DiffusionDet and EMA configs")
        else:
            # Manual fallback setup
            print("⚠️  Using manual config setup")
            from detectron2.config import CfgNode as CN
            
            # Add DiffusionDet config manually
            cfg.MODEL.DiffusionDet = CN()
            cfg.MODEL.DiffusionDet.NUM_PROPOSALS = 800
            cfg.MODEL.DiffusionDet.NUM_CLASSES = 7
            cfg.MODEL.DiffusionDet.SAMPLE_STEP = 1
            
            # Add MODEL_EMA config manually
            cfg.MODEL_EMA = CN()
            cfg.MODEL_EMA.ENABLED = True
            cfg.MODEL_EMA.DECAY = 0.9999
            
            # Add ROI_HEADS config
            cfg.MODEL.ROI_HEADS.NAME = "DiffusionDetROIHead"
            cfg.MODEL.ROI_HEADS.IN_FEATURES = ["p2", "p3", "p4", "p5"]
        
        # Load your config file
        if os.path.exists(config_file):
            cfg.merge_from_file(config_file)
            print(f"✅ Loaded config from: {config_file}")
        else:
            raise FileNotFoundError(f"Config file not found: {config_file}")
        
        return cfg
        
    except Exception as e:
        print(f"❌ Error setting up config: {e}")
        raise

class FixedFLPredictor:
    """FL Model Predictor using your exact configuration."""
    
    def __init__(self, config_file: str, model_path: str, confidence_threshold: float = 0.01):
        """
        Initialize FL predictor.
        
        Args:
            config_file: Path to your diffdet_config.yaml
            model_path: Path to the FL model checkpoint
            confidence_threshold: Minimum confidence for predictions
        """
        self.config_file = config_file
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        
        print(f"Loading FL model from: {model_path}")
        print(f"Using config: {config_file}")
        
        # Setup configuration exactly like your FL training
        self.cfg = setup_config_with_diffusiondet(config_file)
        
        # Update confidence threshold
        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = confidence_threshold
        
        # Set device
        self.cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Build model
        self.model = build_model(self.cfg)
        
        # Load FL model weights
        self._load_fl_weights()
        
        # Setup predictor
        self.predictor = DefaultPredictor(self.cfg)
        
        print(f"✅ FL Predictor initialized with confidence threshold: {confidence_threshold}")
        self._print_model_info()
    
    def _print_model_info(self):
        """Print model information for debugging."""
        print(f"\n🔍 Model Information:")
        print(f"   - Model type: {type(self.model).__name__}")
        print(f"   - META_ARCHITECTURE: {self.cfg.MODEL.META_ARCHITECTURE}")
        print(f"   - NUM_CLASSES: {self.cfg.MODEL.DiffusionDet.NUM_CLASSES}")
        print(f"   - NUM_PROPOSALS: {self.cfg.MODEL.DiffusionDet.NUM_PROPOSALS}")
        print(f"   - SAMPLE_STEP: {self.cfg.MODEL.DiffusionDet.SAMPLE_STEP}")
        print(f"   - BACKBONE: {self.cfg.MODEL.BACKBONE.NAME}")
        print(f"   - Device: {next(self.model.parameters()).device}")
        print(f"   - Model EMA: {self.cfg.MODEL_EMA.ENABLED}")
    
    def _load_fl_weights(self):
        """Load FL model weights."""
        try:
            # Load checkpoint
            checkpoint = torch.load(self.model_path, map_location='cpu')
            
            # Handle different checkpoint formats
            if isinstance(checkpoint, dict):
                if 'model' in checkpoint:
                    state_dict = checkpoint['model']
                elif 'state_dict' in checkpoint:
                    state_dict = checkpoint['state_dict']
                else:
                    # If it's just the state dict itself
                    state_dict = checkpoint
            else:
                state_dict = checkpoint
            
            # Load weights into model
            missing_keys, unexpected_keys = self.model.load_state_dict(state_dict, strict=False)
            
            if missing_keys:
                print(f"⚠️  Missing keys in checkpoint: {len(missing_keys)} keys")
                if len(missing_keys) <= 5:
                    print(f"   Missing keys: {missing_keys}")
                else:
                    print(f"   First 5 missing keys: {missing_keys[:5]}")
            
            if unexpected_keys:
                print(f"⚠️  Unexpected keys in checkpoint: {len(unexpected_keys)} keys")
                if len(unexpected_keys) <= 5:
                    print(f"   Unexpected keys: {unexpected_keys}")
                else:
                    print(f"   First 5 unexpected keys: {unexpected_keys[:5]}")
            
            self.model.eval()
            
            # Move to device
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model.to(device)
            
            print("✅ FL model weights loaded successfully")
            
        except Exception as e:
            print(f"❌ Error loading FL model weights: {e}")
            raise
    
    def predict_single_image(self, image_path: str) -> Dict:
        """
        Make prediction on a single image.
        
        Args:
            image_path: Path to the image
            
        Returns:
            Dictionary with prediction results
        """
        # Load image
        if CV2_AVAILABLE:
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Could not load image: {image_path}")
        else:
            # Fallback to PIL
            pil_image = Image.open(image_path)
            image = np.array(pil_image)
            if len(image.shape) == 3 and image.shape[2] == 3:
                # Convert RGB to BGR for detectron2
                image = image[:, :, ::-1]
        
        # Make prediction
        with torch.no_grad():
            outputs = self.predictor(image)
        
        # Extract predictions
        instances = outputs["instances"]
        
        # Convert to CPU and extract data
        predictions = {
            'image_path': image_path,
            'image_shape': image.shape[:2],  # (height, width)
            'boxes': instances.pred_boxes.tensor.cpu().numpy().tolist() if len(instances) > 0 else [],
            'scores': instances.scores.cpu().numpy().tolist() if len(instances) > 0 else [],
            'classes': instances.pred_classes.cpu().numpy().tolist() if len(instances) > 0 else [],
            'num_detections': len(instances)
        }
        
        # Add class names
        predictions['class_names'] = [KITTI_CLASSES[cls] for cls in predictions['classes']]
        
        return predictions
    
    def visualize_predictions(self, image_path: str, predictions: Dict, save_path: str):
        """
        Visualize predictions on image and save.
        
        Args:
            image_path: Path to original image
            predictions: Prediction results
            save_path: Path to save visualization
        """
        # Load image
        if CV2_AVAILABLE:
            image = cv2.imread(image_path)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            # Fallback to PIL
            pil_image = Image.open(image_path)
            image_rgb = np.array(pil_image)
        
        # Create matplotlib figure
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        ax.imshow(image_rgb)
        
        # Draw bounding boxes
        for i, (box, score, class_id, class_name) in enumerate(zip(
            predictions['boxes'], predictions['scores'], predictions['classes'], predictions['class_names']
        )):
            x1, y1, x2, y2 = box
            
            # Get color for this class
            color = KITTI_COLORS_MPL[class_id] if class_id < len(KITTI_COLORS_MPL) else [1.0, 0.0, 0.0]
            
            # Create rectangle
            rect = patches.Rectangle(
                (x1, y1), x2 - x1, y2 - y1,
                linewidth=2, edgecolor=color, facecolor='none'
            )
            ax.add_patch(rect)
            
            # Add label
            label = f"{class_name}: {score:.3f}"
            ax.text(x1, y1 - 5, label, fontsize=10, color=color,
                   bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
        
        # Remove axes
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"FL Predictions - {Path(image_path).name}", fontsize=14)
        
        # Save visualization
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Visualization saved: {save_path}")

def load_config(config_path: str = "pyproject.toml") -> Dict:
    """Load configuration from TOML file."""
    try:
        with open(config_path, 'r') as f:
            config = toml.load(f)
        return config
    except Exception as e:
        print(f"Error loading config: {e}")
        return {}

def find_fl_model(output_dir: str) -> Optional[str]:
    """Find the FL model file."""
    output_path = Path(output_dir)
    
    # Look for model files in various locations
    possible_paths = [
        output_path / "global" / "final_global_model.pth",
        output_path / "final_global_model.pth",
        output_path / "checkpoints" / "global_model_final.pth"
    ]
    
    # Check predefined paths first
    for path in possible_paths:
        if path.exists():
            return str(path)
    
    # Look for latest round model
    global_models = list(output_path.glob("**/global_model_round_*.pth"))
    global_models.extend(list(output_path.glob("**/global_model_*.pth")))
    
    if global_models:
        # Sort by modification time and get the latest
        latest_model = max(global_models, key=lambda p: p.stat().st_mtime)
        return str(latest_model)
    
    return None

def get_sample_images(samples_dir: str) -> Dict[str, List[str]]:
    """
    Get sample images organized by client.
    
    Args:
        samples_dir: Base directory with client samples
        
    Returns:
        Dictionary mapping client names to image paths
    """
    samples_path = Path(samples_dir)
    client_samples = {}
    
    if not samples_path.exists():
        print(f"Warning: Samples directory not found: {samples_dir}")
        return client_samples
    
    # Get all client directories
    for client_dir in samples_path.iterdir():
        if client_dir.is_dir():
            client_name = client_dir.name
            
            # Find all image files in client directory
            image_files = []
            for ext in ['*.png', '*.jpg', '*.jpeg', '*.PNG', '*.JPG', '*.JPEG']:
                image_files.extend(client_dir.glob(ext))
            
            if image_files:
                client_samples[client_name] = [str(img) for img in sorted(image_files)]
                print(f"Found {len(image_files)} images for client: {client_name}")
    
    return client_samples

def main():
    """Main prediction function."""
    parser = argparse.ArgumentParser(description="Fixed FL Predictions on KITTI samples")
    parser.add_argument("--config", default="pyproject.toml", help="Path to TOML config file")
    parser.add_argument("--diffdet-config", default="configs/diffdet_config.yaml", help="Path to DiffusionDet config")
    parser.add_argument("--output-dir", help="FL output directory (auto-detected if not provided)")
    parser.add_argument("--model-path", help="Path to FL model (auto-detected if not provided)")
    parser.add_argument("--confidence", type=float, default=0.01, help="Confidence threshold")
    parser.add_argument("--max-samples", type=int, default=10, help="Max samples per client")
    parser.add_argument("--no-visualize", action="store_true", help="Skip creating visualizations")
    parser.add_argument("--save-json", action="store_true", help="Also save JSON predictions")
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Get output directory from config or argument
    if args.output_dir:
        output_dir = args.output_dir
    else:
        output_dir = config.get("tool", {}).get("flwr", {}).get("app", {}).get("config", {}).get("output-base-dir", "./output")
    
    # Find model path
    if args.model_path:
        model_path = args.model_path
    else:
        model_path = find_fl_model(output_dir)
        if not model_path:
            print(f"❌ Could not find FL model in {output_dir}")
            print("Please specify --model-path explicitly")
            sys.exit(1)
    
    if not os.path.exists(model_path):
        print(f"❌ Model not found at {model_path}")
        sys.exit(1)
    
    # Check DiffusionDet config
    if not os.path.exists(args.diffdet_config):
        print(f"❌ DiffusionDet config not found: {args.diffdet_config}")
        sys.exit(1)
    
    # Setup paths
    samples_dir = "/mimer/NOBACKUP/groups/naiss2024-5-153/old_projects/Berhane/labelled_kitti/difdet/samples_per_class"
    predictions_dir = f"{output_dir}/predictions"
    
    # Create output directories
    os.makedirs(predictions_dir, exist_ok=True)
    os.makedirs(f"{predictions_dir}/visualizations", exist_ok=True)
    
    print(f"\n🎯 Configuration:")
    print(f"   Model Path: {model_path}")
    print(f"   DiffDet Config: {args.diffdet_config}")
    print(f"   Output Dir: {output_dir}")
    print(f"   Samples Dir: {samples_dir}")
    print(f"   Predictions Dir: {predictions_dir}")
    print(f"   Confidence: {args.confidence}")
    print(f"   Max Samples: {args.max_samples}")
    print(f"   Creating Visualizations: {not args.no_visualize}")
    
    # Register KITTI splits (needed for model initialization)
    register_kitti_splits()
    
    # Initialize predictor
    try:
        predictor = FixedFLPredictor(args.diffdet_config, model_path, args.confidence)
    except Exception as e:
        print(f"❌ Error initializing predictor: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Get sample images
    client_samples = get_sample_images(samples_dir)
    
    if not client_samples:
        print("❌ No sample images found!")
        sys.exit(1)
    
    # Make predictions for each client
    all_predictions = {}
    total_processed = 0
    total_with_detections = 0
    
    for client_name, image_paths in client_samples.items():
        print(f"\n{'='*50}")
        print(f"🔍 Processing client: {client_name}")
        print(f"{'='*50}")
        
        # Limit samples per client
        image_paths = image_paths[:args.max_samples]
        
        client_predictions = []
        client_detections = 0
        
        for i, image_path in enumerate(image_paths, 1):
            image_name = Path(image_path).stem
            print(f"   [{i:2d}/{len(image_paths):2d}] {image_name}...", end=" ")
            
            try:
                # Make prediction
                prediction = predictor.predict_single_image(image_path)
                client_predictions.append(prediction)
                total_processed += 1
                
                # Create visualization
                if not args.no_visualize:
                    vis_filename = f"{client_name}_{image_name}_predictions.png"
                    vis_path = f"{predictions_dir}/visualizations/{vis_filename}"
                    predictor.visualize_predictions(image_path, prediction, vis_path)
                
                # Print detection summary
                if prediction['num_detections'] > 0:
                    detected_classes = prediction['class_names']
                    detected_scores = [f"{s:.3f}" for s in prediction['scores']]
                    
                    print(f"✅ {prediction['num_detections']} objects: ", end="")
                    for cls, score in zip(detected_classes, detected_scores):
                        print(f"{cls}({score}) ", end="")
                    print()
                    
                    client_detections += prediction['num_detections']
                    total_with_detections += 1
                else:
                    print("❌ No objects detected")
                    
            except Exception as e:
                print(f"💥 Error: {e}")
                continue
        
        print(f"\n📊 {client_name} Summary: {client_detections} total detections in {len(client_predictions)} images")
        all_predictions[client_name] = client_predictions
    
    # Save JSON predictions if requested
    if args.save_json:
        predictions_file = f"{predictions_dir}/fl_predictions.json"
        with open(predictions_file, 'w') as f:
            json.dump(all_predictions, f, indent=2)
        print(f"\n💾 JSON predictions saved: {predictions_file}")
    
    # Create and save summary
    summary = {
        'model_path': model_path,
        'config_path': args.diffdet_config,
        'confidence_threshold': args.confidence,
        'total_clients': len(client_samples),
        'total_images_processed': total_processed,
        'total_images_with_detections': total_with_detections,
        'detection_rate': f"{(total_with_detections/total_processed)*100:.1f}%" if total_processed > 0 else "0%",
        'client_summary': {}
    }
    
    for client_name, predictions in all_predictions.items():
        total_detections = sum(p['num_detections'] for p in predictions)
        images_with_detections = sum(1 for p in predictions if p['num_detections'] > 0)
        
        summary['client_summary'][client_name] = {
            'total_images': len(predictions),
            'images_with_detections': images_with_detections,
            'total_detections': total_detections,
            'avg_detections_per_image': total_detections / len(predictions) if predictions else 0,
            'detection_rate': f"{(images_with_detections/len(predictions))*100:.1f}%" if predictions else "0%"
        }
    
    # Save summary
    summary_file = f"{predictions_dir}/prediction_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Print final summary
    print(f"\n{'='*70}")
    print("🎯 FL PREDICTION RESULTS SUMMARY")
    print(f"{'='*70}")
    print(f"Model: {Path(model_path).name}")
    print(f"Config: {Path(args.diffdet_config).name}")
    print(f"Confidence Threshold: {args.confidence}")
    print(f"Total Images Processed: {total_processed}")
    print(f"Images with Detections: {total_with_detections} ({(total_with_detections/total_processed)*100:.1f}%)")
    print(f"\nPer-Client Results:")
    print(f"{'Client':<15} {'Images':<8} {'With Det':<8} {'Det Rate':<10} {'Total Det':<10} {'Avg/Image':<10}")
    print("-" * 70)
    
    for client_name, client_summary in summary['client_summary'].items():
        print(f"{client_name:<15} {client_summary['total_images']:<8} "
              f"{client_summary['images_with_detections']:<8} "
              f"{client_summary['detection_rate']:<10} "
              f"{client_summary['total_detections']:<10} "
              f"{client_summary['avg_detections_per_image']:<10.1f}")
    
    print(f"\n📁 Output Files:")
    print(f"   🖼️  Visualizations: {predictions_dir}/visualizations/")
    print(f"   📊 Summary: {summary_file}")
    if args.save_json:
        print(f"   📋 JSON Data: {predictions_file}")
    
    print(f"\n🎉 Prediction complete! Check visualizations folder for annotated images.")

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# """
# Fixed FL Predictions Script for KITTI samples.
# - Uses correct config from training (via pyproject.toml)
# - Applies confidence threshold before visualizing
# - Supports multiple strategy directories (e.g. output_fedprox, output_fedavg)
# - Handles FL vs CL model loading explicitly
# """

# import os
# import sys
# import json
# import torch
# import argparse
# import toml
# import numpy as np
# from pathlib import Path
# from typing import Dict, List, Optional
# from PIL import Image
# import matplotlib.pyplot as plt
# import matplotlib.patches as patches

# try:
#     import cv2
#     CV2_AVAILABLE = True
# except ImportError:
#     CV2_AVAILABLE = False

# sys.path.append(os.path.dirname(os.path.dirname(__file__)))
# from fl.task import get_model, set_parameters
# from register_kitti_fldata import register_kitti_splits
# from detectron2.engine import DefaultPredictor
# from detectron2.config import get_cfg

# KITTI_CLASSES = {
#     0: "Car", 1: "Van", 2: "Truck", 3: "Pedestrian",
#     4: "Person_sitting", 5: "Cyclist", 6: "Tram"
# }
# KITTI_COLORS = [
#     [255,69,58], [255,159,159], [255,149,0], [255,193,7],
#     [255,235,59], [76,175,80], [139,195,74]
# ]
# KITTI_COLORS_MPL = [[c/255.0 for c in col] for col in KITTI_COLORS]

# class FLPredictor:
#     def __init__(self, model_path: str, config_path: str, confidence_threshold: float, is_cl: bool):
#         print(f"Loading {'CL' if is_cl else 'FL'} model from: {model_path}")
#         self.model_path = model_path
#         self.confidence_threshold = confidence_threshold

#         self.model, self.cfg = self.load_model_and_config(config_path, model_path, is_cl)
#         self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = confidence_threshold
#         self.predictor = DefaultPredictor(self.cfg)

#     def load_model_and_config(self, config_path: str, model_path: str, is_cl: bool):
#         from diffusiondet import add_diffusiondet_config
#         cfg = get_cfg()
#         add_diffusiondet_config(cfg)

#         cfg.set_new_allowed(True)  # ✅ Allow non-standard keys like MODEL_EMA
#         cfg.merge_from_file(config_path)
#         cfg.set_new_allowed(False)  # ✅ Optional: lock it back to prevent errors later

#         cfg.MODEL.WEIGHTS = model_path
#         cfg.freeze()

#         from detectron2.modeling import build_model
#         model = build_model(cfg)
#         model.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
#         model.eval()

#         ckpt = torch.load(model_path, map_location='cpu')
#         state_dict = ckpt['model'] if is_cl else (ckpt if 'model' not in ckpt else ckpt['model'])
#         model.load_state_dict(state_dict)

#         return model, cfg


#     def predict_and_visualize(self, image_path: str, save_path: str):
#         image = cv2.imread(image_path) if CV2_AVAILABLE else np.array(Image.open(image_path))
#         if not CV2_AVAILABLE: image = image[:, :, ::-1]
#         outputs = self.predictor(image)
#         instances = outputs["instances"]
#         boxes = instances.pred_boxes.tensor.cpu().numpy()
#         scores = instances.scores.cpu().numpy()
#         classes = instances.pred_classes.cpu().numpy()

#         filtered = [
#             (box, score, cls)
#             for box, score, cls in zip(boxes, scores, classes)
#             if score >= self.confidence_threshold
#         ]

#         fig, ax = plt.subplots(figsize=(12, 8))
#         ax.imshow(image[:, :, ::-1])
#         for box, score, cls in filtered:
#             x1, y1, x2, y2 = box
#             color = KITTI_COLORS_MPL[cls] if cls < len(KITTI_COLORS_MPL) else [1, 0, 0]
#             rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=2, edgecolor=color, facecolor='none')
#             ax.add_patch(rect)
#             label = f"{KITTI_CLASSES.get(cls, 'Unknown')}: {score:.2f}"
#             ax.text(x1, y1-5, label, fontsize=10, color=color, bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
#         ax.axis('off')
#         plt.tight_layout()
#         plt.savefig(save_path, dpi=150)
#         plt.close()
#         print(f"Saved: {save_path}")

# def load_config(toml_path: str) -> Dict:
#     with open(toml_path, 'r') as f:
#         return toml.load(f)

# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--config", default="pyproject.toml")
#     parser.add_argument("--confidence", type=float, default=0.25)
#     parser.add_argument("--samples-dir", required=True)
#     parser.add_argument("--max-samples", type=int, default=10)
#     parser.add_argument("--cl", action="store_true", help="Load centralized model format")
#     args = parser.parse_args()

#     cfg = load_config(args.config)
#     base_dir = cfg["tool"]["flwr"]["app"]["config"]["output-base-dir"]
#     model_path = os.path.join(base_dir, "global", "final_global_model.pth")
#     config_path = os.path.join(base_dir, "fl_diffdet", "config.yaml")
#     out_dir = os.path.join(base_dir, "predictions", "visualizations")
#     os.makedirs(out_dir, exist_ok=True)

#     register_kitti_splits()
#     predictor = FLPredictor(model_path, config_path, args.confidence, is_cl=args.cl)

#     for client in os.listdir(args.samples_dir):
#         sample_dir = os.path.join(args.samples_dir, client)
#         if not os.path.isdir(sample_dir): continue
#         for i, img_file in enumerate(sorted(os.listdir(sample_dir))[:args.max_samples]):
#             if not img_file.lower().endswith(('.png', '.jpg', '.jpeg')): continue
#             img_path = os.path.join(sample_dir, img_file)
#             out_path = os.path.join(out_dir, f"{client}_{Path(img_file).stem}_pred.png")
#             predictor.predict_and_visualize(img_path, out_path)

# if __name__ == "__main__":
#     main()
