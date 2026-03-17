"""
simple script to make diffusiondet inferences

"""

import os
import torch
import cv2
from glob import glob
from detectron2.config import get_cfg
from detectron2.engine.defaults import DefaultPredictor
from diffusiondet import add_diffusiondet_config

# --- Config ---
CONFIG_PATH = "output_temp_iid/diffdet_config.yaml"
WEIGHTS_PATH = "final_outputs/diffdet_kitti_FedAvg_r100/FedAvgBaseline_final_round_100.pth" #"./output_temp_iid/FedAvg_round_100.pth"
BASE_IMAGE_DIR = "/mimer/NOBACKUP/groups/naiss2024-5-153/old_projects/Berhane/labelled_kitti/difdet/samples_per_class/"
SAVE_DIR = "./inferences_100e1"
CONFIDENCE_THRESHOLD = 0.3
SUFFIX = "_100_new"

# BGR colors for OpenCV
class_colors = {
    0: (0, 0, 255),             # Car - Red
    1: (255, 0, 0),             # Van - Blue
    2: (255, 0, 255),           # Truck - Magenta
    3: (0, 165, 255),           # Pedestrian - Orange
    4: (255, 0, 127),           # Person_sitting - Pink
    5: (0, 255, 0),             # Cyclist - Green
    6: (255, 255, 0)            # Tram - Cyan
}

class_names = ["Car", "Van", "Truck", "Pedestrian", "Person_sitting", "Cyclist", "Tram"]

# --- Setup ---
os.makedirs(SAVE_DIR, exist_ok=True)
cfg = get_cfg()
add_diffusiondet_config(cfg)
cfg.merge_from_file(CONFIG_PATH)
cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
cfg.freeze()

predictor = DefaultPredictor(cfg)

# Load FL checkpoint
checkpoint = torch.load(WEIGHTS_PATH, map_location="cpu")
state_dict = checkpoint.get("model_state_dict", checkpoint)
print(f"[✓] Loading FL model from round {checkpoint.get('round', 'unknown')}")
predictor.model.load_state_dict(state_dict)
predictor.model.eval()
print(f"[✓] Loaded FL model from {WEIGHTS_PATH}")

# --- Inference Loop ---
subfolders = sorted([d for d in os.listdir(BASE_IMAGE_DIR) if os.path.isdir(os.path.join(BASE_IMAGE_DIR, d))])
print(f"[INFO] Found classes: {subfolders}")

for cls in subfolders:
    class_input_dir = os.path.join(BASE_IMAGE_DIR, cls)
    image_paths = sorted(glob(os.path.join(class_input_dir, "*.png")))
    
    print(f"[{cls.upper()}] Processing {len(image_paths)} images...")

    for i, img_path in enumerate(image_paths):
        img = cv2.imread(img_path)
        outputs = predictor(img)
        instances = outputs["instances"].to("cpu")

        # Filter by confidence
        keep = instances.scores > CONFIDENCE_THRESHOLD
        instances = instances[keep]

        # Draw boxes manually
        for j in range(len(instances)):
            box = instances.pred_boxes[j].tensor.numpy()[0]
            class_id = int(instances.pred_classes[j])
            score = float(instances.scores[j])
            
            x1, y1, x2, y2 = map(int, box)
            color = class_colors[class_id]
            class_name = class_names[class_id]
            
            # Draw box
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            
            # Draw label background
            label = f"{class_name} {score:.0%}"
            (label_w, label_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(img, (x1, y1 - label_h - baseline - 5), (x1 + label_w, y1), color, -1)
            
            # Draw label text
            cv2.putText(img, label, (x1, y1 - baseline - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Save
        name, ext = os.path.splitext(os.path.basename(img_path))
        new_filename = f"{name}{SUFFIX}{ext}"
        out_path = os.path.join(SAVE_DIR, new_filename)
        cv2.imwrite(out_path, img)

        print(f"[{cls}] {i+1}/{len(image_paths)} → {new_filename} ({len(instances)} detections)")

print(f"[✓] All classes processed. Results saved to: {SAVE_DIR}")