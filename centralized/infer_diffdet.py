# infer_diffdet.py

"""
This script performs inference using the DiffusionDet model on sample images.
manupulate the SETUP section below to continue.
""""


import os
import cv2
import torch
import numpy as np
import yaml
from tqdm import tqdm
from detectron2.config import get_cfg
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.modeling import build_model
from detectron2.data import MetadataCatalog
from diffusiondet.config import add_diffusiondet_config


# SETUP
config_path = "configs2/diffdet.kitti.res50.yaml"
config_path = "final_outputs/diffdet_cl_kitti_100e/config.yaml"
# weights_path = "output/kitti_res50_90kIter/model_final.pth"
input_dir = "samples_per_class"
output_dir = "output/detection/res_50"
output_dir = "final_outputs/diffdet_cl_kitti_100e/preds2/"
weights_path = "final_outputs/diffdet_cl_kitti_100e/model_final.pth"
confidence_threshold = 0.3
os.makedirs(output_dir, exist_ok=True)


# CLASS INFO
class_names = ["Car", "Van", "Truck", "Pedestrian", "Person_sitting", "Cyclist", "Tram"]
MetadataCatalog.get("kitti_val_split").set(thing_classes=class_names)
metadata = MetadataCatalog.get("kitti_val_split")

# YOLO-style fixed BGR colors (use vivid distinct colors)
class_colors = {
    "Car": (0, 0, 255),             # Red
    "Van": (255, 0, 0),             # Blue
    "Truck": (255, 0, 255),         # Magenta
    "Pedestrian": (0, 165, 255),    # Orange
    "Person_sitting": (255, 0, 127),# Pink
    "Cyclist": (0, 255, 0),         # Green
    "Tram": (255, 255, 0)           # Cyan
}


# LOAD CONFIG
cfg = get_cfg()
add_diffusiondet_config(cfg)

with open(config_path, "r") as f:
    cfg_dict = yaml.safe_load(f)
cfg_dict.pop("MODEL_EMA", None)
cfg.merge_from_other_cfg(cfg.__class__(cfg_dict))

cfg.MODEL.WEIGHTS = weights_path
cfg.INPUT.MIN_SIZE_TEST = 1200
cfg.INPUT.CROP.ENABLED = False
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = confidence_threshold
cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# BUILD MODEL
model = build_model(cfg)
DetectionCheckpointer(model).load(cfg.MODEL.WEIGHTS)
model.eval()


# COLLECT IMAGES
image_paths = []
for cls_name in os.listdir(input_dir):
    cls_dir = os.path.join(input_dir, cls_name)
    if not os.path.isdir(cls_dir):
        continue
    for fname in os.listdir(cls_dir):
        if fname.endswith((".png", ".jpg")):
            image_paths.append(os.path.join(cls_dir, fname))


# INFERENCE + VISUALIZATION (YOLO STYLE)
for image_path in tqdm(image_paths, desc="🔍 Running inference"):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Skipped invalid image: {image_path}")
        continue

    height, width = image.shape[:2]
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).copy()

    inputs = [{"image": torch.as_tensor(image_rgb.transpose(2, 0, 1)).float(), 
               "height": height, "width": width}]

    with torch.no_grad():
        outputs = model(inputs)[0]
        instances = outputs["instances"]
        conf_mask = instances.scores > confidence_threshold
        filtered_instances = instances[conf_mask]

    # Draw boxes manually (YOLO style)
    for i in range(len(filtered_instances)):
        box = filtered_instances.pred_boxes.tensor[i].cpu().numpy().astype(int)
        cls_id = int(filtered_instances.pred_classes[i])
        score = float(filtered_instances.scores[i])
        label = class_names[cls_id]
        color = class_colors[label]

        x1, y1, x2, y2 = box
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

        label_text = f"{label} {int(score * 100)}%"
        ((text_w, text_h), _) = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(image, (x1, y1 - text_h - 4), (x1 + text_w, y1), color, -1)
        cv2.putText(image, label_text, (x1, y1 - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    # Save image
    out_path = os.path.join(output_dir, os.path.basename(image_path))
    cv2.imwrite(out_path, image)

print(f"\nAll detections saved to: {output_dir}")
