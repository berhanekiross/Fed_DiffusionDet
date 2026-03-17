# This script partitions the KITTI dataset to create fl_dataset

import os
import json
import torch
import numpy as np
from pathlib import Path
from collections import defaultdict
import cv2

# Redefine constants after kernel reset
base_dir = Path("/mimer/NOBACKUP/groups/naiss2024-5-153/old_projects/Berhane/labelled_kitti")
yolo_partition_dir = base_dir / "partitions_yolo"
original_label_dir = base_dir / "training" / "label_2"
original_image_dir = base_dir / "training" / "image_2"
output_dir = base_dir / "fl_diffdet/fl_dataset"

annotations_dir = output_dir / "annotations"
images_dir = output_dir / "images"
partitions_dir = output_dir / "partitions"

# Create necessary directories
annotations_dir.mkdir(parents=True, exist_ok=True)
images_dir.mkdir(parents=True, exist_ok=True)
partitions_dir.mkdir(parents=True, exist_ok=True)

# Helper function to convert KITTI labels to COCO-style annotations

def kitti_to_coco(image_list, label_dir, image_dir):
    coco = {
        "images": [],
        "annotations": [],
        "categories": [
            {"id": 0, "name": "Car"},
            {"id": 1, "name": "Van"},
            {"id": 2, "name": "Truck"},
            {"id": 3, "name": "Pedestrian"},
            {"id": 4, "name": "Person_sitting"},
            {"id": 5, "name": "Cyclist"},
            {"id": 6, "name": "Tram"}
        ]
    }
    ann_id = 0
    for img_id, img_file in enumerate(image_list):
        img_name = img_file.name
        stem = img_file.stem
        image_path = image_dir / img_file.name
        label_path = label_dir / f"{stem}.txt"

        if not label_path.exists():
            continue

        # 🔧 Load image to get actual size
        image = cv2.imread(str(image_path))
        if image is None:
            continue

        height, width = image.shape[:2]

        # Add image metadata
        coco["images"].append({
            "id": img_id,
            "file_name": img_name,
            "width": width,
            "height": height
        })

        with open(label_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 15:
                    continue
                cls_name = parts[0]
                if cls_name not in ["Car", "Van", "Truck", "Pedestrian", "Person_sitting", "Cyclist", "Tram"]:
                    continue
                x1, y1, x2, y2 = map(float, parts[4:8])
                coco["annotations"].append({
                    "id": ann_id,
                    "image_id": img_id,
                    "category_id": ["Car", "Van", "Truck", "Pedestrian", "Person_sitting", "Cyclist", "Tram"].index(cls_name),
                    "bbox": [x1, y1, x2 - x1, y2 - y1],
                    "area": (x2 - x1) * (y2 - y1),
                    "iscrowd": 0
                })
                ann_id += 1
    return coco

# Main loop per client
global_val_images = set()

for class_dir in yolo_partition_dir.iterdir():
    if not class_dir.is_dir():
        continue
    class_name = class_dir.name.capitalize()
    images_train = list((class_dir / "images" / "train").glob("*.png"))
    images_val = list((class_dir / "images" / "val").glob("*.png"))

    # Save client image lists
    with open(partitions_dir / f"train_{class_name}.txt", "w") as f:
        for img in images_train:
            f.write(f"{img.name}\n")
    with open(partitions_dir / f"val_{class_name}.txt", "w") as f:
        for img in images_val:
            f.write(f"{img.name}\n")

    # Create symlinks
    for img_path in images_train + images_val:
        dest = images_dir / img_path.name
        if not dest.exists():
            os.symlink(img_path.resolve(), dest)

    # Convert and save annotations
    coco_train = kitti_to_coco(images_train, original_label_dir, original_image_dir)
    with open(annotations_dir / f"train_{class_name}.json", "w") as f:
        json.dump(coco_train, f)

    coco_val = kitti_to_coco(images_val, original_label_dir, original_image_dir)
    with open(annotations_dir / f"val_{class_name}.json", "w") as f:
        json.dump(coco_val, f)

    global_val_images.update(images_val)

# Global val set
global_val_list = sorted(global_val_images)
coco_global_val = kitti_to_coco(global_val_list, original_label_dir, original_image_dir)
with open(annotations_dir / "global_val.json", "w") as f:
    json.dump(coco_global_val, f)


