
import os
from ultralytics import YOLO
from tqdm import tqdm


# Configuration
model_path = "yolo/runs_250/detect/train/weights/best.pt"
input_dir = "samples_per_class"
output_dir = "yolo/detection/res_50"
confidence_threshold = 0.5


# Load YOLOv8 model
model = YOLO(model_path)


# Collect image paths
image_paths = []
for cls_name in os.listdir(input_dir):
    cls_dir = os.path.join(input_dir, cls_name)
    if not os.path.isdir(cls_dir):
        continue
    for fname in os.listdir(cls_dir):
        if fname.endswith((".png", ".jpg")):
            image_paths.append(os.path.join(cls_dir, fname))


# Inference using YOLO’s built-in rendering and saving
for image_path in tqdm(image_paths, desc="YOLO Inference"):
    model(
        source=image_path,
        conf=confidence_threshold,
        save=True,
        project=output_dir,
        name="",  # avoid subdir like 'predict/'
        exist_ok=True
    )

print(f"\nYOLO inference results saved to: {output_dir}")
