import os
import random
import shutil
from pathlib import Path

# Classes of interest
classes = ['Car', 'Van', 'Truck', 'Pedestrian', 'Person_sitting', 'Cyclist', 'Tram']
labels_dir = "/mimer/NOBACKUP/groups/naiss2024-5-153/Berhane/labelled_kitti/training/labels/val"
images_dir = "/mimer/NOBACKUP/groups/naiss2024-5-153/Berhane/labelled_kitti/training/images/val"
output_dir = "samples_per_class"
os.makedirs(output_dir, exist_ok=True)

# Map string class -> integer (same as YOLO format)
class_to_id = {cls: i for i, cls in enumerate(classes)}

# Find images with each class
samples = {cls: [] for cls in classes}
for label_file in os.listdir(labels_dir):
    with open(os.path.join(labels_dir, label_file)) as f:
        content = f.read()
        for line in content.strip().split('\n'):
            cls_id = int(line.split()[0])
            for cls, cls_index in class_to_id.items():
                if cls_id == cls_index:
                    samples[cls].append(label_file.replace(".txt", ".png"))
                    break

# Randomly sample 3 images per class
for cls in classes:
    selected = random.sample(list(set(samples[cls])), k=min(3, len(samples[cls])))
    out_dir = os.path.join(output_dir, cls)
    os.makedirs(out_dir, exist_ok=True)
    for file in selected:
        src = os.path.join(images_dir, file)
        dst = os.path.join(out_dir, file)
        shutil.copy(src, dst)

print("✅ Sample images copied to:", output_dir)
