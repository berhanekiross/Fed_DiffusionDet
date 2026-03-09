import os
import shutil
import random
from pathlib import Path

# Paths
base_dir = Path("/mimer/NOBACKUP/groups/naiss2024-5-153/old_projects/Berhane/labelled_kitti/training")
image_dir = base_dir / "image_2"
label_dir = base_dir / "label_yolo"

images_out = base_dir / "images"
labels_out = base_dir / "labels"

train_ratio = 0.8

# Collect all image filenames
image_files = sorted(image_dir.glob("*.png"))
random.seed(42)
random.shuffle(image_files)

# Split
split_idx = int(len(image_files) * train_ratio)
train_imgs = image_files[:split_idx]
val_imgs = image_files[split_idx:]

# Prepare output folders
for split in ["train", "val"]:
    (images_out / split).mkdir(parents=True, exist_ok=True)
    (labels_out / split).mkdir(parents=True, exist_ok=True)

# Move images and labels
for split, files in [("train", train_imgs), ("val", val_imgs)]:
    for img_path in files:
        lbl_path = label_dir / (img_path.stem + ".txt")

        shutil.copy(img_path, images_out / split / img_path.name)
        if lbl_path.exists():
            shutil.copy(lbl_path, labels_out / split / lbl_path.name)

# import ace_tools as tools; tools.display_dataframe_to_user(name="Split Summary", dataframe={
#     "Set": ["Train", "Val"],
#     "Images": [len(train_imgs), len(val_imgs)]
# })