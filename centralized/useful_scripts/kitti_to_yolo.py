import os
import cv2

# Paths
image_dir = "/mimer/NOBACKUP/groups/naiss2024-5-153/Berhane/labelled_kitti/training/image_2"
label_kitti_dir = "/mimer/NOBACKUP/groups/naiss2024-5-153/Berhane/labelled_kitti/training/label_2"
label_yolo_dir = "/mimer/NOBACKUP/groups/naiss2024-5-153/Berhane/labelled_kitti/training/label_yolo"

# Class list used in KITTI to YOLO
kitti_classes = ['Car', 'Van', 'Truck', 'Pedestrian', 'Person_sitting', 'Cyclist', 'Tram']
class_to_id = {cls: i for i, cls in enumerate(kitti_classes)}

# Ensure output dir exists
os.makedirs(label_yolo_dir, exist_ok=True)

# Gather errors
missing_images = []
converted_files = 0
skipped_objects = 0

# Process each KITTI label file
for label_file in sorted(os.listdir(label_kitti_dir)):
    if not label_file.endswith('.txt'):
        continue

    file_id = label_file.split('.')[0]
    image_path = os.path.join(image_dir, f"{file_id}.png")
    label_path = os.path.join(label_kitti_dir, label_file)
    output_label_path = os.path.join(label_yolo_dir, label_file)

    if not os.path.exists(image_path):
        missing_images.append(image_path)
        continue

    img = cv2.imread(image_path)
    height, width = img.shape[:2]
    yolo_lines = []

    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            cls = parts[0]
            if cls not in class_to_id:
                skipped_objects += 1
                continue
            x1, y1, x2, y2 = map(float, parts[4:8])
            cx = (x1 + x2) / 2 / width
            cy = (y1 + y2) / 2 / height
            w = (x2 - x1) / width
            h = (y2 - y1) / height
            yolo_lines.append(f"{class_to_id[cls]} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")

    with open(output_label_path, 'w') as f:
        f.write('\n'.join(yolo_lines))

    converted_files += 1

converted_files, skipped_objects, len(missing_images)
