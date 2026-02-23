import os
import json
import cv2
from tqdm import tqdm

# === CONFIG ===
KITTI_LABEL_DIR = '/mimer/NOBACKUP/groups/naiss2024-5-153/Berhane/labelled_kitti/training/label_2'
KITTI_IMAGE_DIR = '/mimer/NOBACKUP/groups/naiss2024-5-153/Berhane/labelled_kitti/training/image_2'

OUTPUT_JSON = 'datasets/kitti_coco/annotations/instances_train2017.json'

VALID_CLASSES = [
    'Car', 'Van', 'Truck', 'Pedestrian', 'Person_sitting', 
    'Cyclist', 'Tram', 'DontCare'
]

IMAGE_SUFFIX = '.png'  # or '.jpg' if converted

def get_image_info(file_id, height, width):
    return {
        "id": int(file_id),
        "file_name": f"{file_id:06d}{IMAGE_SUFFIX}",
        "height": height,
        "width": width
    }

def get_annotation(obj_id, file_id, category_id, bbox):
    x, y, w, h = bbox
    return {
        "id": obj_id,
        "image_id": int(file_id),
        "category_id": category_id,
        "bbox": [x, y, w, h],
        "area": w * h,
        "iscrowd": 0
    }

def parse_label_line(line):
    parts = line.strip().split()
    cls = parts[0]
    bbox = list(map(float, parts[4:8]))
    return cls, bbox

def main():
    images = []
    annotations = []
    categories = [{"id": i+1, "name": name} for i, name in enumerate(VALID_CLASSES)]
    class_to_id = {name: i+1 for i, name in enumerate(VALID_CLASSES)}

    obj_id = 1
    for fname in tqdm(sorted(os.listdir(KITTI_LABEL_DIR))):
        if not fname.endswith('.txt'):
            continue
        file_id = int(fname.split('.')[0])
        img_path = os.path.join(KITTI_IMAGE_DIR, f"{file_id:06d}{IMAGE_SUFFIX}")
        if not os.path.exists(img_path):
            continue
        img = cv2.imread(img_path)
        height, width = img.shape[:2]
        images.append(get_image_info(file_id, height, width))

        with open(os.path.join(KITTI_LABEL_DIR, fname), 'r') as f:
            for line in f:
                cls, box = parse_label_line(line)
                if cls not in class_to_id:
                    continue
                x1, y1, x2, y2 = box
                w = x2 - x1
                h = y2 - y1
                if w <= 0 or h <= 0:
                    continue
                annotations.append(get_annotation(obj_id, file_id, class_to_id[cls], [x1, y1, w, h]))
                obj_id += 1

    coco_format = {
        "images": images,
        "annotations": annotations,
        "categories": categories
    }

    os.makedirs(os.path.dirname(OUTPUT_JSON), exist_ok=True)
    with open(OUTPUT_JSON, 'w') as f:
        json.dump(coco_format, f, indent=2)

    print(f"COCO-format annotations saved to: {OUTPUT_JSON}")

if __name__ == "__main__":
    main()
