import json
import os
from pathlib import Path
from datetime import datetime

def create_kitti_coco_json_with_dontcare():
    # Paths
    image_dir = "/mimer/NOBACKUP/groups/naiss2024-5-153/old_projects/Berhane/labelled_kitti/training/image_2/"
    label_dir = "/mimer/NOBACKUP/groups/naiss2024-5-153/old_projects/Berhane/labelled_kitti/training/label_2/"
    output_path = "datasets/kitti_coco/annotations/train2017_with_misc.json"
    
    # KITTI categories (including Misc as DontCare equivalent)
    categories = [
        {"id": 0, "name": "Car", "supercategory": "vehicle"},
        {"id": 1, "name": "Van", "supercategory": "vehicle"},
        {"id": 2, "name": "Truck", "supercategory": "vehicle"},
        {"id": 3, "name": "Pedestrian", "supercategory": "person"},
        {"id": 4, "name": "Person_sitting", "supercategory": "person"},
        {"id": 5, "name": "Cyclist", "supercategory": "person"},
        {"id": 6, "name": "Tram", "supercategory": "vehicle"},
        {"id": 7, "name": "Misc", "supercategory": "ignore"}
    ]
    
    # Category name to ID mapping
    cat_name_to_id = {cat["name"]: cat["id"] for cat in categories}
    
    # Initialize COCO format structure
    coco_data = {
        "info": {
            "description": "KITTI Dataset in COCO format (with Misc/DontCare)",
            "version": "1.0",
            "year": 2024,
            "contributor": "KITTI Team",
            "date_created": datetime.now().strftime("%Y-%m-%d")
        },
        "licenses": [
            {
                "id": 1,
                "name": "KITTI License",
                "url": "http://www.cvlibs.net/datasets/kitti/"
            }
        ],
        "categories": categories,
        "images": [],
        "annotations": []
    }
    
    # Get all image files
    image_files = sorted([f for f in os.listdir(image_dir) if f.endswith('.png')])
    
    annotation_id = 1
    
    for img_id, img_file in enumerate(image_files):
        # Get image path and dimensions
        img_path = os.path.join(image_dir, img_file)
        
        # For KITTI, we'll use standard dimensions, but you can use PIL to get actual dimensions
        # from PIL import Image
        # img = Image.open(img_path)
        # width, height = img.size
        
        # Standard KITTI dimensions (you can modify if needed)
        if img_file.startswith('00000') and int(img_file.split('.')[0]) < 1000:
            width, height = 1224, 370  # Some early images have different dimensions
        else:
            width, height = 1242, 375  # Standard KITTI dimensions
        
        # Add image info
        coco_data["images"].append({
            "id": img_id,
            "file_name": img_file,
            "width": width,
            "height": height,
            "date_captured": "2011-09-26"
        })
        
        # Process corresponding label file
        label_file = img_file.replace('.png', '.txt')
        label_path = os.path.join(label_dir, label_file)
        
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                lines = f.readlines()
            
            for line in lines:
                parts = line.strip().split()
                if len(parts) < 15:  # KITTI format has 15 fields
                    continue
                
                class_name = parts[0]
                
                # Skip if class not in our categories (map DontCare to Misc)
                if class_name == "DontCare":
                    class_name = "Misc"
                elif class_name not in cat_name_to_id:
                    continue
                
                # KITTI bbox format: left, top, right, bottom (parts[4:8])
                left = float(parts[4])
                top = float(parts[5])
                right = float(parts[6])
                bottom = float(parts[7])
                
                # Convert to COCO format: x, y, width, height
                x = left
                y = top
                width_bbox = right - left
                height_bbox = bottom - top
                
                # Skip invalid bboxes
                if width_bbox <= 0 or height_bbox <= 0:
                    continue
                
                area = width_bbox * height_bbox
                
                # Add annotation
                coco_data["annotations"].append({
                    "id": annotation_id,
                    "image_id": img_id,
                    "category_id": cat_name_to_id[class_name],
                    "bbox": [x, y, width_bbox, height_bbox],
                    "area": area,
                    "iscrowd": 0,
                    "segmentation": []
                })
                
                annotation_id += 1
    
    # Save to JSON file
    with open(output_path, 'w') as f:
        json.dump(coco_data, f, indent=2)
    
    print(f"✅ Created {output_path}")
    print(f"📊 Statistics:")
    print(f"   - Images: {len(coco_data['images'])}")
    print(f"   - Annotations: {len(coco_data['annotations'])}")
    print(f"   - Categories: {len(coco_data['categories'])}")
    
    # Print category distribution
    cat_counts = {}
    for ann in coco_data['annotations']:
        cat_id = ann['category_id']
        cat_name = next(cat['name'] for cat in categories if cat['id'] == cat_id)
        cat_counts[cat_name] = cat_counts.get(cat_name, 0) + 1
    
    print(f"📈 Category distribution:")
    for cat_name, count in sorted(cat_counts.items()):
        print(f"   - {cat_name}: {count}")

if __name__ == "__main__":
    create_kitti_coco_json_with_dontcare()