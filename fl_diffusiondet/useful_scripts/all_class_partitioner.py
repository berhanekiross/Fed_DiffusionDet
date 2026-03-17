
"""
KITTI Dataset Re-partitioning Script (FIXED)
Creates new partitions with proper client mapping where images are duplicated across clients.
Each image containing a specific class goes to that client's partition.
Supports both DiffusionDet (COCO format) and YOLO implementations.
"""

import os
import json
import shutil
import random
from pathlib import Path
from collections import defaultdict
from PIL import Image
import argparse

# KITTI class mapping (7 object classes, IDs 0-6)
KITTI_CLASSES = {
    0: "Car",
    1: "Van", 
    2: "Truck",
    3: "Pedestrian",
    4: "Person_sitting", 
    5: "Cyclist",
    6: "Tram"
}

# CLIENT MAPPING: Each client gets images containing specific object classes
# Note: Person_sitting and Tram are merged into Tram_Sitting client
CLIENT_MAPPING = {
    "Car": [0],           # Images with Car objects
    "Van": [1],           # Images with Van objects
    "Truck": [2],         # Images with Truck objects
    "Pedestrian": [3],    # Images with Pedestrian objects
    "Tram_Sitting": [4, 6],  # Images with Person_sitting OR Tram objects (MERGED)
    "Cyclist": [5]        # Images with Cyclist objects
}

def parse_kitti_label(label_path):
    """Parse KITTI label file and return list of objects."""
    objects = []
    
    if not os.path.exists(label_path):
        return objects
    
    with open(label_path, 'r') as f:
        for line in f.readlines():
            parts = line.strip().split()
            if len(parts) < 15:
                continue
                
            class_name = parts[0]
            
            # Map class name to ID
            class_id = None
            for cid, cname in KITTI_CLASSES.items():
                if cname == class_name:
                    class_id = cid
                    break
            
            if class_id is None:
                continue
                
            # Parse bounding box (in KITTI format: left, top, right, bottom)
            left = float(parts[4])
            top = float(parts[5]) 
            right = float(parts[6])
            bottom = float(parts[7])
            
            objects.append({
                'class_id': class_id,
                'class_name': class_name,
                'bbox': [left, top, right, bottom]
            })
    
    return objects

def get_image_dimensions(image_path):
    """Get actual image dimensions."""
    try:
        with Image.open(image_path) as img:
            return img.size  # (width, height)
    except Exception as e:
        print(f"Error reading image {image_path}: {e}")
        return (1242, 375)  # KITTI default fallback

def create_directory_structure(base_dir):
    """Create the directory structure for all clients."""
    base_path = Path(base_dir)
    
    # Create client directories
    for client_name in CLIENT_MAPPING.keys():
        client_dir = base_path / client_name
        
        # Create images subdirectories
        (client_dir / "images" / "train").mkdir(parents=True, exist_ok=True)
        (client_dir / "images" / "val").mkdir(parents=True, exist_ok=True)
        
        # Create labels subdirectories  
        (client_dir / "labels" / "train").mkdir(parents=True, exist_ok=True)
        (client_dir / "labels" / "val").mkdir(parents=True, exist_ok=True)
    
    # Create global_val directory
    (base_path / "global_val" / "images").mkdir(parents=True, exist_ok=True)
    (base_path / "global_val" / "labels").mkdir(parents=True, exist_ok=True)
    
    print(f"Created directory structure in {base_dir}")

def partition_images_by_class(images_dir, labels_dir):
    """
    Partition images based on object classes present.
    Each image goes to ALL clients whose object classes appear in that image.
    This means images will be duplicated across multiple clients.
    """
    
    # Find all images and their corresponding labels
    image_files = []
    for ext in ['*.png', '*.jpg', '*.jpeg']:
        image_files.extend(Path(images_dir).glob(ext))
    
    # Group images by client based on object classes present
    client_images = defaultdict(list)
    
    print(f"Processing {len(image_files)} images...")
    
    for image_path in image_files:
        image_name = image_path.stem
        label_path = Path(labels_dir) / f"{image_name}.txt"
        
        # Parse objects in this image
        objects = parse_kitti_label(label_path)
        
        if not objects:
            continue
            
        # Determine which clients this image belongs to
        image_classes = set(obj['class_id'] for obj in objects)
        
        # Add this image to ALL relevant clients
        for client_name, client_class_ids in CLIENT_MAPPING.items():
            # If image contains ANY object class for this client
            if any(class_id in image_classes for class_id in client_class_ids):
                client_images[client_name].append({
                    'image_path': str(image_path),
                    'label_path': str(label_path),
                    'image_name': image_name,
                    'objects': objects
                })
    
    # Print statistics
    print("\nPartitioning Statistics:")
    total_unique_images = len(image_files)
    total_distributed_images = sum(len(images) for images in client_images.values())
    
    for client_name, images in client_images.items():
        unique_images = len(set(img['image_name'] for img in images))
        print(f"  {client_name}: {len(images)} images ({unique_images} unique)")
    
    print(f"\nTotal unique images: {total_unique_images}")
    print(f"Total distributed images: {total_distributed_images}")
    print(f"Duplication factor: {total_distributed_images / total_unique_images:.2f}")
    
    return client_images

def split_train_val(client_images, val_ratio=0.2, random_seed=42):
    """Split each client's images into train/val."""
    random.seed(random_seed)
    
    client_splits = {}
    
    for client_name, images in client_images.items():
        # Shuffle images
        shuffled_images = images.copy()
        random.shuffle(shuffled_images)
        
        # Calculate split point
        val_count = max(1, int(len(images) * val_ratio))  # At least 1 for val
        train_count = len(images) - val_count
        
        client_splits[client_name] = {
            'train': shuffled_images[:train_count],
            'val': shuffled_images[train_count:]
        }
        
        print(f"{client_name}: {train_count} train, {val_count} val")
    
    return client_splits

def copy_files_to_partitions(client_splits, base_partition_dir, images_dir):
    """Copy/symlink images and labels to partition directories."""
    
    for client_name, splits in client_splits.items():
        client_dir = Path(base_partition_dir) / client_name
        
        for split_name, images in splits.items():
            print(f"Processing {client_name} {split_name}: {len(images)} images")
            
            for image_data in images:
                image_name = image_data['image_name']
                src_image = image_data['image_path']
                src_label = image_data['label_path']
                
                # Determine file extension
                image_ext = Path(src_image).suffix
                
                # Destination paths
                dst_image = client_dir / "images" / split_name / f"{image_name}{image_ext}"
                dst_label = client_dir / "labels" / split_name / f"{image_name}.txt"
                
                # Copy image (or create symlink for space efficiency)
                if not dst_image.exists():
                    try:
                        # Create symlink to save space
                        dst_image.symlink_to(os.path.abspath(src_image))
                    except:
                        # Fallback to copying if symlink fails
                        shutil.copy2(src_image, dst_image)
                
                # Copy label file (keep ALL object classes in labels)
                if os.path.exists(src_label) and not dst_label.exists():
                    shutil.copy2(src_label, dst_label)

def create_global_val_set(client_splits, base_partition_dir, global_val_per_client=20):
    """Create global validation set by sampling from each client's val set."""
    
    global_val_dir = Path(base_partition_dir) / "global_val"
    selected_images = {}  # Use dict to avoid duplicates by image_name
    
    for client_name, splits in client_splits.items():
        val_images = splits['val']
        
        # Sample images for global val (up to global_val_per_client)
        sample_count = min(global_val_per_client, len(val_images))
        sampled_images = random.sample(val_images, sample_count)
        
        print(f"Global val: Selected {sample_count} images from {client_name}")
        
        for image_data in sampled_images:
            image_name = image_data['image_name']
            
            # Only add if not already selected (to avoid duplicates)
            if image_name not in selected_images:
                selected_images[image_name] = image_data
    
    # Now copy the selected images to global_val
    for image_name, image_data in selected_images.items():
        src_image = image_data['image_path']
        src_label = image_data['label_path']
        
        # Determine file extension
        image_ext = Path(src_image).suffix
        
        # Destination paths  
        dst_image = global_val_dir / "images" / f"{image_name}{image_ext}"
        dst_label = global_val_dir / "labels" / f"{image_name}.txt"
        
        # Copy to global val
        if not dst_image.exists():
            try:
                dst_image.symlink_to(os.path.abspath(src_image))
            except:
                shutil.copy2(src_image, dst_image)
        
        if os.path.exists(src_label) and not dst_label.exists():
            shutil.copy2(src_label, dst_label)
    
    print(f"Created global validation set with {len(selected_images)} unique images")
    return list(selected_images.values())

def convert_bbox_to_coco(kitti_bbox, img_width, img_height):
    """Convert KITTI bbox to COCO format."""
    left, top, right, bottom = kitti_bbox
    
    x = left
    y = top  
    width = right - left
    height = bottom - top
    
    # Ensure bounds
    x = max(0, x)
    y = max(0, y)
    width = min(width, img_width - x)
    height = min(height, img_height - y)
    
    return [x, y, width, height]

def create_coco_annotation(client_splits, base_partition_dir, fl_dataset_dir):
    """Create COCO-style JSON annotations for DiffusionDet."""
    
    fl_annotations_dir = Path(fl_dataset_dir) / "annotations"
    fl_annotations_dir.mkdir(parents=True, exist_ok=True)
    
    # COCO template
    def create_coco_template():
        return {
            "info": {
                "description": "KITTI Dataset - Federated Learning Partition",
                "version": "1.0",
                "year": 2024,
                "contributor": "FL KITTI Project",
                "date_created": "2024",
                "url": "http://www.cvlibs.net/datasets/kitti/"
            },
            "licenses": [
                {
                    "id": 1,
                    "name": "KITTI License",
                    "url": "http://www.cvlibs.net/datasets/kitti/"
                }
            ],
            "categories": [
                {"id": cid + 1, "name": cname, "supercategory": "object"} 
                for cid, cname in KITTI_CLASSES.items()
            ],
            "images": [],
            "annotations": []
        }
    
    # Process each client and split
    for client_name, splits in client_splits.items():
        for split_name, images in splits.items():
            
            coco_data = create_coco_template()
            annotation_id = 1
            
            for img_id, image_data in enumerate(images, 1):
                image_name = image_data['image_name']
                src_image = image_data['image_path']
                
                # Get actual image dimensions
                img_width, img_height = get_image_dimensions(src_image)
                
                # Determine file extension
                image_ext = Path(src_image).suffix
                
                # Add image info
                coco_data["images"].append({
                    "id": img_id,
                    "file_name": f"{image_name}{image_ext}",
                    "width": img_width,
                    "height": img_height
                })
                
                # Add ALL annotations for this image (not filtered by client)
                # This maintains the original object class distribution
                for obj in image_data['objects']:
                    coco_bbox = convert_bbox_to_coco(obj['bbox'], img_width, img_height)
                    area = coco_bbox[2] * coco_bbox[3]  # width * height
                    
                    coco_data["annotations"].append({
                        "id": annotation_id,
                        "image_id": img_id,
                        "category_id": obj['class_id'] + 1,  # COCO uses 1-based IDs
                        "bbox": coco_bbox,
                        "area": area,
                        "iscrowd": 0
                    })
                    
                    annotation_id += 1
            
            # Save COCO JSON
            json_filename = f"{split_name}_{client_name}.json"
            json_path = fl_annotations_dir / json_filename
            
            with open(json_path, 'w') as f:
                json.dump(coco_data, f, indent=2)
            
            print(f"Created {json_filename}: {len(coco_data['images'])} images, {len(coco_data['annotations'])} annotations")

def create_global_val_coco(base_partition_dir, fl_dataset_dir):
    """Create COCO JSON for global validation set."""
    
    global_val_dir = Path(base_partition_dir) / "global_val"
    fl_annotations_dir = Path(fl_dataset_dir) / "annotations"
    
    # COCO template
    coco_data = {
        "info": {
            "description": "KITTI Dataset - Global Validation Set",
            "version": "1.0",
            "year": 2024,
            "contributor": "FL KITTI Project",
            "date_created": "2024",
            "url": "http://www.cvlibs.net/datasets/kitti/"
        },
        "licenses": [
            {
                "id": 1,
                "name": "KITTI License", 
                "url": "http://www.cvlibs.net/datasets/kitti/"
            }
        ],
        "categories": [
            {"id": cid + 1, "name": cname, "supercategory": "object"}
            for cid, cname in KITTI_CLASSES.items()
        ],
        "images": [],
        "annotations": []
    }
    
    # Process global val images
    global_images_dir = global_val_dir / "images"
    global_labels_dir = global_val_dir / "labels"
    
    annotation_id = 1
    
    for img_id, image_path in enumerate(global_images_dir.glob("*"), 1):
        if image_path.is_file() and image_path.suffix.lower() in ['.png', '.jpg', '.jpeg']:
            image_name = image_path.stem
            label_path = global_labels_dir / f"{image_name}.txt"
            
            # Get image dimensions
            img_width, img_height = get_image_dimensions(image_path)
            
            # Add image info
            coco_data["images"].append({
                "id": img_id,
                "file_name": image_path.name,
                "width": img_width,
                "height": img_height
            })
            
            # Parse and add annotations
            objects = parse_kitti_label(label_path)
            for obj in objects:
                coco_bbox = convert_bbox_to_coco(obj['bbox'], img_width, img_height)
                area = coco_bbox[2] * coco_bbox[3]
                
                coco_data["annotations"].append({
                    "id": annotation_id,
                    "image_id": img_id,
                    "category_id": obj['class_id'] + 1,
                    "bbox": coco_bbox,
                    "area": area,
                    "iscrowd": 0
                })
                
                annotation_id += 1
    
    # Save global val JSON
    json_path = fl_annotations_dir / "global_val.json"
    with open(json_path, 'w') as f:
        json.dump(coco_data, f, indent=2)
    
    print(f"Created global_val.json: {len(coco_data['images'])} images, {len(coco_data['annotations'])} annotations")

def create_fl_dataset_structure(client_splits, base_partition_dir, fl_dataset_dir, images_dir):
    """Create FL dataset structure with symlinks and partition files."""
    
    fl_dataset_path = Path(fl_dataset_dir)
    fl_images_dir = fl_dataset_path / "images"
    fl_partitions_dir = fl_dataset_path / "partitions"
    
    # Create directories
    fl_images_dir.mkdir(parents=True, exist_ok=True)
    fl_partitions_dir.mkdir(parents=True, exist_ok=True)
    
    # Create symlinks for all images and partition files
    all_images = set()
    
    for client_name, splits in client_splits.items():
        for split_name, images in splits.items():
            
            # Create partition file for YOLO (lowercase client name)
            partition_file = fl_partitions_dir / f"{split_name}_{client_name.lower()}.txt"
            
            with open(partition_file, 'w') as f:
                for image_data in images:
                    image_name = image_data['image_name']
                    src_image = image_data['image_path']
                    
                    # Determine file extension
                    image_ext = Path(src_image).suffix
                    full_image_name = f"{image_name}{image_ext}"
                    
                    # Create symlink in fl_dataset/images if not exists
                    dst_image = fl_images_dir / full_image_name
                    if not dst_image.exists():
                        try:
                            dst_image.symlink_to(os.path.abspath(src_image))
                        except:
                            shutil.copy2(src_image, dst_image)
                    
                    # Write to partition file
                    f.write(f"{full_image_name}\n")
                    all_images.add(full_image_name)
            
            print(f"Created partition file: {partition_file.name}")
    
    # Create symlinks for global val images
    global_val_dir = Path(base_partition_dir) / "global_val" / "images"
    if global_val_dir.exists():
        for image_path in global_val_dir.glob("*"):
            if image_path.is_file() and image_path.suffix.lower() in ['.png', '.jpg', '.jpeg']:
                dst_image = fl_images_dir / image_path.name
                if not dst_image.exists():
                    try:
                        dst_image.symlink_to(os.path.abspath(image_path))
                    except:
                        shutil.copy2(image_path, dst_image)
    
    print(f"Created FL dataset structure with {len(all_images)} unique images")

def main():
    # Configuration
    IMAGES_DIR = "/mimer/NOBACKUP/groups/naiss2024-5-153/old_projects/Berhane/labelled_kitti/training/image_2/"
    LABELS_DIR = "/mimer/NOBACKUP/groups/naiss2024-5-153/old_projects/Berhane/labelled_kitti/training/label_2/"
    BASE_PARTITION_DIR = "/mimer/NOBACKUP/groups/naiss2024-5-153/old_projects/Berhane/labelled_kitti/all_class_partition/"
    FL_DATASET_DIR = "./fl_dataset/"
    
    print("=== KITTI Re-partitioning Script (FIXED) ===")
    print(f"Images dir: {IMAGES_DIR}")
    print(f"Labels dir: {LABELS_DIR}")
    print(f"Output partition dir: {BASE_PARTITION_DIR}")
    print(f"FL dataset dir: {FL_DATASET_DIR}")
    print(f"Client mapping: {CLIENT_MAPPING}")
    
    # Step 1: Create directory structure
    print("\n1. Creating directory structure...")
    create_directory_structure(BASE_PARTITION_DIR)
    
    # Step 2: Partition images by class
    print("\n2. Partitioning images by object classes...")
    client_images = partition_images_by_class(IMAGES_DIR, LABELS_DIR)
    
    # Step 3: Split into train/val
    print("\n3. Splitting into train/val...")
    client_splits = split_train_val(client_images, val_ratio=0.2)
    
    # Step 4: Copy files to partitions
    print("\n4. Copying files to partition directories...")
    copy_files_to_partitions(client_splits, BASE_PARTITION_DIR, IMAGES_DIR)
    
    # Step 5: Create global validation set
    print("\n5. Creating global validation set...")
    create_global_val_set(client_splits, BASE_PARTITION_DIR)
    
    # Step 6: Create COCO annotations
    print("\n6. Creating COCO-style annotations...")
    create_coco_annotation(client_splits, BASE_PARTITION_DIR, FL_DATASET_DIR)
    create_global_val_coco(BASE_PARTITION_DIR, FL_DATASET_DIR)
    
    # Step 7: Create FL dataset structure
    print("\n7. Creating FL dataset structure...")
    create_fl_dataset_structure(client_splits, BASE_PARTITION_DIR, FL_DATASET_DIR, IMAGES_DIR)
    
    print("\n=== PARTITIONING COMPLETE ===")
    print(f"New partitions created in: {BASE_PARTITION_DIR}")
    print(f"FL dataset created in: {FL_DATASET_DIR}")
    print("\nFinal client list:")
    for client_name, class_ids in CLIENT_MAPPING.items():
        class_names = [KITTI_CLASSES[cid] for cid in class_ids]
        print(f"  {client_name}: {class_names}")
    
    print("\nKey changes made:")
    print("- Person_sitting and Tram merged into Tram_Sitting client")
    print("- Images are duplicated across clients (one image can be in multiple clients)")
    print("- All object classes preserved in labels and COCO annotations")
    print("- Global validation set created from unique images across clients")
    print("- Both YOLO and DiffusionDet formats supported")

if __name__ == "__main__":
    main()