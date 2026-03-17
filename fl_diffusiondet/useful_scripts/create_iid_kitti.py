#!/usr/bin/env python3
"""
Create IID partitions from existing non-IID KITTI FL dataset.
Generates both COCO-style annotations and YOLO-compatible text files.
"""

import json
import os
import shutil
from pathlib import Path
from collections import defaultdict
import random
import numpy as np
from typing import Dict, List, Any

def load_json(file_path: str) -> Dict[str, Any]:
    """Load JSON file."""
    with open(file_path, 'r') as f:
        return json.load(f)

def save_json(data: Dict[str, Any], file_path: str) -> None:
    """Save data to JSON file."""
    with open(file_path, 'w') as f:
        json.dump(data, f, separators=(',', ':'))

def pool_training_data(annotations_dir: str) -> tuple:
    """Pool all training data from non-IID annotations."""
    print("[POOL] Collecting all training data from non-IID partitions...")
    
    pooled_images = []
    pooled_annotations = []
    categories = None
    info = None
    
    # Get all train_*.json files (excluding global_val.json)
    train_files = [f for f in os.listdir(annotations_dir) if f.startswith('train_') and f.endswith('.json')]
    
    annotation_id_offset = 0
    image_id_mapping = {}  # old_id -> new_id
    next_image_id = 1
    
    for train_file in train_files:
        print(f"[POOL] Processing {train_file}")
        data = load_json(os.path.join(annotations_dir, train_file))
        
        # Store metadata from first file
        if categories is None:
            categories = data['categories']
            info = data['info']
        
        # Process images with ID remapping
        for img in data['images']:
            old_id = img['id']
            new_id = next_image_id
            image_id_mapping[old_id] = new_id
            
            img_copy = img.copy()
            img_copy['id'] = new_id
            pooled_images.append(img_copy)
            next_image_id += 1
        
        # Process annotations with remapped image IDs
        for ann in data['annotations']:
            ann_copy = ann.copy()
            ann_copy['id'] = ann_copy['id'] + annotation_id_offset
            ann_copy['image_id'] = image_id_mapping[ann['image_id']]
            pooled_annotations.append(ann_copy)
        
        annotation_id_offset += len(data['annotations'])
    
    print(f"[POOL] Pooled {len(pooled_images)} images and {len(pooled_annotations)} annotations")
    return pooled_images, pooled_annotations, categories, info

def create_balanced_partitions(images: List[Dict], annotations: List[Dict], num_clients: int = 6) -> tuple:
    """Create balanced IID partitions."""
    print(f"[PARTITION] Creating {num_clients} balanced IID partitions...")
    
    # Group annotations by image_id for efficient lookup
    img_to_anns = defaultdict(list)
    for ann in annotations:
        img_to_anns[ann['image_id']].append(ann)
    
    # Calculate objects per image for balanced distribution
    images_with_stats = []
    for img in images:
        img_id = img['id']
        num_objects = len(img_to_anns[img_id])
        images_with_stats.append({
            'image': img,
            'annotations': img_to_anns[img_id],
            'num_objects': num_objects
        })
    
    # Sort by number of objects for balanced distribution
    images_with_stats.sort(key=lambda x: x['num_objects'])
    
    # Initialize client partitions
    client_partitions = [{'images': [], 'annotations': [], 'total_objects': 0} for _ in range(num_clients)]
    
    # Distribute images in round-robin fashion for balance
    for i, img_data in enumerate(images_with_stats):
        client_idx = i % num_clients
        client_partitions[client_idx]['images'].append(img_data['image'])
        client_partitions[client_idx]['annotations'].extend(img_data['annotations'])
        client_partitions[client_idx]['total_objects'] += img_data['num_objects']
    
    # Print distribution statistics
    for i, partition in enumerate(client_partitions):
        print(f"[PARTITION] Client {i}: {len(partition['images'])} images, "
              f"{len(partition['annotations'])} annotations, "
              f"{partition['total_objects']} objects")
    
    return client_partitions

def create_validation_splits(client_partitions: List[Dict], val_ratio: float = 0.2) -> tuple:
    """Split each client's data into train/val."""
    print(f"[SPLIT] Creating train/val splits (val_ratio={val_ratio})...")
    
    train_partitions = []
    val_partitions = []
    
    for client_id, partition in enumerate(client_partitions):
        images = partition['images']
        
        # Shuffle and split
        random.shuffle(images)
        split_idx = int(len(images) * (1 - val_ratio))
        
        train_images = images[:split_idx]
        val_images = images[split_idx:]
        
        # Get image IDs for filtering annotations
        train_img_ids = {img['id'] for img in train_images}
        val_img_ids = {img['id'] for img in val_images}
        
        # Split annotations
        train_anns = [ann for ann in partition['annotations'] if ann['image_id'] in train_img_ids]
        val_anns = [ann for ann in partition['annotations'] if ann['image_id'] in val_img_ids]
        
        train_partitions.append({
            'images': train_images,
            'annotations': train_anns
        })
        
        val_partitions.append({
            'images': val_images,
            'annotations': val_anns
        })
        
        print(f"[SPLIT] Client {client_id}: Train={len(train_images)} imgs/{len(train_anns)} anns, "
              f"Val={len(val_images)} imgs/{len(val_anns)} anns")
    
    return train_partitions, val_partitions

def save_coco_annotations(partitions: List[Dict], output_dir: str, prefix: str, 
                         categories: List[Dict], info: Dict) -> None:
    """Save partitions as COCO-style JSON annotations."""
    print(f"[SAVE] Saving COCO annotations to {output_dir}")
    os.makedirs(output_dir, exist_ok=True)
    
    for client_id, partition in enumerate(partitions):
        # Reassign annotation IDs sequentially
        for i, ann in enumerate(partition['annotations']):
            ann['id'] = i + 1
        
        coco_data = {
            'info': info,
            'licenses': [],
            'categories': categories,
            'images': partition['images'],
            'annotations': partition['annotations']
        }
        
        output_file = os.path.join(output_dir, f"{prefix}_client_{client_id}.json")
        save_json(coco_data, output_file)
        print(f"[SAVE] Saved {output_file}")

def save_yolo_partitions(partitions: List[Dict], output_dir: str, prefix: str) -> None:
    """Save partitions as YOLO-compatible text files."""
    print(f"[SAVE] Saving YOLO partitions to {output_dir}")
    os.makedirs(output_dir, exist_ok=True)
    
    for client_id, partition in enumerate(partitions):
        output_file = os.path.join(output_dir, f"{prefix}_client_{client_id}.txt")
        
        with open(output_file, 'w') as f:
            for img in partition['images']:
                # Extract image filename from path
                img_filename = img['file_name']
                f.write(f"client_{client_id}/{img_filename}\n")
        
        print(f"[SAVE] Saved {output_file}")

def copy_global_validation(src_dir: str, dst_dir: str) -> None:
    """Copy global_val.json to IID annotations directory."""
    src_file = os.path.join(src_dir, 'global_val.json')
    dst_file = os.path.join(dst_dir, 'global_val.json')
    
    if os.path.exists(src_file):
        shutil.copy2(src_file, dst_file)
        print(f"[COPY] Copied global_val.json to {dst_file}")
        
        # Also create YOLO version
        data = load_json(src_file)
        yolo_file = os.path.join(os.path.dirname(dst_dir), 'partitions_iid', 'global_val.txt')
        os.makedirs(os.path.dirname(yolo_file), exist_ok=True)
        
        with open(yolo_file, 'w') as f:
            for img in data['images']:
                f.write(f"global/{img['file_name']}\n")
        print(f"[COPY] Created YOLO global_val.txt at {yolo_file}")
    else:
        print(f"[WARNING] {src_file} not found")

def main():
    # Configuration
    FL_DATASET_DIR = "fl_dataset"
    ANNOTATIONS_DIR = os.path.join(FL_DATASET_DIR, "annotations")
    ANNOTATIONS_IID_DIR = os.path.join(FL_DATASET_DIR, "annotations_iid")
    PARTITIONS_IID_DIR = os.path.join(FL_DATASET_DIR, "partitions_iid")
    NUM_CLIENTS = 6
    VAL_RATIO = 0.2
    RANDOM_SEED = 42
    
    # Set random seed for reproducibility
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    
    print(f"[START] Creating IID partitions with {NUM_CLIENTS} clients")
    print(f"[CONFIG] Source: {ANNOTATIONS_DIR}")
    print(f"[CONFIG] Output: {ANNOTATIONS_IID_DIR}, {PARTITIONS_IID_DIR}")
    print(f"[CONFIG] Validation ratio: {VAL_RATIO}")
    
    # Step 1: Pool all training data
    pooled_images, pooled_annotations, categories, info = pool_training_data(ANNOTATIONS_DIR)
    
    # Step 2: Create balanced IID partitions
    client_partitions = create_balanced_partitions(pooled_images, pooled_annotations, NUM_CLIENTS)
    
    # Step 3: Split into train/val
    train_partitions, val_partitions = create_validation_splits(client_partitions, VAL_RATIO)
    
    # Step 4: Save COCO annotations
    save_coco_annotations(train_partitions, ANNOTATIONS_IID_DIR, "train", categories, info)
    save_coco_annotations(val_partitions, ANNOTATIONS_IID_DIR, "val", categories, info)
    
    # Step 5: Save YOLO partitions
    save_yolo_partitions(train_partitions, PARTITIONS_IID_DIR, "train")
    save_yolo_partitions(val_partitions, PARTITIONS_IID_DIR, "val")
    
    # Step 6: Copy global validation
    copy_global_validation(ANNOTATIONS_DIR, ANNOTATIONS_IID_DIR)
    
    # Final statistics
    total_train_images = sum(len(p['images']) for p in train_partitions)
    total_val_images = sum(len(p['images']) for p in val_partitions)
    total_train_anns = sum(len(p['annotations']) for p in train_partitions)
    total_val_anns = sum(len(p['annotations']) for p in val_partitions)
    
    print("\n" + "="*60)
    print("IID PARTITIONING COMPLETE")
    print("="*60)
    print(f"Total training images: {total_train_images}")
    print(f"Total validation images: {total_val_images}")
    print(f"Total training annotations: {total_train_anns}")
    print(f"Total validation annotations: {total_val_anns}")
    print(f"Average images per client: {total_train_images // NUM_CLIENTS}")
    print(f"Average annotations per client: {total_train_anns // NUM_CLIENTS}")
    print("\nOutput directories:")
    print(f"  COCO annotations: {ANNOTATIONS_IID_DIR}")
    print(f"  YOLO partitions: {PARTITIONS_IID_DIR}")
    print("="*60)

if __name__ == "__main__":
    main()