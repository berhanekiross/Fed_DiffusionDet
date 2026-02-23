#!/usr/bin/env python3
"""
Create 1000-image test subset from ZOD dataset
Creates new COCO JSON files without copying images
"""

import json
import random
from pathlib import Path
from collections import Counter

def create_test_subset(subset_size=1000, seed=42):
    """Create test subset with balanced class distribution"""
    
    # Paths
    zod_base = Path("/mimer/NOBACKUP/groups/naiss2024-5-153/Berhane/zod_processed")
    annotations_dir = zod_base / "annotations/centralized"
    output_dir = zod_base / "annotations/test_subset"
    
    # Create output directory
    output_dir.mkdir(exist_ok=True)
    
    print(f"Creating {subset_size}-image test subset...")
    
    # Load full training dataset
    train_file = annotations_dir / "train_coco.json"
    with open(train_file, 'r') as f:
        train_data = json.load(f)
    
    print(f"Loaded {len(train_data['images'])} training images")
    
    # Set random seed for reproducibility
    random.seed(seed)
    
    # Randomly sample images
    sampled_images = random.sample(train_data['images'], subset_size)
    sampled_image_ids = {img['id'] for img in sampled_images}
    
    print(f"Sampled {len(sampled_images)} images")
    
    # Filter annotations for sampled images
    subset_annotations = [
        ann for ann in train_data['annotations'] 
        if ann['image_id'] in sampled_image_ids
    ]
    
    print(f"Found {len(subset_annotations)} annotations")
    
    # Count class distribution
    class_counts = Counter()
    for ann in subset_annotations:
        class_counts[ann['category_id']] += 1
    
    # Create subset COCO data
    subset_coco = {
        "info": {
            **train_data['info'],
            "description": f"ZOD Test Subset ({subset_size} images)",
            "date_created": "2025-01-28"
        },
        "licenses": train_data['licenses'],
        "images": sampled_images,
        "annotations": subset_annotations,
        "categories": train_data['categories']
    }
    
    # Save subset
    subset_file = output_dir / f"train_subset_{subset_size}.json"
    with open(subset_file, 'w') as f:
        json.dump(subset_coco, f, separators=(',', ':'))
    
    # Also create smaller validation subset (200 images)
    val_subset_size = min(200, subset_size // 5)
    val_file = annotations_dir / "val_coco.json"
    
    with open(val_file, 'r') as f:
        val_data = json.load(f)
    
    sampled_val_images = random.sample(val_data['images'], val_subset_size)
    sampled_val_image_ids = {img['id'] for img in sampled_val_images}
    
    subset_val_annotations = [
        ann for ann in val_data['annotations']
        if ann['image_id'] in sampled_val_image_ids
    ]
    
    subset_val_coco = {
        "info": {
            **val_data['info'],
            "description": f"ZOD Validation Subset ({val_subset_size} images)",
            "date_created": "2025-01-28"
        },
        "licenses": val_data['licenses'],
        "images": sampled_val_images,
        "annotations": subset_val_annotations,
        "categories": val_data['categories']
    }
    
    val_subset_file = output_dir / f"val_subset_{val_subset_size}.json"
    with open(val_subset_file, 'w') as f:
        json.dump(subset_val_coco, f, separators=(',', ':'))
    
    # Print statistics
    print(f"\n=== Test Subset Created ===")
    print(f"Train subset: {len(sampled_images)} images, {len(subset_annotations)} annotations")
    print(f"Val subset: {len(sampled_val_images)} images, {len(subset_val_annotations)} annotations")
    print(f"Files saved to: {output_dir}")
    
    # Class distribution
    print(f"\nClass Distribution (Train Subset):")
    category_names = {cat['id']: cat['name'] for cat in train_data['categories']}
    total_anns = len(subset_annotations)
    
    for cat_id, count in sorted(class_counts.items()):
        class_name = category_names.get(cat_id, f"Class_{cat_id}")
        pct = count / total_anns * 100
        print(f"  {class_name}: {count} ({pct:.1f}%)")
    
    print(f"\n✅ Subset files created:")
    print(f"  - {subset_file}")
    print(f"  - {val_subset_file}")
    print(f"\n💡 Use these files for quick DiffusionDet testing!")
    
    return subset_file, val_subset_file

def register_test_subset():
    """Register the test subset with detectron2"""
    from detectron2.data.datasets import register_coco_instances
    from detectron2.data import MetadataCatalog
    
    # Paths
    zod_base = Path("/mimer/NOBACKUP/groups/naiss2024-5-153/Berhane/zod_processed")
    subset_dir = zod_base / "annotations/test_subset"
    images_dir = zod_base / "images"
    
    # Register subset datasets
    register_coco_instances(
        "zod_train_subset_1000",
        {},
        str(subset_dir / "train_subset_1000.json"),
        str(images_dir)
    )
    
    register_coco_instances(
        "zod_val_subset_200", 
        {},
        str(subset_dir / "val_subset_200.json"),
        str(images_dir)
    )
    
    # Set metadata (same as full dataset)
    zod_metadata = {
        "thing_classes": [
            "Vehicle", "TrafficSign", "PoleObject", "TrafficGuide", "TrafficSignal",
            "Pedestrian", "VulnerableVehicle", "TrafficBeacon", "DynamicBarrier", "Animal"
        ],
        "thing_colors": [
            [255, 69, 58], [255, 149, 0], [158, 158, 158], [255, 235, 59], [255, 193, 7],
            [76, 175, 80], [139, 195, 74], [156, 39, 176], [103, 58, 183], [121, 85, 72]
        ]
    }
    
    MetadataCatalog.get("zod_train_subset_1000").set(**zod_metadata)
    MetadataCatalog.get("zod_val_subset_200").set(**zod_metadata)
    
    print(f"✅ Test subsets registered:")
    print(f"  - 'zod_train_subset_1000' (1000 images)")
    print(f"  - 'zod_val_subset_200' (200 images)")

if __name__ == "__main__":
    # Create subsets
    train_file, val_file = create_test_subset(subset_size=1000)
    
    # Register with detectron2
    print(f"\nRegistering subsets with detectron2...")
    register_test_subset()
    
    print(f"\n🚀 Ready for quick DiffusionDet testing!")
    print(f"   Use: cfg.DATASETS.TRAIN = ('zod_train_subset_1000',)")
    print(f"   Use: cfg.DATASETS.TEST = ('zod_val_subset_200',)")