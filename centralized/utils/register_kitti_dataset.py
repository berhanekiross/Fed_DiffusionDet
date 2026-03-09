"""
KITTI Dataset Registration for DiffusionDet
This script registers the KITTI dataset in COCO format with Detectron2
"""

import os
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import register_coco_instances

def register_kitti_dataset():
    """
    Register KITTI dataset in COCO format for DiffusionDet training
    """
    
    # Define your dataset paths
    DATASET_ROOT = "/mimer/NOBACKUP/groups/naiss2024-5-153/old_projects/Berhane/labelled_kitti/centralized/datasets/kitti_coco/"
    
    # Register training dataset
    register_coco_instances(
        "kitti_train",  # Dataset name (must match config file)
        {},  # Metadata (will be set below)
        os.path.join(DATASET_ROOT, "annotations/train2017_no_misc.json"),  # Annotation file path
        os.path.join(DATASET_ROOT, "images/train2017")  # Images directory path
    )
    
    # Register validation dataset (you can create a validation split or use the same for now)
    register_coco_instances(
        "kitti_val",   # Dataset name (must match config file)
        {},  # Metadata (will be set below)  
        os.path.join(DATASET_ROOT, "annotations/train2017_no_misc.json"),  # For now, use same as train
        os.path.join(DATASET_ROOT, "images/train2017")  # For now, use same as train
    )
    
    # Set metadata for KITTI dataset (7 classes without Misc/DontCare)
    kitti_metadata = {
        "thing_classes": [
            "Car",           # ID: 0
            "Van",           # ID: 1  
            "Truck",         # ID: 2
            "Pedestrian",    # ID: 3
            "Person_sitting",# ID: 4
            "Cyclist",       # ID: 5
            "Tram"           # ID: 6
        ],
        "thing_colors": [
            [220, 20, 60],    # Car - Red
            [119, 11, 32],    # Van - Dark Red
            [0, 0, 142],      # Truck - Blue
            [0, 0, 230],      # Pedestrian - Bright Blue
            [106, 0, 228],    # Person_sitting - Purple
            [0, 60, 100],     # Cyclist - Dark Blue
            [0, 80, 100]      # Tram - Darker Blue
        ]
    }
    
    # Apply metadata to both datasets
    MetadataCatalog.get("kitti_train").set(**kitti_metadata)
    MetadataCatalog.get("kitti_val").set(**kitti_metadata)
    
    print("KITTI dataset registered successfully!")
    print(f"   - Training dataset: kitti_train")
    print(f"   - Validation dataset: kitti_val") 
    print(f"   - Number of classes: {len(kitti_metadata['thing_classes'])}")
    print(f"   - Classes: {', '.join(kitti_metadata['thing_classes'])}")
    
    return kitti_metadata

def verify_dataset_registration():
    """
    Verify that the dataset was registered correctly
    """
    try:
        # Check if datasets are registered
        train_dataset = DatasetCatalog.get("kitti_train")
        val_dataset = DatasetCatalog.get("kitti_val")
        
        # Get metadata
        train_metadata = MetadataCatalog.get("kitti_train")
        val_metadata = MetadataCatalog.get("kitti_val")
        
        print(f"\n Dataset Verification:")
        print(f"   - Training samples: {len(train_dataset)}")
        print(f"   - Validation samples: {len(val_dataset)}")
        print(f"   - Classes: {train_metadata.thing_classes}")
        
        # Show a sample annotation
        if len(train_dataset) > 0:
            sample = train_dataset[0]
            print(f"\nSample annotation:")
            print(f"   - Image ID: {sample.get('image_id', 'N/A')}")
            print(f"   - File name: {sample.get('file_name', 'N/A')}")
            print(f"   - Number of annotations: {len(sample.get('annotations', []))}")
            
            if sample.get('annotations'):
                ann = sample['annotations'][0]
                category_id = ann.get('category_id', 0)
                class_name = train_metadata.thing_classes[category_id] if category_id < len(train_metadata.thing_classes) else "Unknown"
                print(f"   - First annotation class: {class_name} (ID: {category_id})")
                print(f"   - First annotation bbox: {ann.get('bbox', 'N/A')}")
        
        return True
        
    except Exception as e:
        print(f"Dataset verification failed: {str(e)}")
        return False

def create_train_val_split(split_ratio=0.8):
    """
    Optional: Create a proper train/validation split from your dataset
    """
    import json
    import random
    
    # Load the original annotation file
    annotation_file = "/mimer/NOBACKUP/groups/naiss2024-5-153/old_projects/Berhane/labelled_kitti/centralized/datasets/kitti_coco/annotations/train2017_no_misc.json"
    
    with open(annotation_file, 'r') as f:
        data = json.load(f)
    
    # Get all image IDs
    image_ids = [img['id'] for img in data['images']]
    random.shuffle(image_ids)
    
    # Split images
    split_point = int(len(image_ids) * split_ratio)
    train_image_ids = set(image_ids[:split_point])
    val_image_ids = set(image_ids[split_point:])
    
    # Create training split
    train_data = {
        'info': data['info'].copy(),
        'licenses': data['licenses'].copy(),
        'categories': data['categories'].copy(),
        'images': [img for img in data['images'] if img['id'] in train_image_ids],
        'annotations': [ann for ann in data['annotations'] if ann['image_id'] in train_image_ids]
    }
    
    # Create validation split  
    val_data = {
        'info': data['info'].copy(),
        'licenses': data['licenses'].copy(),
        'categories': data['categories'].copy(),
        'images': [img for img in data['images'] if img['id'] in val_image_ids],
        'annotations': [ann for ann in data['annotations'] if ann['image_id'] in val_image_ids]
    }
    
    # Save splits
    train_file = annotation_file.replace('.json', '_train.json')
    val_file = annotation_file.replace('.json', '_val.json')
    
    with open(train_file, 'w') as f:
        json.dump(train_data, f, indent=2)
    
    with open(val_file, 'w') as f:
        json.dump(val_data, f, indent=2)
    
    print(f"Created train/val splits:")
    print(f"   - Training: {len(train_data['images'])} images, {len(train_data['annotations'])} annotations")
    print(f"   - Validation: {len(val_data['images'])} images, {len(val_data['annotations'])} annotations")
    print(f"   - Files saved: {train_file}, {val_file}")
    
    return train_file, val_file

if __name__ == "__main__":
    # Register the dataset
    metadata = register_kitti_dataset()
    
    # Verify registration
    if verify_dataset_registration():
        print("\n Dataset registration completed successfully!")

    else:
        print("\nDataset registration failed. Please check file paths and JSON format.")
        
    # Optionally create train/validation split
    create_split = input("\nDo you want to create a train/validation split? (y/n): ").lower().strip()
    if create_split == 'y':
        try:
            train_file, val_file = create_train_val_split()
            print(f"\n Train/val split created successfully!")
            print(f" Update your config file to use:")
            print(f"   - Training: {os.path.basename(train_file)}")
            print(f"   - Validation: {os.path.basename(val_file)}")
        except Exception as e:
            print(f" Failed to create train/val split: {str(e)}")
