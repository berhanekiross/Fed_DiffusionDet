"""
Register KITTI Train/Validation splits for DiffusionDet
Run this after creating the train/val split
"""

import os
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import register_coco_instances

def register_kitti_splits():
    """
    Register KITTI train/validation splits
    """
    
    # Define paths for splits
    DATASET_ROOT = "/mimer/NOBACKUP/groups/naiss2024-5-153/old_projects/Berhane/labelled_kitti/centralized/datasets/kitti_coco"
    # IMAGES_PATH = "/mimer/NOBACKUP/groups/naiss2024-5-153/old_projects/Berhane/labelled_kitti/training/image_2"
    IMAGES_PATH = "/mimer/NOBACKUP/groups/naiss2024-5-153/old_projects/Berhane/labelled_kitti/centralized/datasets/kitti_coco/images/train2017/"
    
    # Register training split
    register_coco_instances(
        "kitti_train_split",
        {},
        os.path.join(DATASET_ROOT, "annotations/train2017_no_misc_train_fixed_dimensions.json"),
        IMAGES_PATH
    )
    
    # Register validation split  
    register_coco_instances(
        "kitti_val_split",
        {},
        os.path.join(DATASET_ROOT, "annotations/train2017_no_misc_val_fixed_dimensions.json"),
        IMAGES_PATH
    )
    
    # Set metadata for both datasets
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
            [255, 69, 58],    # Car - Red (matching YOLO)
            [255, 159, 159],  # Van - Light Red/Pink 
            [255, 149, 0],    # Truck - Orange 
            [255, 193, 7],    # Pedestrian - Yellow/Orange 
            [255, 235, 59],   # Person_sitting - Yellow 
            [76, 175, 80],    # Cyclist - Green 
            [139, 195, 74],   # Tram - Light Green 
            [158, 158, 158]   # Misc - Gray 
        ]
    }
    
    # Apply metadata
    MetadataCatalog.get("kitti_train_split").set(**kitti_metadata)
    MetadataCatalog.get("kitti_val_split").set(**kitti_metadata)
    
    print("KITTI train/val splits registered successfully!")
    
    # Verify registration
    train_dataset = DatasetCatalog.get("kitti_train_split")
    val_dataset = DatasetCatalog.get("kitti_val_split")
    
    print(f"Split Statistics:")
    print(f"   - Training samples: {len(train_dataset)}")
    print(f"   - Validation samples: {len(val_dataset)}")
    print(f"   - Total samples: {len(train_dataset) + len(val_dataset)}")
    print(f"   - Train/Val ratio: {len(train_dataset)/(len(train_dataset) + len(val_dataset)):.1%}/{len(val_dataset)/(len(train_dataset) + len(val_dataset)):.1%}")

if __name__ == "__main__":
    register_kitti_splits()
