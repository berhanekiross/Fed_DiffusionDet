

#!/usr/bin/env python3
"""
Clean KITTI FL Dataset Registration - Compatible with train_net.py and client.py
Supports both Non-IID and IID partitions
Single registration functions, no duplicates, minimal and clean
"""

import os
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import register_coco_instances
from pycocotools.coco import COCO

# Configuration
DATASET_ROOT = "/mimer/NOBACKUP/groups/naiss2024-5-153/old_projects/Berhane/labelled_kitti/fl_diffusiondet/fl_dataset"
ANNOTATIONS_DIR = os.path.join(DATASET_ROOT, "annotations")
ANNOTATIONS_IID_DIR = os.path.join(DATASET_ROOT, "annotations_iid")
IMAGES_DIR = os.path.join(DATASET_ROOT, "images")

# KITTI classes and colors
CLIENTS = ["Car", "Van", "Truck", "Pedestrian", "Tram_Sitting", "Cyclist"]
TRUE_CLASSES = ["Car", "Van", "Truck", "Pedestrian", "Person_sitting", "Cyclist", "Tram"]
NUM_IID_CLIENTS = 6

KITTI_COLORS = [
    [255, 69, 58],    # Car - Red
    [255, 159, 159],  # Van - Light Red/Pink
    [255, 149, 0],    # Truck - Orange
    [255, 193, 7],    # Pedestrian - Yellow/Orange
    [255, 235, 59],   # Person_sitting - Yellow
    [76, 175, 80],    # Cyclist - Green
    [139, 195, 74]    # Tram - Light Green
]

# Global registry to track what's been registered
_registered_noniid = False
_registered_iid = False

def _clean_existing_datasets(dataset_prefix):
    """Clean existing datasets with given prefix."""
    for name in DatasetCatalog.list():
        if name.startswith(dataset_prefix) or name == "global_val":
            DatasetCatalog.remove(name)
            MetadataCatalog.remove(name)

def register_kitti_splits():
    """
    Register Non-IID FL KITTI datasets once
    Compatible with both train_net.py and client.py
    """
    global _registered_noniid
    if _registered_noniid:
        return

    print("[REGISTER] Registering Non-IID KITTI FL datasets...")
    
    # Clear any existing non-IID registrations
    _clean_existing_datasets("fl_kitti_")
    
    metadata = {
        "thing_classes": TRUE_CLASSES,
        "thing_colors": KITTI_COLORS,
        "num_classes": len(TRUE_CLASSES),
    }

    # Register client-specific datasets
    registered_count = 0
    for client in CLIENTS:
        train_json = os.path.join(ANNOTATIONS_DIR, f"train_{client}.json")
        if os.path.exists(train_json):
            train_name = f"fl_kitti_train_{client}"
            register_coco_instances(train_name, {}, train_json, IMAGES_DIR)
            MetadataCatalog.get(train_name).set(**metadata)
            print(f"✓ Registered: {train_name}")
            registered_count += 1
            
        val_json = os.path.join(ANNOTATIONS_DIR, f"val_{client}.json")
        if os.path.exists(val_json):
            val_name = f"fl_kitti_val_{client}"
            register_coco_instances(val_name, {}, val_json, IMAGES_DIR)
            MetadataCatalog.get(val_name).set(**metadata)
            print(f"✓ Registered: {val_name}")
    
    # Register global validation
    global_val_json = os.path.join(ANNOTATIONS_DIR, "global_val.json")
    if os.path.exists(global_val_json):
        register_coco_instances("global_val", {}, global_val_json, IMAGES_DIR)
        MetadataCatalog.get("global_val").set(**metadata)
        print("✓ Registered: global_val")
    
    _registered_noniid = True
    print(f"[REGISTER] Non-IID registration complete: {registered_count//2} clients")

def register_kitti_iid_splits():
    """
    Register IID FL KITTI datasets
    Creates balanced client partitions for fair comparison
    """
    global _registered_iid
    if _registered_iid:
        print("[REGISTER] IID datasets already registered")
        return

    print("[REGISTER] Registering IID KITTI FL datasets...")
    
    # Check if IID annotations directory exists
    if not os.path.exists(ANNOTATIONS_IID_DIR):
        print(f"[ERROR] IID annotations directory not found: {ANNOTATIONS_IID_DIR}")
        print("[ERROR] Please run create_iid_partitions.py first")
        return
    
    # Clear any existing IID registrations
    _clean_existing_datasets("fl_kitti_iid_")
    
    metadata = {
        "thing_classes": TRUE_CLASSES,
        "thing_colors": KITTI_COLORS,
        "num_classes": len(TRUE_CLASSES),
    }

    # Register IID client datasets
    registered_count = 0
    for client_id in range(NUM_IID_CLIENTS):
        train_json = os.path.join(ANNOTATIONS_IID_DIR, f"train_client_{client_id}.json")
        if os.path.exists(train_json):
            # train_name = f"fl_kitti_train_client_{client_id}"
            train_name = f"fl_kitti_iid_train_client_{client_id}"
            register_coco_instances(train_name, {}, train_json, IMAGES_DIR)
            MetadataCatalog.get(train_name).set(**metadata)
            print(f"✓ Registered: {train_name}")
            registered_count += 1
            
        val_json = os.path.join(ANNOTATIONS_IID_DIR, f"val_client_{client_id}.json")
        if os.path.exists(val_json):
            # val_name = f"fl_kitti_val_client_{client_id}"
            val_name = f"fl_kitti_iid_val_client_{client_id}"
            register_coco_instances(val_name, {}, val_json, IMAGES_DIR)
            MetadataCatalog.get(val_name).set(**metadata)
            print(f"✓ Registered: {val_name}")
    
    # Register IID global validation (reuse from non-IID if exists)
    global_val_json = os.path.join(ANNOTATIONS_IID_DIR, "global_val.json")
    if not os.path.exists(global_val_json):
        # Fall back to non-IID global_val
        global_val_json = os.path.join(ANNOTATIONS_DIR, "global_val.json")
    
    if os.path.exists(global_val_json):
        # Use same name as non-IID for compatibility
        register_coco_instances("global_val", {}, global_val_json, IMAGES_DIR)
        MetadataCatalog.get("global_val").set(**metadata)
        print("✓ Registered: global_val (IID)")
    
    _registered_iid = True
    print(f"[REGISTER] IID registration complete: {registered_count//2} clients")

def reset_registrations():
    """Reset registration flags - useful for testing."""
    global _registered_noniid, _registered_iid
    _registered_noniid = False
    _registered_iid = False
    print("[REGISTER] Registration flags reset")

# Dataset name helper functions for Non-IID
def get_client_train_dataset(client: str) -> str:
    """Get Non-IID training dataset name for a client"""
    return f"fl_kitti_train_{client}"

def get_client_val_dataset(client: str) -> str:
    """Get Non-IID validation dataset name for a client"""
    return f"fl_kitti_val_{client}"

# Dataset name helper functions for IID
def get_iid_client_train_dataset(client_id: int) -> str:
    """Get IID training dataset name for a client"""
    return f"fl_kitti_iid_train_client_{client_id}"

def get_iid_client_val_dataset(client_id: int) -> str:
    """Get IID validation dataset name for a client"""
    return f"fl_kitti_iid_val_client_{client_id}"

def get_global_val_dataset() -> str:
    """Get global validation dataset name (same for both IID and Non-IID)"""
    return "global_val"

def get_available_clients():
    """Get list of available Non-IID clients"""
    return CLIENTS.copy()

def get_num_iid_clients():
    """Get number of IID clients"""
    return NUM_IID_CLIENTS

def list_registered_datasets():
    """List all currently registered FL datasets."""
    all_datasets = DatasetCatalog.list()
    fl_datasets = [name for name in all_datasets if name.startswith("fl_kitti_") or name == "global_val"]
    
    print("\n[REGISTERED DATASETS]")
    print("-" * 40)
    
    # Group by type
    noniid_train = [d for d in fl_datasets if d.startswith("fl_kitti_train_")]
    noniid_val = [d for d in fl_datasets if d.startswith("fl_kitti_val_")]
    iid_train = [d for d in fl_datasets if "iid_train" in d]
    iid_val = [d for d in fl_datasets if "iid_val" in d]
    global_val = [d for d in fl_datasets if d == "global_val"]
    
    if noniid_train:
        print(f"Non-IID Train ({len(noniid_train)}): {noniid_train}")
    if noniid_val:
        print(f"Non-IID Val ({len(noniid_val)}): {noniid_val}")
    if iid_train:
        print(f"IID Train ({len(iid_train)}): {iid_train}")
    if iid_val:
        print(f"IID Val ({len(iid_val)}): {iid_val}")
    if global_val:
        print(f"Global Val: {global_val}")
    
    if not fl_datasets:
        print("No FL datasets registered")
    print("-" * 40)
    
    return fl_datasets

# Auto-register Non-IID when imported (safe with singleton pattern)
try:
    register_kitti_splits()
except Exception as e:
    print(f"Auto Non-IID registration failed: {e}")
    print("Call register_kitti_splits() manually if needed")