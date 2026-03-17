"""fl: Configuration utilities for accessing pyproject.toml settings."""

import os
from pathlib import Path
from flwr.common import Context

def get_config_value(context: Context, key: str, default=None):
    return context.run_config.get(key, default)

def get_data_partition_type(context: Context):
    """Get data partition type (iid or non_iid) from TOML config"""
    return get_config_value(context, 'data_partition_type', 'iid').lower()

def get_kitti_client_classes(context: Context):
    """Get KITTI client classes from comma-separated string in TOML"""
    classes_str = get_config_value(context, 'kitti-classes', 
                                  "Car,Van,Truck,Pedestrian,Tram_Sitting,Cyclist")
    
    # Parse comma-separated string into list
    classes = [cls.strip() for cls in classes_str.split(',')]
    
    print(f"[CONFIG] Using client classes: {classes}")
    return classes  

def get_num_iid_clients(context: Context):
    """Get number of IID clients from config"""
    return get_config_value(context, 'num_iid_clients', 6)

# Non-IID dataset name functions
def get_client_train_dataset_name(client_name: str):
    return f"fl_kitti_train_{client_name}"

def get_client_val_dataset_name(client_name: str):
    return f"fl_kitti_val_{client_name}"

# IID dataset name functions  
def get_iid_client_train_dataset_name(client_id: int):
    return f"fl_kitti_iid_train_client_{client_id}"
    

def get_iid_client_val_dataset_name(client_id: int):
    return f"fl_kitti_iid_val_client_{client_id}"
    


def get_global_val_dataset_name():
    return "global_val"

def get_dataset_names_for_partition(context: Context, partition_id: int, num_partitions: int):
    """
    Get appropriate dataset names based on partition type configuration.
    Returns (train_name, val_name, client_identifier)
    """
    partition_type = get_data_partition_type(context)
    
    if partition_type == 'iid':
        # IID: Use client_id directly
        num_clients = get_num_iid_clients(context)
        client_id = partition_id % num_clients
        
        train_name = get_iid_client_train_dataset_name(client_id)
        val_name = get_iid_client_val_dataset_name(client_id)
        client_identifier = f"client_{client_id}"
        
        print(f"[CONFIG] IID mode: partition {partition_id} -> client_id {client_id}")
        
    else:  # non_iid (default)
        # Non-IID: Use class-based assignment
        available_classes = get_kitti_client_classes(context)
        client_class = available_classes[partition_id % len(available_classes)]
        
        train_name = get_client_train_dataset_name(client_class)
        val_name = get_client_val_dataset_name(client_class)
        client_identifier = client_class
        
        print(f"[CONFIG] Non-IID mode: partition {partition_id} -> class {client_class}")
    
    return train_name, val_name, client_identifier