#!/usr/bin/env python3
"""
Create client-specific dataset folders with copied images and labels
"""
import os
import shutil
from pathlib import Path

def create_client_dataset(client_name, partition_dir="fl_dataset/partitions", 
                         source_labels="/mimer/NOBACKUP/groups/naiss2024-5-153/old_projects/Berhane/labelled_kitti/training/labels",
                         output_base="client_datasets"):
    """
    Create client dataset with proper YOLO structure
    """
    print(f"Creating dataset for client: {client_name}")
    
    # Create client directory structure
    client_dir = Path(output_base) / client_name
    for subdir in ["images/train", "images/val", "labels/train", "labels/val"]:
        (client_dir / subdir).mkdir(parents=True, exist_ok=True)
    
    # Process train and val splits
    for split in ["train", "val"]:
        partition_file = Path(partition_dir) / f"{split}_{client_name.lower()}.txt"
        
        if not partition_file.exists():
            print(f"Warning: {partition_file} not found, skipping {split}")
            continue
            
        # Read image paths
        with open(partition_file, 'r') as f:
            image_paths = [line.strip() for line in f if line.strip()]
        
        copied_count = 0
        for img_path in image_paths:
            img_path = Path(img_path)
            if not img_path.exists():
                print(f"Warning: Image not found: {img_path}")
                continue
                
            # Copy image
            dest_img = client_dir / "images" / split / img_path.name
            shutil.copy2(img_path, dest_img)
            
            # Copy corresponding label
            label_name = img_path.stem + ".txt"
            src_label = Path(source_labels) / label_name
            dest_label = client_dir / "labels" / split / label_name
            
            if src_label.exists():
                shutil.copy2(src_label, dest_label)
            else:
                print(f"Warning: Label not found: {src_label}")
            
            copied_count += 1
        
        print(f"  {split}: Copied {copied_count} images and labels")
    
    # Create dataset.yaml for this client
    yaml_content = f"""# {client_name} Client Dataset
path: {client_dir.absolute()}
train: images/train
val: images/val

names:
  0: Car
  1: Van
  2: Truck
  3: Pedestrian
  4: Person_sitting
  5: Cyclist
  6: Tram
"""
    
    yaml_file = client_dir / "dataset.yaml"
    with open(yaml_file, 'w') as f:
        f.write(yaml_content)
    
    print(f"  Created: {yaml_file}")
    return client_dir

def main():
    """Create datasets for all clients"""
    clients = ["Car", "Van", "Truck", "Pedestrian", "Tram_Sitting", "Cyclist"]
    
    for client in clients:
        try:
            client_dir = create_client_dataset(client)
            print(f"✓ {client} dataset created at: {client_dir}")
        except Exception as e:
            print(f"✗ Error creating {client} dataset: {e}")
    
    print("\nDone! Update your task.py to use these client datasets:")
    print("config_path = f'client_datasets/{client_class}/dataset.yaml'")

if __name__ == "__main__":
    main()