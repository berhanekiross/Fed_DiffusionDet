#!/usr/bin/env python3
"""Convert KITTI labels to YOLO format for FL training."""

import os
import glob
from pathlib import Path


def kitti_to_yolo(kitti_label_path, image_width=1242, image_height=375):
    """Convert single KITTI label file to YOLO format."""
    
    # KITTI class mapping (same as your FL setup)
    class_mapping = {
        'Car': 0, 'Van': 1, 'Truck': 2, 'Pedestrian': 3,
        'Person_sitting': 4, 'Cyclist': 5, 'Tram': 6
    }
    
    yolo_lines = []
    
    try:
        with open(kitti_label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 15:  # KITTI format has 15 fields
                    continue
                
                class_name = parts[0]
                
                # Skip unknown classes
                if class_name not in class_mapping:
                    continue
                
                class_id = class_mapping[class_name]
                
                # KITTI bounding box: left, top, right, bottom (pixels)
                left = float(parts[4])
                top = float(parts[5])
                right = float(parts[6])
                bottom = float(parts[7])
                
                # Convert to YOLO format (normalized center x, y, width, height)
                x_center = (left + right) / 2.0 / image_width
                y_center = (top + bottom) / 2.0 / image_height
                width = (right - left) / image_width
                height = (bottom - top) / image_height
                
                # YOLO format: class_id x_center y_center width height
                yolo_line = f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
                yolo_lines.append(yolo_line)
    
    except Exception as e:
        print(f"Error processing {kitti_label_path}: {e}")
    
    return yolo_lines


def convert_all_labels():
    """Convert all KITTI labels to YOLO format."""
    
    # Paths
    kitti_label_dir = "/mimer/NOBACKUP/groups/naiss2024-5-153/old_projects/Berhane/labelled_kitti/training/label_2"
    yolo_label_dir = "/mimer/NOBACKUP/groups/naiss2024-5-153/old_projects/Berhane/labelled_kitti/training/labels_yolo"
    
    # Create output directory
    os.makedirs(yolo_label_dir, exist_ok=True)
    
    # Get all KITTI label files
    kitti_files = glob.glob(os.path.join(kitti_label_dir, "*.txt"))
    
    print(f"Converting {len(kitti_files)} KITTI labels to YOLO format...")
    
    converted_count = 0
    
    for kitti_file in kitti_files:
        filename = os.path.basename(kitti_file)
        yolo_file = os.path.join(yolo_label_dir, filename)
        
        # Convert this file
        yolo_lines = kitti_to_yolo(kitti_file)
        
        # Save YOLO format
        with open(yolo_file, 'w') as f:
            f.write('\n'.join(yolo_lines))
        
        converted_count += 1
        
        if converted_count % 1000 == 0:
            print(f"Converted {converted_count}/{len(kitti_files)} files...")
    
    print(f"✅ Conversion complete! {converted_count} files converted.")
    print(f"YOLO labels saved to: {yolo_label_dir}")
    
    # Show sample
    sample_file = os.path.join(yolo_label_dir, "000001.txt")
    if os.path.exists(sample_file):
        print(f"\nSample YOLO label (000001.txt):")
        with open(sample_file, 'r') as f:
            print(f.read()[:200] + "...")


def update_dataset_configs():
    """Update your dataset configs to point to YOLO labels."""
    
    print("\n📝 Update your dataset configs:")
    print("In your yolo_datasets/*.yaml files, change:")
    print("  path: /mimer/NOBACKUP/groups/naiss2024-5-153/old_projects/Berhane/labelled_kitti/training")
    print("  # This will use:")
    print("  # - images from: training/image_2/")
    print("  # - labels from: training/labels_yolo/")


if __name__ == "__main__":
    convert_all_labels()
    update_dataset_configs()