import os
import shutil
from pathlib import Path

def create_iid_dataset():
    # Paths
    source_partitions = "/mimer/NOBACKUP/groups/naiss2024-5-153/old_projects/Berhane/labelled_kitti/fl_app/fl_dataset/partitions_iid"
    target_partitions = "/mimer/NOBACKUP/groups/naiss2024-5-153/old_projects/Berhane/labelled_kitti/fl_yolo/fl_dataset/partitions_iid"
    image_source_dir = "/mimer/NOBACKUP/groups/naiss2024-5-153/old_projects/Berhane/labelled_kitti/training/images"
    
    # Create target directory
    os.makedirs(target_partitions, exist_ok=True)
    
    # Process all files in the source partitions directory
    for filename in os.listdir(source_partitions):
        if filename.endswith('.txt'):
            source_file = os.path.join(source_partitions, filename)
            target_file = os.path.join(target_partitions, filename)
            
            print(f"Processing {filename}...")
            
            with open(source_file, 'r') as f_in, open(target_file, 'w') as f_out:
                for line in f_in:
                    line = line.strip()
                    if line:
                        # Remove the client_*/ prefix and add the full image path
                        if '/' in line:
                            # Extract just the filename
                            image_name = line.split('/')[-1]
                            # Create full path
                            full_path = os.path.join(image_source_dir, image_name)
                            f_out.write(full_path + '\n')
                        else:
                            # If no client prefix, assume it's already a proper path
                            f_out.write(line + '\n')
            
            print(f"  -> Created {target_file} with corrected paths")

def verify_dataset():
    """Verify that the created dataset has correct paths"""
    target_partitions = "/mimer/NOBACKUP/groups/naiss2024-5-153/old_projects/Berhane/labelled_kitti/fl_yolo/fl_dataset/partitions_iid"
    
    print("\nVerifying created files...")
    
    for filename in os.listdir(target_partitions):
        if filename.endswith('.txt'):
            file_path = os.path.join(target_partitions, filename)
            
            with open(file_path, 'r') as f:
                lines = f.readlines()
                if lines:
                    first_line = lines[0].strip()
                    print(f"{filename}: {first_line}")
                    
                    # Check if file exists
                    if os.path.exists(first_line):
                        print(f"  ✓ First image exists")
                    else:
                        print(f"  ✗ First image missing: {first_line}")

if __name__ == "__main__":
    print("Creating IID dataset partitions with corrected paths...")
    create_iid_dataset()
    verify_dataset()
    print("Done!")