import os
import yaml
from pathlib import Path

def create_client_yaml_files():
    # Base directories
    partitions_dir = "/mimer/NOBACKUP/groups/naiss2024-5-153/old_projects/Berhane/labelled_kitti/fl_yolo/fl_dataset/partitions_iid"
    training_base = "/mimer/NOBACKUP/groups/naiss2024-5-153/old_projects/Berhane/labelled_kitti/training"
    output_dir = "/mimer/NOBACKUP/groups/naiss2024-5-153/old_projects/Berhane/labelled_kitti/fl_yolo/datasets_iid"
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Class names
    class_names = {
        0: "Car",
        1: "Van", 
        2: "Truck",
        3: "Pedestrian",
        4: "Person_sitting",
        5: "Cyclist",
        6: "Tram"
    }
    
    # Find all client files
    client_files = {}
    for filename in os.listdir(partitions_dir):
        if filename.startswith('train_client_'):
            client_id = filename.split('_')[-1].split('.')[0]
            client_files[client_id] = {
                'train': filename,
                'val': f"val_client_{client_id}.txt"
            }
    
    # Create YAML file for each client
    for client_id, files in client_files.items():
        yaml_content = {
            'names': class_names,
            'train': os.path.join(partitions_dir, files['train']),
            'val': os.path.join(partitions_dir, files['val']),
            'path': training_base,
            'nc': 7,
            'task': 'detect'
        }
        
        yaml_filename = os.path.join(output_dir, f"client_{client_id}.yaml")
        
        with open(yaml_filename, 'w') as f:
            yaml.dump(yaml_content, f, default_flow_style=False, sort_keys=False)
        
        print(f"Created YAML config for client_{client_id}: {yaml_filename}")
    
    # Also create a global config that references all clients
    create_global_yaml(partitions_dir, training_base, class_names)

def create_global_yaml(partitions_dir, training_base, class_names):
    """Create a global YAML file that can be used for centralized training"""
    global_yaml = {
        'names': class_names,
        'train': os.path.join(partitions_dir, "global_train.txt"),
        'val': os.path.join(partitions_dir, "global_val.txt"),
        'path': training_base,
        'nc': 7,
        'task': 'detect'
    }
    
    # Check if global files exist, if not create them by combining all clients
    global_train_file = os.path.join(partitions_dir, "global_train.txt")
    global_val_file = os.path.join(partitions_dir, "global_val.txt")
    
    if not os.path.exists(global_train_file):
        print("Creating global train file...")
        with open(global_train_file, 'w') as global_train:
            for filename in os.listdir(partitions_dir):
                if filename.startswith('train_client_'):
                    with open(os.path.join(partitions_dir, filename), 'r') as client_file:
                        global_train.write(client_file.read())
    
    if not os.path.exists(global_val_file):
        print("Creating global val file...")
        with open(global_val_file, 'w') as global_val:
            for filename in os.listdir(partitions_dir):
                if filename.startswith('val_client_'):
                    with open(os.path.join(partitions_dir, filename), 'r') as client_file:
                        global_val.write(client_file.read())
    
    global_yaml_filename = os.path.join(partitions_dir, "global.yaml")
    with open(global_yaml_filename, 'w') as f:
        yaml.dump(global_yaml, f, default_flow_style=False, sort_keys=False)
    
    print(f"Created global YAML config: {global_yaml_filename}")

def verify_yaml_files():
    """Verify that the created YAML files are valid"""
    partitions_dir = "/mimer/NOBACKUP/groups/naiss2024-5-153/old_projects/Berhane/labelled_kitti/fl_yolo/datasets_iid"
    
    print("\nVerifying YAML files...")
    
    for filename in os.listdir(partitions_dir):
        if filename.endswith('.yaml'):
            yaml_file = os.path.join(partitions_dir, filename)
            
            try:
                with open(yaml_file, 'r') as f:
                    config = yaml.safe_load(f)
                
                # Check required fields
                required_fields = ['names', 'train', 'val', 'path', 'nc', 'task']
                missing_fields = [field for field in required_fields if field not in config]
                
                if missing_fields:
                    print(f"❌ {filename}: Missing fields {missing_fields}")
                else:
                    # Check if train/val files exist
                    train_exists = os.path.exists(config['train'])
                    val_exists = os.path.exists(config['val'])
                    
                    if train_exists and val_exists:
                        print(f"✅ {filename}: Valid")
                    else:
                        print(f"❌ {filename}: Train exists: {train_exists}, Val exists: {val_exists}")
                        
            except Exception as e:
                print(f"❌ {filename}: Error loading YAML - {e}")

if __name__ == "__main__":
    print("Creating YAML configuration files for each client...")
    create_client_yaml_files()
    verify_yaml_files()
    print("Done!")