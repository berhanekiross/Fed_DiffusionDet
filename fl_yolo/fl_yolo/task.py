"""fl: Minimal YOLO task for FL - based on original flwr demo structure"""

import os
import torch
from collections import OrderedDict
from ultralytics import YOLO
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import yaml

class Net:
    def __init__(self, client_name="global"):
        self.model = YOLO("yolov8n.pt")
        if hasattr(self.model.model, 'nc'):
            self.model.model.nc = 7  
        print(f"[YOLO] Model initialized for {client_name}, will reshape on first training")

    
    def state_dict(self):
        """Return model state dict for FL parameter exchange."""
        return self.model.model.state_dict()
    
    def load_state_dict(self, state_dict, strict=True):
        """Load state dict into model."""
        self.model.model.load_state_dict(state_dict, strict=strict)
    
    def parameters(self):
        """Return model parameters."""
        return self.model.model.parameters()
    
    def to(self, device):
        """Move model to device."""
        self.model.model.to(device)
        return self


def get_weights(net):
    """Extract model weights as list of numpy arrays."""
    return [val.cpu().numpy() for _, val in net.state_dict().items()]

def set_weights(net, parameters):
    """Set model weights from list of numpy arrays."""
    # Exit inference mode and clone tensors to avoid in-place update issues
    with torch.no_grad():
        params_dict = net.state_dict()
        
        # Check parameter count
        if len(parameters) != len(params_dict):
            print(f"[WARNING] Parameter count mismatch: expected {len(params_dict)}, got {len(parameters)}")
        
        # Create new state dict with cloned tensors
        new_state_dict = OrderedDict()
        for (key, _), new_value in zip(params_dict.items(), parameters):
            # Clone the tensor to avoid inference mode issues
            new_state_dict[key] = torch.tensor(new_value.copy()).clone()
        
        try:
            print(f"[WEIGHTS] Setting weights for {len(new_state_dict)} parameters")
            # Load the cloned state dict
            net.model.model.load_state_dict(new_state_dict, strict=False)
            
            # Ensure correct number of classes
            if hasattr(net.model.model, 'nc'):
                net.model.model.nc = 7
                
        except RuntimeError as e:
            print(f"[WARNING] Partial weight loading: {e}")
            pass

def load_data(partition_id: int, num_partitions: int):
    """Load client data - comment/uncomment the dataset you want to use."""
    
    # =================================================================
    # 🔄 DATASET SELECTION - Comment/Uncomment to switch
    # =================================================================
    
    # Option 1: IID partitions (balanced data across clients)
    # config_path = f"datasets_iid/client_{partition_id}.yaml"
    # client_name = f"client_{partition_id}"
    # print(f"[DATA] Client {partition_id} loading IID data")
    
    # Option 2: Non-IID partitions (class-specific clients)
    available_classes = ["Car", "Van", "Truck", "Pedestrian", "Tram_Sitting", "Cyclist"]
    client_name = available_classes[partition_id % len(available_classes)]
    config_path = f"datasets/{client_name}_dataset.yaml"
    print(f"[DATA] Client {partition_id} assigned to class: {client_name}")
    
    # =================================================================
    
    if not os.path.exists(config_path):
        print(f"[ERROR] Dataset config not found: {config_path}")
        return None, None, client_name, 0
    
    # Read dataset size from config
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        train_file = config.get('train')
        if train_file and os.path.exists(train_file):
            with open(train_file, 'r') as f:
                dataset_size = len([line for line in f if line.strip()])
        else:
            print(f"[WARNING] Train file not found: {train_file}")
            dataset_size = 100  # Fallback
            
    except Exception as e:
        print(f"[WARNING] Could not read dataset size: {e}")
        dataset_size = 100  # Fallback
    
    print(f"[DATA] {client_name} has {dataset_size} training samples")
    print(f"[DATA] Using config: {config_path}")
    
    return config_path, config_path, client_name, dataset_size

# def load_data(partition_id: int, num_partitions: int):
#     """Load client data - use existing dataset configs."""
#     available_classes = ["Car", "Van", "Truck", "Pedestrian", "Tram_Sitting", "Cyclist"]
#     client_class = available_classes[partition_id % len(available_classes)]
    
#     print(f"[DATA] Client {partition_id} assigned to: {client_class}")
#     config_path = f"datasets/{client_class}_dataset.yaml"
    
#     if not os.path.exists(config_path):
#         print(f"[ERROR] Dataset config not found: {config_path}")
#         return None, None, client_class, 0
    
#     # Read actual dataset size from the partition file
#     try:
#         with open(config_path, 'r') as f:
#             config = yaml.safe_load(f)
        
#         train_file = config.get('train')
#         if train_file and os.path.exists(train_file):
#             with open(train_file, 'r') as f:
#                 dataset_size = len([line for line in f if line.strip()])
#         else:
#             dataset_size = 100  # Fallback
            
#         print(f"[DATA] {client_class} has {dataset_size} training samples")
        
#     except Exception as e:
#         print(f"[DATA] Could not read dataset size: {e}, using default 100")
#         dataset_size = 100
    
#     print(f"[DATA] Using existing config: {config_path}")
#     return config_path, config_path, client_class, dataset_size

def train(net, dataset_config, epochs, device="cuda", **kwargs):
    """Minimal training function."""
    print(f"[TRAIN] Starting training with {dataset_config}")
    client_name = kwargs.get('client_name', 'unknown')
    round_num = kwargs.get('server_round', 1)
    workers = kwargs.get('workers', 4)
    epochs = kwargs.get('local-ephochs', 1)
    
    print(f"[TRAIN] Client {client_name} - Round {round_num}")
    
    # Force explicit data validation
    if not dataset_config or not os.path.exists(dataset_config):
        raise FileNotFoundError(f"Custom dataset required: {dataset_config}")
    
    # Ensure model has correct number of classes before training
    if hasattr(net.model.model, 'nc'):
        net.model.model.nc = 7
    
    try:
        results = net.model.train(
            data=dataset_config,  
            epochs=epochs,
            device=device,
            verbose=True,
            batch=32,
            # lr0=0.001,
            # mosaic=0.8,
            # mixup=0.1, 
            # copy_paste=0.1, 
            # cutmix=0.1,
            save=True,
            plots=True,
            val=True,
            workers=workers,
            cache="disk",
            project="output_yolo_temp",
            name=client_name,
            exist_ok=True,
            resume=False,  
        )
        # results = net.model.train(
        #     data=dataset_config,  # Must use 'data=' parameter
        #     epochs=epochs,
        #     device=device,
        #     verbose=True,
        #     save=True,
        #     plots=True,
        #     val=True,
        #     workers=workers,
        #     cache="disk",
        #     project="output_yolo_temp",
        #     name=client_name,
        #     exist_ok=True,
        #     resume=False,  
        # )
        # Extract simple loss
        train_loss = 0.0
        if hasattr(results, 'results_dict'):
            train_loss = results.results_dict.get('train/box_loss', 0.0)
        
        print(f"[TRAIN] Completed. Loss: {train_loss}")
        return train_loss, {"loss": train_loss}
        
    except Exception as e:
        print(f"[TRAIN] Error: {e}")
        return 0.0, {"error": str(e)}

def test(net, dataset_config, device="cpu", **kwargs):
    """Minimal test function with client-specific output."""
    
    client_name = kwargs.get('client_name', 'unknown')
    round_num = kwargs.get('server_round', 1)
    workers = kwargs.get('workers', 2)
    
    print(f"[TEST] Client {client_name} - Round {round_num}")
    
    try:
        results = net.model.val(
            data=dataset_config,
            device=device,
            verbose=False,
            save=True,  
            plots=True,  
            workers=workers,
            cache="disk",
            project="output_yolo_temp",
            name=client_name,
            exist_ok=True,
        )
        
        # Extract metrics
        val_loss = 0.0
        accuracy = 0.0
        
        if hasattr(results, 'results_dict'):
            val_loss = results.results_dict.get('val/box_loss', 0.0)
            accuracy = results.results_dict.get('metrics/mAP50(B)', 0.0)
        
        metrics = {"val_loss": val_loss, "accuracy": accuracy}
        print(f"[TEST] Client {client_name} - Loss: {val_loss:.4f}, mAP50: {accuracy:.4f}")
        
        return val_loss, metrics, {"summary": metrics}
        
    except Exception as e:
        print(f"[TEST] Error: {e}")
        return 0.0, {"error": str(e)}, {"error": str(e)}
    

