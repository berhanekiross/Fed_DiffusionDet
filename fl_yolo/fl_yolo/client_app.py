"""fl: Minimal YOLO client app - based on original flwr demo structure"""

import torch
import os
from flwr.client import ClientApp, NumPyClient
from ultralytics import YOLO
from flwr.common import Context
from fl_yolo.task import Net, get_weights, load_data, set_weights, test, train

class FlowerClient(NumPyClient):
    def __init__(self, net, trainloader, valloader, client_class, dataset_size):
        self.client_class = client_class
        self.trainloader = trainloader
        self.valloader = valloader
        self.dataset_size = dataset_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net = net
        
        # self.checkpoint_path = checkpoint_path
        print(f"[CLIENT {client_class}] Initialized on {self.device}")

    def fit(self, parameters, config, **kwargs):
        """Train the model with received parameters."""
        round_num = config.get('num-server-rounds', 1) 
        num_epochs = config.get('local_epochs', 1)
        print(f"[CLIENT {self.client_class}] FIT START - Round {round_num}")
        
        try:
            # Set global model parameters
            set_weights(self.net, parameters)
        
            # Train the model
            train_loss, _ = train(
                self.net, 
                self.trainloader,
                epochs=num_epochs,  # Use the value from config
                device=self.device,
                client_name=self.client_class,  
                server_round=round_num
            )
            print(f"[CLIENT {self.client_class}] train COMPLETED ")
            # Get updated model parameters
            updated_weights = get_weights(self.net)
            metrics = {
                "train_loss": float(train_loss),
                "client": self.client_class,
                "round": round_num,
                "local_epochs": num_epochs,
            }
            print(f"[CLIENT {self.client_class}] FIT SUCCESS - Loss: {train_loss}")
            return updated_weights, self.dataset_size, metrics
            
        except Exception as e:
            print(f"[CLIENT {self.client_class}] FIT ERROR: {e}")
            import traceback
            traceback.print_exc()
            raise

    def evaluate(self, parameters, config):
        """Evaluate the model with received parameters."""
        print(f"[CLIENT {self.client_class}] EVALUATE START")
        
        try:
            # Set global model parameters
            set_weights(self.net, parameters)
            
            # Evaluate the model
            loss, metrics, _ = test(
                self.net, 
                self.valloader,
                self.device,
                client_name=self.client_class,  # Add this
                server_round=config.get('server_round', 1)  # Add this
            )
            
            # Simple metrics for flower
            eval_metrics = {
                "val_loss": float(loss),
                "accuracy": float(metrics.get("accuracy", 0.0)),
                "client": self.client_class,
            }
            
            print(f"[CLIENT {self.client_class}] EVALUATE SUCCESS - Loss: {loss}")
            return float(loss), self.dataset_size, eval_metrics
            
        except Exception as e:
            print(f"[CLIENT {self.client_class}] EVALUATE ERROR: {e}")
            # Return dummy values to prevent crash
            return 1.0, self.dataset_size, {"error": str(e)}


# Load model once at module level
# net = Net()
# def client_fn(context: Context):
#     """Create a Flower client instance."""
#     partition_id = context.node_config["partition-id"]
#     num_partitions = context.node_config["num-partitions"]
    
#     # Load data for this client
#     trainloader, valloader, client_class, dataset_size = load_data(partition_id, num_partitions)
#     # net = Net(client_name=client_class)
#     return FlowerClient(net, trainloader, valloader, client_class, dataset_size).to_client()


# # Create ClientApp
# app = ClientApp(client_fn)

# def client_fn(context: Context):
#     """Create a Flower client instance with disk-based persistence."""
#     partition_id = context.node_config["partition-id"]
#     num_partitions = context.node_config["num-partitions"]
#     # local_epochs = context.run_config.get("local-epochs", 1)
    
#     # Load data for this client - this determines the ACTUAL client class
#     trainloader, valloader, client_class, dataset_size = load_data(partition_id, num_partitions)
    
#     # Use the CORRECT client_class for checkpoint path
#     # checkpoint_path = f"output_yolo_temp/{client_class}/fl_checkpoint.pt"
    
#     # Use YOLO's own checkpoint (more reliable)
#     yolo_checkpoint = f"output_yolo_temp/{client_class}/weights/last.pt"
    
#     if os.path.exists(yolo_checkpoint):
#         print(f"[CLIENT {client_class}] Loading YOLO checkpoint from {yolo_checkpoint}")
#         net = Net()
#         # Load the entire YOLO model, not just state dict
#         net.model = YOLO(yolo_checkpoint)
#     else:
#         print(f"[CLIENT {client_class}] No checkpoint found, creating new model")
#         net = Net()
    
#     return FlowerClient(
#         net,
#         trainloader, 
#         valloader, 
#         client_class, 
#         dataset_size,
#         # local_epochs,
#         # checkpoint_path=checkpoint_path  # Still save FL checkpoint for backup
#     ).to_client()
# app = ClientApp(client_fn)


def client_fn(context: Context):
    """Create a Flower client instance using YOLO's checkpointing."""
    
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    
    trainloader, valloader, client_class, dataset_size = load_data(partition_id, num_partitions)
    
    # Only use YOLO's checkpoint
    yolo_checkpoint = f"output_yolo_temp/{client_class}/weights/last.pt"
    
    if os.path.exists(yolo_checkpoint):
        print(f"[CLIENT {client_class}] Loading YOLO checkpoint from {yolo_checkpoint}")
        net = Net()
        net.model = YOLO(yolo_checkpoint)
    else:
        print(f"[CLIENT {client_class}] No checkpoint found, creating new model")
        net = Net()
    
    return FlowerClient(
        net,
        trainloader, 
        valloader, 
        client_class, 
        dataset_size
        # No checkpoint_path needed
    ).to_client()
app = ClientApp(client_fn)