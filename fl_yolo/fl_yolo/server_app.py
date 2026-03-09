"""fl: Minimal YOLO server app - based on original flwr demo structure"""

import os
import torch
from ultralytics import YOLO
from flwr.common import Context, ndarrays_to_parameters
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.strategy import FedAvg
from fl_yolo.task import Net, get_weights, set_weights
from flwr.common import Context, ndarrays_to_parameters, parameters_to_ndarrays

# # working initial script
# class MinimalFedAvg(FedAvg):
#     """Minimal FedAvg strategy with basic logging."""
    
#     def __init__(self, output_base="./output_yolo_temp", local_epochs=1, **kwargs):
#         super().__init__(**kwargs)
#         self.output_base = output_base
#         self.local_epochs = local_epochs
#         print("[SERVER] Using minimal FedAvg strategy")
#     # def __init__(self, context, output_base="./output_yolo_temp", **kwargs):
#     #     super().__init__(**kwargs)
#     #     self.context = context
#     #     self.output_base = output_base
#     #     print("[SERVER] Using minimal FedAvg strategy")
        
#     def aggregate_fit(self, server_round, results, failures):
#         # Standard aggregation
#         aggregated_parameters, aggregated_metrics = super().aggregate_fit(
#             server_round, results, failures
#         )
        
#         if aggregated_parameters is not None:
#             # Save aggregated model to disk
#             self.save_global_model(aggregated_parameters, server_round)
        
#         return aggregated_parameters, aggregated_metrics
    
#     def configure_fit(self, server_round, parameters, client_manager):
#         """Configure the fit process for each client."""
#         # Get the default client instructions
#         client_instructions = super().configure_fit(server_round, parameters, client_manager)
        
#         # Add local_epochs to each client's configuration
#         local_epochs = self.context.run_config.get("local-epochs", 1)
#         for i, (client_proxy, fit_ins) in enumerate(client_instructions):
#             fit_ins.config["local_epochs"] = local_epochs
#             print(f"[SERVER] Client {i} configured with {local_epochs} local epochs")
        
#         return client_instructions
    
#     def save_global_model(self, parameters, round_num):
#         """Save global aggregated model."""
#         try:
#             # Create global model directory
#             global_dir = os.path.join(self.output_base, "global", "weights")
#             os.makedirs(global_dir, exist_ok=True)
#             global_net = Net(client_name="global")
#             set_weights(global_net, parameters_to_ndarrays(parameters))
#             checkpoint_path = os.path.join(global_dir, f"round_{round_num}.pt")
#             latest_path = os.path.join(global_dir, "latest.pt")
            
#             # Save the model state dict
#             torch.save(global_net.model.model.state_dict(), checkpoint_path)
#             torch.save(global_net.model.model.state_dict(), latest_path)
            
#             print(f"[SERVER] Saved global model for round {round_num} to {checkpoint_path}")
            
#         except Exception as e:
#             print(f"[SERVER] Could not save global model: {e}")
#             import traceback
#             traceback.print_exc()

#     def aggregate_evaluate(self, server_round, results, failures):
#         """Aggregate evaluation results with basic logging."""
#         aggregated_loss, aggregated_metrics = super().aggregate_evaluate(
#             server_round, results, failures
#         )
        
#         if aggregated_loss is not None:
#             print(f"[SERVER] Round {server_round}: Average validation loss = {aggregated_loss:.4f}")
        
#         return aggregated_loss, aggregated_metrics


# def server_fn(context: Context):
#     """Create server components with minimal configuration."""

#     num_rounds = context.run_config.get("num-server-rounds", 3)
#     fraction_fit = context.run_config.get("fraction-fit", 1)
#     min_available_clients = context.run_config.get("min-available-clients", 6)
#     fraction_evaluate = context.run_config.get("fraction-evaluate", 1.0)
#     output_base = context.run_config.get("output_base", "./output_yolo_temp")
#     local_epochs = context.run_config.get("local-epochs", 3)
    
#     # Initialize model
#     net = Net(client_name="global")
#     initial_parameters = ndarrays_to_parameters(get_weights(net))
    
#     print(f"[SERVER] Model initialized with {len(get_weights(net))} parameter arrays")
#     print(f"[SERVER] Global models will be saved to {output_base}/global/")
    
#     strategy = MinimalFedAvg(
#         context=context,
#         output_base=output_base,  
#         fraction_fit=fraction_fit, 
#         fraction_evaluate=fraction_evaluate, 
#         min_available_clients=min_available_clients,
#         initial_parameters=initial_parameters,
#         local_epochs=local_epochs,
#     )    

#     # strategy = MinimalFedAvg(
#     #     context=context,  
#     #     output_base=output_base,  
#     #     fraction_fit=fraction_fit, 
#     #     fraction_evaluate=fraction_evaluate, 
#     #     min_available_clients=min_available_clients,
#     #     initial_parameters=initial_parameters,
#     # )

#     config = ServerConfig(num_rounds=num_rounds)
    
#     return ServerAppComponents(strategy=strategy, config=config)
# app = ServerApp(server_fn=server_fn)


class MinimalFedAvg(FedAvg):
    """Minimal FedAvg strategy with basic logging."""
    
    def __init__(self, output_base="./output_yolo_temp", **kwargs):
        super().__init__(**kwargs)
        self.output_base = output_base
        print("[SERVER] Using minimal FedAvg strategy")

    def aggregate_fit(self, server_round, results, failures):
        # Standard aggregation
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(
            server_round, results, failures
        )

        if aggregated_parameters is not None:
            self.save_global_model(aggregated_parameters, server_round)
        
        return aggregated_parameters, aggregated_metrics
        
    # def save_global_model(self, parameters, round_num):
    #     """Save global aggregated model."""
    #     try:
    #         # Create global model directory
    #         global_dir = os.path.join(self.output_base, "global", "weights")
    #         os.makedirs(global_dir, exist_ok=True)
            
    #         # Convert parameters to check structure
    #         param_arrays = parameters_to_ndarrays(parameters)
    #         num_params = len(param_arrays)
            
    #         # Determine if we need a 7-class or 80-class model based on param count
    #         # 355 params = trained 7-class model, 127 params = fresh 80-class model
    #         if num_params > 300:  # This is a trained model with 7 classes
    #             # Try to load from any existing client checkpoint to get correct structure
    #             client_checkpoints = [
    #                 "output_yolo_temp/Car/weights/last.pt",
    #                 "output_yolo_temp/Van/weights/last.pt",
    #                 "output_yolo_temp/Truck/weights/last.pt",
    #                 "output_yolo_temp/Pedestrian/weights/last.pt",
    #                 "output_yolo_temp/Cyclist/weights/last.pt",
    #                 "output_yolo_temp/Tram_Sitting/weights/last.pt"
    #             ]
                
    #             global_net = None
    #             for checkpoint in client_checkpoints:
    #                 if os.path.exists(checkpoint):
    #                     try:
    #                         print(f"[SERVER] Loading model structure from {checkpoint}")
    #                         global_net = Net(client_name="global")
    #                         global_net.model = YOLO(checkpoint)
    #                         # Verify it has the right number of parameters
    #                         if len(get_weights(global_net)) == num_params:
    #                             break
    #                     except:
    #                         continue
                
    #             if global_net is None:
    #                 print(f"[SERVER] Warning: Could not find matching model structure for {num_params} params")
    #                 return
    #         else:
    #             # Fresh model structure
    #             global_net = Net(client_name="global")
            
    #         # Set the aggregated weights
    #         try:
    #             set_weights(global_net, param_arrays)
    #         except Exception as e:
    #             print(f"[SERVER] Warning during weight setting: {e}")
    #             # Don't continue if weights don't match - it won't work
    #             return
            
    #         # Save the model
    #         checkpoint_path = os.path.join(global_dir, f"round_{round_num}.pt")
    #         latest_path = os.path.join(global_dir, "latest.pt")
            
    #         torch.save(global_net.model.model.state_dict(), checkpoint_path)
    #         torch.save(global_net.model.model.state_dict(), latest_path)
            
    #         print(f"[SERVER] Saved global model for round {round_num} to {checkpoint_path}")
            
    #     except Exception as e:
    #         print(f"[SERVER] Could not save global model: {e}")
    #         import traceback
    #         traceback.print_exc()
    
    def save_global_model(self, parameters, round_num):
        """Save global aggregated model as complete YOLO checkpoint."""
        try:
            # Create global model directory
            global_dir = os.path.join(self.output_base, "global", "weights")
            os.makedirs(global_dir, exist_ok=True)
            
            # Convert parameters to check structure
            param_arrays = parameters_to_ndarrays(parameters)
            num_params = len(param_arrays)
            
            # Load a reference model to get the proper checkpoint structure
            reference_model_path = None
            
            if num_params > 300:  # This is a trained model with 7 classes
                # Try to find an existing client checkpoint
                client_checkpoints = [
                    "output_yolo_temp/Car/weights/last.pt",
                    "output_yolo_temp/Van/weights/last.pt", 
                    "output_yolo_temp/Truck/weights/last.pt",
                    "output_yolo_temp/Pedestrian/weights/last.pt",
                    "output_yolo_temp/Cyclist/weights/last.pt",
                    "output_yolo_temp/Tram_Sitting/weights/last.pt"
                ]
                # iid clients
                # client_checkpoints = [
                #     "output_yolo_temp/client_0/weights/last.pt",
                #     "output_yolo_temp/client_1/weights/last.pt", 
                #     "output_yolo_temp/client_2/weights/last.pt",
                #     "output_yolo_temp/client_3/weights/last.pt",
                #     "output_yolo_temp/client_4/weights/last.pt",
                #     "output_yolo_temp/client_5/weights/last.pt"
                # ]
                
                for checkpoint in client_checkpoints:
                    if os.path.exists(checkpoint):
                        reference_model_path = checkpoint
                        break
                        
                if reference_model_path is None:
                    print(f"[SERVER] Warning: No reference model found for {num_params} params")
                    return
            else:
                # Use base YOLOv8n as reference
                reference_model_path = "yolov8n.pt"
            
            # Load the reference model to get checkpoint structure
            print(f"[SERVER] Loading reference model from {reference_model_path}")
            reference_model = Net(client_name="global")
            reference_model.model = YOLO(reference_model_path)
            
            # Set the aggregated weights
            try:
                set_weights(reference_model, param_arrays)
            except Exception as e:
                print(f"[SERVER] Warning during weight setting: {e}")
                return
            
            # Create proper checkpoint paths
            checkpoint_path = os.path.join(global_dir, f"round_{round_num}.pt")
            latest_path = os.path.join(global_dir, "latest.pt")
            
            # Save complete YOLO checkpoint (not just state_dict)
            reference_model.model.save(checkpoint_path)
            reference_model.model.save(latest_path)
            
            print(f"[SERVER] Saved complete YOLO checkpoint for round {round_num} to {checkpoint_path}")
            
        except Exception as e:
            print(f"[SERVER] Could not save global model: {e}")
            import traceback
            traceback.print_exc()

    def aggregate_evaluate(self, server_round, results, failures):
        """Aggregate evaluation results with basic logging."""
        aggregated_loss, aggregated_metrics = super().aggregate_evaluate(
            server_round, results, failures
        )
        
        if aggregated_loss is not None:
            print(f"[SERVER] Round {server_round}: Average validation loss = {aggregated_loss:.4f}")
        
        return aggregated_loss, aggregated_metrics


def server_fn(context: Context):
    """Create server components with minimal configuration."""
    
    # Read basic configuration
    num_rounds = context.run_config.get("num-server-rounds", 3)
    fraction_fit = context.run_config.get("fraction-fit", 1)
    fraction_evaluate = context.run_config.get("fraction-evaluate", 1.0)
    output_base = context.run_config.get("output_base", "./output_yolo_temp")
    
    # Initialize model
    net = Net(client_name="global")
    initial_parameters = ndarrays_to_parameters(get_weights(net))
    
    print(f"[SERVER] Model initialized with {len(get_weights(net))} parameter arrays")
    print(f"[SERVER] Global models will be saved to {output_base}/global/")
    
    strategy = MinimalFedAvg(
        output_base=output_base,  
        fraction_fit=fraction_fit, 
        fraction_evaluate=fraction_evaluate, 
        initial_parameters=initial_parameters,
    )    

    config = ServerConfig(num_rounds=num_rounds)
    
    return ServerAppComponents(strategy=strategy, config=config)
app = ServerApp(server_fn=server_fn)