

# """fl: A Flower / PyTorch server app with periodic model saving using detectron2."""

# import torch
# import os
# import atexit
# # torch.cuda.set_per_process_memory_fraction(0.125)
# from detectron2.checkpoint import DetectionCheckpointer
# from flwr.common import Context, ndarrays_to_parameters, parameters_to_ndarrays
# from flwr.server import ServerApp, ServerAppComponents, ServerConfig
# from flwr.server.strategy import FedAvg
# from fl.task import Net, get_weights


# class FedAvgWithPeriodicSave(FedAvg):
#     """FedAvg strategy that saves model every 5th round using detectron2 format."""
    
#     def __init__(self, save_dir="./saved_models", save_every=10, **kwargs):
#         super().__init__(**kwargs)
#         self.save_dir = save_dir
#         self.save_every = save_every
#         self.final_parameters = None
#         self.current_round = 0
#         os.makedirs(save_dir, exist_ok=True)
    
#     def aggregate_fit(self, server_round, results, failures):
#         """Aggregate client results and save model periodically."""
        
#         # Perform standard aggregation
#         aggregated_parameters, aggregated_metrics = super().aggregate_fit(
#             server_round, results, failures
#         )
        
#         # Always update final parameters (overwrites previous)
#         if aggregated_parameters is not None:
#             self.final_parameters = aggregated_parameters
#             self.current_round = server_round
#             print(f"[SERVER] Round {server_round} completed, parameters updated")
            
#             # Save every N rounds and always save latest
#             if server_round % self.save_every == 0:
#                 self.save_model(server_round, is_periodic=True)
            
#             # Always save latest (overwrites each round)
#             self.save_model(server_round, is_latest=True)
        
#         return aggregated_parameters, aggregated_metrics
    
#     def save_model(self, round_num, is_periodic=False, is_latest=False, is_final=False):
#         """Save model using detectron2's format."""
#         if self.final_parameters is None:
#             print("[SERVER] No parameters to save!")
#             return
            
#         try:
                        
#             # Create model template
#             model = Net()
            
#             # Convert parameters to state dict
#             params_dict = zip(model.state_dict().keys(), 
#                              parameters_to_ndarrays(self.final_parameters))
#             state_dict = {k: torch.tensor(v) for k, v in params_dict}
            
#             # Load weights into model
#             model.load_state_dict(state_dict)
            
#             # Use detectron2's checkpointer (handles complex models properly)
#             checkpointer = DetectionCheckpointer(model, save_dir=self.save_dir)
            
#             # Determine save name and type
#             if is_latest:
#                 save_name = "latest_model"
#                 save_type = "Latest"
#             elif is_final:
#                 save_name = "final_diffdet_fl_model"
#                 save_type = "Final"
#             elif is_periodic:
#                 save_name = f"model_round_{round_num}"
#                 save_type = f"Round {round_num}"
#             else:
#                 save_name = f"model_round_{round_num}"
#                 save_type = f"Round {round_num}"
            
#             # Save using detectron2 format
#             model_path = checkpointer.save(save_name)
            
#             # Also save as standard checkpoint format
#             checkpoint_data = {
#                 "model": state_dict,
#                 "iteration": round_num,
#                 "fl_training": True,
#                 "model_type": "DiffusionDet",
#                 "save_type": save_type
#             }
            
#             checkpoint_path = os.path.join(self.save_dir, f"{save_name}_checkpoint.pth")
#             torch.save(checkpoint_data, checkpoint_path)
            
#             if is_periodic:
#                 print(f"[SERVER] ✅ Periodic save completed:")
#                 print(f"[SERVER]   Detectron2 format: {model_path}")
#                 print(f"[SERVER]   Checkpoint format: {checkpoint_path}")
#             elif is_latest:
#                 print(f"[SERVER] 💾 Latest model updated: {model_path}")
#             elif is_final:
#                 print(f"[SERVER] ✅ Final FL model saved:")
#                 print(f"[SERVER]   Detectron2 format: {model_path}")
#                 print(f"[SERVER]   Checkpoint format: {checkpoint_path}")
#                 print(f"[SERVER]   Training completed after {round_num} rounds")
                
#                 # Save final config info
#                 config_info = {
#                     'rounds_trained': round_num,
#                     'model_type': 'DiffusionDet',
#                     'config_file': 'configs/diffdet_config.yaml',
#                     'detectron2_model': model_path,
#                     'checkpoint_file': checkpoint_path
#                 }
                
#                 config_path = os.path.join(self.save_dir, "model_info.txt")
#                 with open(config_path, 'w') as f:
#                     for key, value in config_info.items():
#                         f.write(f"{key}: {value}\n")
#                 print(f"[SERVER]   Info: {config_path}")
            
#         except Exception as e:
#             print(f"[SERVER] ❌ Error saving {save_type.lower()} model: {e}")
#             import traceback
#             print(f"[SERVER] Full error: {traceback.format_exc()}")
    
#     def save_final_model(self):
#         """Save the final global model (called by atexit)."""
#         if self.final_parameters is not None:
#             self.save_model(self.current_round, is_final=True)

# net = Net()
# def server_fn(context: Context):
#     """Create server components with initial model parameters and periodic saving."""
#     # Read configuration
#     num_rounds = context.run_config["num-server-rounds"]
#     fraction_fit = context.run_config["fraction-fit"]
    
#     print(f"[SERVER] Starting FL server for {num_rounds} rounds")
#     print(f"[SERVER] Fraction fit: {fraction_fit}")
    
#     # Initialize model and extract parameters
#     ndarrays = get_weights(net)
#     parameters = ndarrays_to_parameters(ndarrays)
    
#     print(f"[SERVER] Loaded initial parameters from DiffusionDet checkpoint")
#     print(f"[SERVER] Number of parameter arrays: {len(ndarrays)}")

#     run_config = context.run_config
#     strategy = FedAvgWithPeriodicSave(
#         save_dir="./saved_models",
#         save_every=10,
#         fraction_fit=fraction_fit,
#         fraction_evaluate=1.0,
#         min_available_clients=2,
#         initial_parameters=parameters,
#         # Pass the config to clients
#         on_fit_config_fn=lambda server_round: {
#             **run_config,  # Include all TOML config
#             "server_round": server_round
#         },
#         on_evaluate_config_fn=lambda server_round: {
#             **run_config,  # Include all TOML config  
#             "server_round": server_round
#         }
#     )
    
#     # Register cleanup function to save final model
#     atexit.register(strategy.save_final_model)
    
#     # Create server configuration
#     config = ServerConfig(num_rounds=num_rounds)

#     return ServerAppComponents(strategy=strategy, config=config)


# # Create ServerApp
# app = ServerApp(server_fn=server_fn)



"""fl: A Flower / PyTorch server app with configurable strategies from TOML."""

import torch
import os
import atexit
from detectron2.checkpoint import DetectionCheckpointer
from flwr.common import Context, ndarrays_to_parameters, parameters_to_ndarrays
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from fl.task import Net, get_weights
from fl.strategies import create_strategy


net = Net()

def server_fn(context: Context):
    """Create server components with TOML-configured strategy."""
    # Read configuration
    num_rounds = context.run_config["num-server-rounds"]
    fraction_fit = context.run_config["fraction-fit"]
    run_config = context.run_config
    
    # Strategy configuration from TOML
    strategy_name = run_config.get("strategy", "fedavg")
    save_every = run_config.get("save_every", 10)
    output_base = run_config.get("output_base", "./output_temp")
    
    # Strategy-specific hyperparameters
    strategy_params = {
        "output_base": output_base,
        "save_every": save_every,
        "fraction_fit": fraction_fit,
        "fraction_evaluate": 1.0,
        "min_available_clients": 2,
        "enable_global_eval": run_config.get("enable_global_eval", False),
        "global_eval_frequency": run_config.get("global_eval_frequency", 5),
    }
    
    # Add strategy-specific parameters
    if strategy_name == "fedprox":
        strategy_params["mu"] = run_config.get("mu", 0.01)
    elif strategy_name.startswith("size_"):
        strategy_params["adjustment"] = run_config.get("adjustment", "sqrt")
        
    # Initialize model and extract parameters
    ndarrays = get_weights(net)
    parameters = ndarrays_to_parameters(ndarrays)
    strategy_params["initial_parameters"] = parameters
    
    # Add config passing functions
    strategy_params.update({
        "on_fit_config_fn": lambda server_round: {
            **run_config,
            "server_round": server_round,
            "mu": run_config.get("mu", 0.01),  # Ensure mu is passed
            "strategy": run_config.get("strategy", "fedavg")  # Ensure strategy is passed
        },
        "on_evaluate_config_fn": lambda server_round: {
            **run_config,
            "server_round": server_round
        }
    })

    # strategy_params.update({
    #     "on_fit_config_fn": lambda server_round: {
    #         **run_config,
    #         "server_round": server_round
    #     },
    #     "on_evaluate_config_fn": lambda server_round: {
    #         **run_config,
    #         "server_round": server_round
    #     }
    # })
    
    # Create strategy using factory
    strategy_params["shared_net"] = net
    strategy = create_strategy(strategy_name, **strategy_params)
    
    print(f"[SERVER] Created strategy: {type(strategy).__name__}")

    # Register cleanup function to save final model
    atexit.register(strategy.save_final_model)
    
    # Create server configuration
    config = ServerConfig(num_rounds=num_rounds)

    return ServerAppComponents(strategy=strategy, config=config)


# Create ServerApp
app = ServerApp(server_fn=server_fn)