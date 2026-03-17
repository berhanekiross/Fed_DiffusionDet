"""fl: Federated learning strategies for DiffusionDet."""

import torch
import os
import atexit
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from flwr.common import FitRes, Parameters, Scalar, ndarrays_to_parameters, parameters_to_ndarrays
from flwr.server.strategy import FedAvg
from flwr.server.client_proxy import ClientProxy
from detectron2.checkpoint import DetectionCheckpointer
from fl.task import Net, test, get_weights
from fl.task import get_cfg, add_diffusiondet_config, add_model_ema_configs
from diffusiondet import DiffusionDetDatasetMapper
from detectron2.data import build_detection_test_loader
from fl.config_utils import get_global_val_dataset_name
import json


model = Net()
class FedAvgBaseline(FedAvg):
    """Standard FedAvg baseline strategy with model saving and server-side global evaluation."""
        
    def __init__(self, save_every=10, shared_net=None, global_eval_frequency=5, 
                enable_global_eval=False, output_base=None, **kwargs):
        super().__init__(**kwargs)

        self.output_base = output_base  # Keep for model saving
        self.save_every = save_every
        self.final_parameters = None
        self.current_round = 0
        self.global_eval_frequency = global_eval_frequency
        self.enable_global_eval = enable_global_eval
        self.shared_net = shared_net
        self._global_loader = None
        
        # Create directories
        os.makedirs(output_base, exist_ok=True)
        
        # Create global evaluation results directory under output_base
        if self.enable_global_eval:
            self.global_eval_dir = os.path.join(self.output_base, "global_evaluation")
            os.makedirs(self.global_eval_dir, exist_ok=True)
            print(f"[STRATEGY] Global evaluation enabled every {global_eval_frequency} rounds")
            # print(f"[STRATEGY] Global eval output: {self.global_eval_dir}")
        
        # print("[STRATEGY] Using FedAvg baseline")
    
    def _get_global_validation_loader(self):
        """Get global validation dataset loader (lazy initialization)."""
        if self._global_loader is None:           
            # Create config for global validation
            cfg = get_cfg()
            add_diffusiondet_config(cfg)
            add_model_ema_configs(cfg)
            cfg.merge_from_file("configs/diffdet_config.yaml")
            
            # Build global validation loader
            global_val_name = get_global_val_dataset_name()
            val_mapper = DiffusionDetDatasetMapper(cfg, is_train=True)
            self._global_loader = build_detection_test_loader(cfg, global_val_name, mapper=val_mapper)
            
            # print(f"[STRATEGY] Initialized global validation loader: {global_val_name}")
        
        return self._global_loader
    
    def evaluate_global_model(self, server_round, parameters):
        """Evaluate global model on server using global validation dataset."""
        if not self.enable_global_eval or server_round % self.global_eval_frequency != 0:
            return {}
        
        # print(f"[STRATEGY] Running global evaluation - Round {server_round}")
        
        try:         
            # Use self.shared_net instead of creating new Net()
            model = self.shared_net
            params_dict = zip(model.state_dict().keys(), parameters_to_ndarrays(parameters))
            state_dict = {k: torch.tensor(v) for k, v in params_dict}
            model.load_state_dict(state_dict)
            
            # Get global validation dataset
            global_loader = self._get_global_validation_loader()
            
            # Run evaluation
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            model.to(device)
            
            eval_config = {"perform_eval": True, "perform_logging": False}
            loss, metrics, loss_tracking = test(model, global_loader, device, eval_config=eval_config)
            
            # Save global evaluation results
            self._save_global_evaluation(server_round, loss, metrics, loss_tracking)
            
            # Return summary metrics for aggregation
            summary_metrics = {
                "global_loss": float(loss),
                "global_precision_mean": metrics.get("precision_mean", 0.0),
                "global_recall_mean": metrics.get("recall_mean", 0.0),
                "global_f1_mean": metrics.get("f1_mean", 0.0),
                "global_map": metrics.get("ap_metrics", {}).get("map", 0.0),
                "global_map50": metrics.get("ap_metrics", {}).get("map50", 0.0)
            }
            
            print(f"[STRATEGY] Global eval complete - Loss: {loss:.4f}, mAP: {summary_metrics['global_map']:.4f}")
            return summary_metrics
            
        except Exception as e:
            print(f"[STRATEGY] Global evaluation failed: {e}")
            import traceback
            print(f"[STRATEGY] Traceback: {traceback.format_exc()}")
            return {"global_eval_error": str(e)}
            
    def _save_global_evaluation(self, server_round, loss, metrics, loss_tracking):
        """Save detailed global evaluation results to single comprehensive file."""
        import json
        
        # Single comprehensive file for all rounds
        comprehensive_file = os.path.join(self.global_eval_dir, "global_evaluation_all_rounds.json")
        
        # Load existing data or create new
        if os.path.exists(comprehensive_file):
            with open(comprehensive_file, 'r') as f:
                all_data = json.load(f)
        else:
            all_data = {
                "strategy": self.__class__.__name__,
                "evaluation_type": "global_server_side", 
                "rounds": {}
            }
        
        # Add this round's complete data (readable multi-line)
        all_data["rounds"][f"round_{server_round}"] = {
            "round": server_round,
            "global_loss": loss,
            "detection_metrics": metrics,
            "loss_tracking": loss_tracking,
            "timestamp": server_round,
            "summary": {
                "precision_mean": metrics.get("precision_mean", 0.0),
                "recall_mean": metrics.get("recall_mean", 0.0), 
                "f1_mean": metrics.get("f1_mean", 0.0),
                "map": metrics.get("ap_metrics", {}).get("map", 0.0),
                "map50": metrics.get("ap_metrics", {}).get("map50", 0.0)
            }
        }
        
        # Save compact JSON (only the output is compact)
        with open(comprehensive_file, 'w') as f:
            json.dump(all_data, f, separators=(',', ':'), default=str)
            # json.dump(all_data, f, indent=1, default=str)
        
        print(f"[STRATEGY] Saved global evaluation to: {comprehensive_file}")
        
    def save_final_model(self):
        """Save the final global model (called by atexit)."""
        if self.final_parameters is not None:
            self.save_model(self.current_round, f"{self.__class__.__name__}_final")
    
    def aggregate_fit(self, server_round, results, failures):
        """Standard FedAvg aggregation with saving and global evaluation."""
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(
            server_round, results, failures
        )
        
        if aggregated_parameters is not None:
            self.final_parameters = aggregated_parameters
            self.current_round = server_round
            
            # Periodic model saving
            if server_round % self.save_every == 0:
                self.save_model(server_round, "FedAvg")
            
            # Server-side global evaluation
            global_eval_metrics = self.evaluate_global_model(server_round, aggregated_parameters)
            if global_eval_metrics:
                aggregated_metrics.update(global_eval_metrics)
        
        return aggregated_parameters, aggregated_metrics
     

    def save_model(self, round_num, strategy_name):
        """Save model checkpoint."""
        if self.final_parameters is None:
            print(f"[{strategy_name}] No parameters to save!")
            return
        
        try:
            if self.shared_net is not None:
                model = self.shared_net.model
                print(f"[{strategy_name}] Using shared_net.model: {type(model)}")
            else:
                net = Net()
                model = net.model
                print(f"[{strategy_name}] Created new net.model: {type(model)}")
            
            # Debug: Check if model has state_dict
            print(f"[{strategy_name}] Model has state_dict: {hasattr(model, 'state_dict')}")
            print(f"[{strategy_name}] Model type: {type(model)}")
            
            # Just save as simple PyTorch checkpoint (skip DetectionCheckpointer for now)
            params_dict = zip(model.state_dict().keys(), 
                            parameters_to_ndarrays(self.final_parameters))
            state_dict = {k: torch.tensor(v) for k, v in params_dict}
            
            # Simple PyTorch save
            checkpoint_path = os.path.join(self.output_base, f"{strategy_name}_round_{round_num}.pth")
            torch.save({
                "model_state_dict": state_dict,
                "round": round_num,
                "strategy": strategy_name
            }, checkpoint_path)
            
            print(f"[{strategy_name}] Saved model to: {checkpoint_path}")
            
        except Exception as e:
            print(f"[{strategy_name}] Save error: {e}")
            import traceback
            print(f"[{strategy_name}] Traceback: {traceback.format_exc()}")

class FedProxStrategy(FedAvgBaseline):
    """FedProx strategy with proximal term."""
    
    def __init__(self, mu=0.01, **kwargs):
        # Remove mu from kwargs before passing to parent to avoid duplicate parameter
        kwargs.pop('mu', None)
        super().__init__(**kwargs)
        self.mu = mu  # Proximal term coefficient
        # print(f"[STRATEGY] Using FedProx (mu={mu})")
        
    # def configure_fit(self, server_round, parameters, client_manager):
    #     """Send global model and proximal coefficient to clients."""
        
    #     # Debug: Check if on_fit_config_fn exists
    #     print(f"[FEDPROX] Has on_fit_config_fn: {hasattr(self, 'on_fit_config_fn')}")
        
    #     # Get the full run_config from on_fit_config_fn if available
    #     base_config = {}
    #     if hasattr(self, 'on_fit_config_fn') and self.on_fit_config_fn:
    #         try:
    #             base_config = self.on_fit_config_fn(server_round)
    #             print(f"[FEDPROX] Base config keys: {list(base_config.keys())}")
    #         except Exception as e:
    #             print(f"[FEDPROX] Error getting base config: {e}")
        
    #     # Add FedProx-specific config
    #     config =base_config
    #     # config = {
    #     #     **base_config,
    #     #     "strategy": "fedprox",
    #     #     "mu": self.mu,
    #     #     "server_round": server_round,  # Ensure this is set
    #     # }
        
    #     print(f"[FEDPROX] Final config keys: {list(config.keys())}")

        
    #     # Standard client sampling
    #     sample_size, min_num_clients = self.num_fit_clients(
    #         client_manager.num_available()
    #     )
    #     clients = client_manager.sample(
    #         num_clients=sample_size, min_num_clients=min_num_clients
    #     )
    #     print(f"[FEDPROX] Sampled {len(clients)} clients")
    #     print(f"[FEDPROX] About to return config to clients")    
    #     return [(client, config) for client in clients]

class FedNovaStrategy(FedAvgBaseline):
    """FedNova strategy for heterogeneous local updates."""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.client_momentums = {}  # Track client momentum
        print("[STRATEGY] Using FedNova")
    
    def aggregate_fit(self, server_round, results, failures):
        """FedNova aggregation with normalized local updates."""
        if not results:
            return None, {}
        
        # Extract weights and local steps from each client
        weights_results = []
        total_tau = 0  # Total effective local steps
        
        for client_proxy, fit_res in results:
            if fit_res.parameters is not None:
                # Get local steps from client metrics
                local_steps = fit_res.metrics.get("local_steps", 1)
                client_tau = local_steps
                
                weights = parameters_to_ndarrays(fit_res.parameters)
                weights_results.append((weights, fit_res.num_examples, client_tau))
                total_tau += client_tau * fit_res.num_examples
        
        if not weights_results:
            return None, {}
        
        # Normalize by total tau
        total_examples = sum(num_examples for _, num_examples, _ in weights_results)
        normalized_tau = total_tau / total_examples
        
        # FedNova aggregation
        aggregated_weights = []
        for i in range(len(weights_results[0][0])):  # For each layer
            layer_weights = []
            total_weight = 0
            
            for weights, num_examples, client_tau in weights_results:
                # FedNova normalization factor
                nova_weight = (client_tau / normalized_tau) * num_examples
                layer_weights.append(weights[i] * nova_weight)
                total_weight += nova_weight
            
            # Aggregate this layer
            aggregated_layer = sum(layer_weights) / total_weight
            aggregated_weights.append(aggregated_layer)
        
        # Create aggregated parameters
        aggregated_parameters = ndarrays_to_parameters(aggregated_weights)
        
        # Update tracking
        if aggregated_parameters is not None:
            self.final_parameters = aggregated_parameters
            self.current_round = server_round
            
            if server_round % self.save_every == 0:
                self.save_model(server_round, "FedNova")
        
        # Aggregate metrics
        aggregated_metrics = {}
        if results:
            total_examples = sum(fit_res.num_examples for _, fit_res in results)
            avg_loss = sum(fit_res.metrics.get("train_loss", 0) * fit_res.num_examples 
                          for _, fit_res in results) / total_examples
            aggregated_metrics = {"avg_train_loss": avg_loss}
        
        return aggregated_parameters, aggregated_metrics
    
    # def configure_fit(self, server_round, parameters, client_manager):
    #     """Configure FedNova training."""
    #     config = {
    #         "server_round": server_round,
    #         "strategy": "fednova", 
    #         "local_epochs": 1
    #     }
        
    #     sample_size, min_num_clients = self.num_fit_clients(
    #         client_manager.num_available()
    #     )
    #     clients = client_manager.sample(
    #         num_clients=sample_size, min_num_clients=min_num_clients
    #     )
        
    #     return [(client, config) for client in clients]


class SizeAdjustedFedAvg(FedAvgBaseline):
    """FedAvg with client data size adjustments (sqrt weighting)."""
    
    def __init__(self, adjustment="sqrt", **kwargs):
        super().__init__(**kwargs)
        self.adjustment = adjustment  # "sqrt", "log", or "linear"
        print(f"[STRATEGY] Using Size-Adjusted FedAvg ({adjustment} weighting)")
    
    def aggregate_fit(self, server_round, results, failures):
        """Aggregate with adjusted client weights."""
        if not results:
            return None, {}
        
        # Calculate adjusted weights
        weights_results = []
        total_adjusted_weight = 0
        
        for client_proxy, fit_res in results:
            if fit_res.parameters is not None:
                num_examples = fit_res.num_examples
                
                # Apply size adjustment
                if self.adjustment == "sqrt":
                    adjusted_weight = np.sqrt(num_examples)
                elif self.adjustment == "log":
                    adjusted_weight = np.log(num_examples + 1)
                elif self.adjustment == "cbrt":  # Cube root
                    adjusted_weight = np.power(num_examples, 1/3)
                else:  # linear (standard)
                    adjusted_weight = num_examples
                
                weights = parameters_to_ndarrays(fit_res.parameters)
                weights_results.append((weights, adjusted_weight))
                total_adjusted_weight += adjusted_weight
        
        if not weights_results:
            return None, {}
        
        # Weighted average with adjusted weights
        aggregated_weights = []
        for i in range(len(weights_results[0][0])):  # For each layer
            layer_sum = sum(weights[i] * weight 
                           for weights, weight in weights_results)
            aggregated_layer = layer_sum / total_adjusted_weight
            aggregated_weights.append(aggregated_layer)
        
        aggregated_parameters = ndarrays_to_parameters(aggregated_weights)
        
        # Update tracking
        if aggregated_parameters is not None:
            self.final_parameters = aggregated_parameters
            self.current_round = server_round
            
            if server_round % self.save_every == 0:
                self.save_model(server_round, f"SizeAdj{self.adjustment.title()}")
        
        # Aggregate metrics (using original example counts)
        aggregated_metrics = {}
        if results:
            total_examples = sum(fit_res.num_examples for _, fit_res in results)
            avg_loss = sum(fit_res.metrics.get("train_loss", 0) * fit_res.num_examples 
                          for _, fit_res in results) / total_examples
            aggregated_metrics = {"avg_train_loss": avg_loss}
        
        return aggregated_parameters, aggregated_metrics


def create_strategy(strategy_name: str, **kwargs):
    """Factory function to create strategies with global evaluation support."""
    
    # Extract global evaluation parameters from kwargs
    global_eval_params = {
        'enable_global_eval': kwargs.pop('enable_global_eval', False),
        'global_eval_frequency': kwargs.pop('global_eval_frequency', 5)
    }
    
    # Extract strategy-specific parameters
    mu_param = kwargs.pop('mu', 0.05)  # Remove mu from kwargs
    adjustment_param = kwargs.pop('adjustment', 'sqrt')  # Remove adjustment from kwargs
    
    strategies = {
        "fedavg": FedAvgBaseline,
        "fedprox": lambda **k: FedProxStrategy(mu=mu_param, **k),  # Pass mu explicitly
        "fednova": FedNovaStrategy,
        "size_sqrt": lambda **k: SizeAdjustedFedAvg(adjustment="sqrt", **k),
        "size_log": lambda **k: SizeAdjustedFedAvg(adjustment="log", **k),
        "size_cbrt": lambda **k: SizeAdjustedFedAvg(adjustment="cbrt", **k),
    }
    
    if strategy_name not in strategies:
        raise ValueError(f"Unknown strategy: {strategy_name}. Available: {list(strategies.keys())}")
    
    # Add global evaluation parameters back to kwargs
    kwargs.update(global_eval_params)
    
    return strategies[strategy_name](**kwargs)