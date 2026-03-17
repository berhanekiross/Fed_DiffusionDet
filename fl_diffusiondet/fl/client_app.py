
"""fl: A Flower / PyTorch app."""

import os
import torch
import json
# torch.cuda.set_per_process_memory_fraction(0.12)

import logging
from flwr.client import ClientApp, NumPyClient
from flwr.common import Context
from fl.task import Net, get_weights, load_data, set_weights, test, train, coco_test
from diffusiondet.util.model_ema import may_build_model_ema

class FlowerClient(NumPyClient):
    """Flower client for federated DiffusionDet training."""

    def __init__(self, net, trainloader, valloader, client_class, dataset_size=100):
        print(f"[INIT] Starting FlowerClient init for {client_class}", flush=True)
        if hasattr(self, '_initialized'):
            return

        try:
            self.net = net
            self.trainloader = trainloader
            self.valloader = valloader
            self.client_class = client_class
            self.dataset_size = dataset_size  
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            self.net.to(self.device)
            self._initialized = True
            # self.logger = self.setup_client_logger(client_class)
            print(f"[CLIENT {client_class}] Initialized on device: {self.device}")
            print(f"[INIT] FlowerClient init completed for {client_class}", flush=True)
        except Exception as e:
            print(f"[INIT] ERROR in FlowerClient init: {e}", flush=True)
            raise    
 
    def fit(self, parameters, config):
        # print(f"[CLIENT {self.client_class}] FIT method entry point reached", flush=True)
        """Enhanced fit method with comprehensive tracking."""
        current_round = config.get("server_round", "unknown")
        import sys
        print(f"[CLIENT {self.client_class}] FIT START - Round {config.get('server_round', 'unknown')}", flush=True)
        sys.stdout.flush()
        try:    
            set_weights(self.net, parameters)
            
            # Enhanced training with loss tracking
            train_loss, train_loss_tracking = train(
                self.net, 
                self.trainloader, 
                epochs=1, 
                device=self.device,
                server_round=current_round,  # Add this
                lr_config=config  # Add this (passes TOML config)
            )
            updated_weights = get_weights(self.net)
            
            # Evaluation with comprehensive metrics
            eval_metrics = {}
            val_loss_tracking = {}
            
            # print(f"[CLIENT {self.client_class}] Checking if evaluation is needed for round {current_round}")
            if self.should_evaluate(config, current_round):
                val_loss, eval_metrics, val_loss_tracking = test(self.net, self.valloader, self.device, eval_config=config)
            
            # Organize all tracking data
            comprehensive_loss_tracking = {
                "training": train_loss_tracking,
                "validation": val_loss_tracking
            }
            
            # Save comprehensive metrics
            self.save_comprehensive_metrics(current_round, eval_metrics, comprehensive_loss_tracking, config)
            
            # Return only simple metrics through Flower
            simple_metrics = {
                "train_loss": train_loss,
                "client": self.client_class,
                "round": current_round,
                "training_iterations": len(train_loss_tracking.get("detailed_losses", [])),
                "validation_batches": val_loss_tracking.get("batch_count", 0)
            }
            # print(f"[CLIENT {self.client_class}] FIT SUCCESS", flush=True)
            return (updated_weights, self.dataset_size, simple_metrics)
        except Exception as e:
            print(f"[CLIENT {self.client_class}] FIT ERROR: {str(e)}", flush=True)
            import traceback
            traceback.print_exc()
            sys.stdout.flush()
            raise

    def should_evaluate(self, config, current_round):
        """Check if should evaluate this round."""
        perform_eval = config.get("perform_eval", False)
        # print(f"[DEBUG] perform_eval from config: {perform_eval} (type: {type(perform_eval)})")
        
        if not perform_eval:
            return False
        
        eval_frequency = config.get("eval_frequency", 1)
        should_eval = (current_round == 1 or current_round % eval_frequency == 0)
        # print(f"[DEBUG] Round {current_round}, eval_frequency {eval_frequency}, should_evaluate: {should_eval}")
        return should_eval

    def save_comprehensive_metrics(self, round_num, eval_metrics, loss_tracking, config):
        """Save comprehensive metrics to organized directory structure."""
        # print(f"[CLIENT {self.client_class}] Saving comprehensive metrics for round {round_num}", flush=True)
        output_base = config.get("output_base", "./diffdet_analysis")
        
        # Create organized directory structure
        directories = {
            'losses': os.path.join(output_base, 'losses'),
            'metrics': os.path.join(output_base, 'metrics'), 
            'confusion': os.path.join(output_base, 'confusion')
        }
        
        for dir_path in directories.values():
            os.makedirs(dir_path, exist_ok=True)
        
        # 1. Save loss tracking
        loss_filename = os.path.join(directories['losses'], f"client_{self.client_class}_losses.json")
        
        # Load existing loss data or create new
        if os.path.exists(loss_filename):
            with open(loss_filename, 'r') as f:
                all_losses = json.load(f)
        else:
            all_losses = {"rounds": {}, "client_info": {"class": self.client_class, "device": str(self.device)}}
        
        # Add this round's loss data
        all_losses["rounds"][f"round_{round_num}"] = {
            "timestamp": round_num,
            "training_losses": loss_tracking.get("training", {}),
            "validation_losses": loss_tracking.get("validation", {}),
            "round_summary": {
                "avg_train_loss": loss_tracking.get("training", {}).get("summary", {}).get("final_avg_loss", 0.0),
                "avg_val_loss": loss_tracking.get("validation", {}).get("summary", {}).get("avg_loss", 0.0)
            }
        }
        
        with open(loss_filename, 'w') as f:
            json.dump(all_losses, f, separators=(',', ':'), default=str)
            # json.dump(all_losses, f, indent=1, default=str)
        
        # 2. Save detection metrics (without confusion matrix)
        metrics_filename = os.path.join(directories['metrics'], f"client_{self.client_class}_metrics.json")
        
        # Separate confusion matrix from other metrics
        metrics_to_save = eval_metrics.copy()
        confusion_matrix = metrics_to_save.pop('confusion_matrix', None)
        
        # Load existing metrics or create new
        if os.path.exists(metrics_filename):
            with open(metrics_filename, 'r') as f:
                all_metrics = json.load(f)
        else:
            all_metrics = {"rounds": {}, "client_info": {"class": self.client_class}}
        
        # Add this round's metrics
        all_metrics["rounds"][f"round_{round_num}"] = {
            "detection_metrics": metrics_to_save,
            "round_info": {
                "timestamp": round_num,
                "evaluation_batches": metrics_to_save.get("total_predictions", 0)
            }
        }
        
        with open(metrics_filename, 'w') as f:
            json.dump(all_metrics, f, separators=(',', ':'), default=str)
            # json.dump(all_metrics, f, indent=1, default=str)

        # 3. Save confusion matrix (all rounds in one file)
        if confusion_matrix is not None:
            confusion_filename = os.path.join(directories['confusion'], 
                                            f"client_{self.client_class}_confusion_all_rounds.json")
            
            # Load existing or create new
            if os.path.exists(confusion_filename):
                with open(confusion_filename, 'r') as f:
                    all_confusion = json.load(f)
            else:
                all_confusion = {"client": self.client_class, "rounds": {}}
            
            # Add this round's data
            all_confusion["rounds"][f"round_{round_num}"] = {
                "confusion_matrix": confusion_matrix,
                "matrix_shape": [len(confusion_matrix), len(confusion_matrix[0])],
                "class_names": eval_metrics.get("class_names", ["Car", "Van", "Truck", "Pedestrian", "Person_sitting", "Cyclist", "Tram"])
            }
            
            with open(confusion_filename, 'w') as f:
                json.dump(all_confusion, f, separators=(',', ':'), default=str)
            # with open(confusion_filename, 'w') as f:
            #     json.dump(all_confusion, f, indent=1, default=str)

    def evaluate(self, parameters, config):
        """Switch between evaluators based on config"""
        if config.get("use_coco_eval", False):
            return self.coco_evaluate(parameters, config)
        else:
            return self.original_evaluate(parameters, config)

    def original_evaluate(self, parameters, config):
        """Evaluate model with received parameters."""
        # print(f"[CLIENT {self.client_class}] Starting evaluate")
        
        set_weights(self.net, parameters)
        loss, detection_metrics, _ = test(self.net, self.valloader, self.device)

        flat_metrics = self.flatten_for_flower(detection_metrics)  
        flat_metrics["client"] = self.client_class
        
        return float(loss), self.dataset_size, flat_metrics

    def coco_evaluate(self, parameters, config):
        """Evaluate using COCO evaluator with logging compatibility"""
        set_weights(self.net, parameters)
        
        # Get loss from regular test function  
        loss, _, loss_tracking = test(self.net, self.valloader, self.device, eval_config={"perform_eval": False})
        
        # Run COCO evaluation
        coco_results = coco_test(self.net, self.client_class, self.device)
        bbox_metrics = coco_results.get("bbox", {})
        
        # Format for your logger (same structure as FastDetectionMetrics)
        eval_metrics = {
            "evaluator_type": "coco",
            "coco_ap": bbox_metrics.get("AP", 0.0),
            "coco_ap50": bbox_metrics.get("AP50", 0.0), 
            "coco_ap75": bbox_metrics.get("AP75", 0.0),
            "coco_ar": bbox_metrics.get("AR", 0.0),
            "class_names": ["Car", "Van", "Truck", "Pedestrian", "Person_sitting", "Cyclist", "Tram"],
            "total_predictions": bbox_metrics.get("total_predictions", 0),
            # No confusion_matrix for COCO
        }
        
        # Organize loss tracking (same format as your custom evaluator)
        comprehensive_loss_tracking = {
            "training": {},  # No training loss in evaluation
            "validation": loss_tracking
        }
        
        # Save using existing logger - SAME CALL as custom evaluator
        self.save_comprehensive_metrics(
            config.get("server_round", "unknown"), 
            eval_metrics, 
            comprehensive_loss_tracking, 
            config
        )
        
        # Return for Flower
        return (float(loss), self.dataset_size, {"coco_ap": bbox_metrics.get("AP", 0.0)})

    def flatten_for_flower(self, metrics):
        flat = {}
        for k, v in metrics.items():
            if isinstance(v, (int, float, str, bool)):
                flat[k] = v
            elif isinstance(v, list) and all(isinstance(i, (int, float, str, bool)) for i in v):
                flat[k] = v
            elif isinstance(v, dict):
                if k == "confidence_analysis":
                    # Special handling for confidence analysis - flatten to summary stats
                    flat[f"{k}_num_thresholds"] = len(v)
                    # Just take a few key thresholds for Flower
                    key_thresholds = ["conf_0.1", "conf_0.5", "conf_0.9"]
                    for thresh in key_thresholds:
                        if thresh in v:
                            thresh_data = v[thresh]
                            if "precision_mean" in thresh_data:
                                flat[f"precision_mean_{thresh}"] = float(thresh_data["precision_mean"])
                            if "recall_mean" in thresh_data:
                                flat[f"recall_mean_{thresh}"] = float(thresh_data["recall_mean"])
                            if "f1_mean" in thresh_data:
                                flat[f"f1_mean_{thresh}"] = float(thresh_data["f1_mean"])
                else:
                    # Regular dict flattening
                    for sub_k, sub_v in v.items():
                        if isinstance(sub_v, (int, float, str, bool)):
                            flat[f"{k}_{sub_k}"] = sub_v
                        else:
                            print(f"[WARN] Nested field {k}->{sub_k} of type {type(sub_v)} skipped")
            else:
                print(f"[WARN] Skipping metric {k} of type {type(v)}")
        return flat

# Load model once at module level
net = Net()
if net.cfg.MODEL_EMA.ENABLED and not hasattr(net.model, 'ema'):
    may_build_model_ema(net.cfg, net.model)
    print("[MODULE] EMA built once for shared model")
    
def client_fn(context: Context):
    """Create a Flower client instance using shared model."""
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
   
    trainloader, valloader, client_class, dataset_size = load_data(partition_id, num_partitions, context)
    
    return FlowerClient(net, trainloader, valloader, client_class, dataset_size).to_client()

# Create ClientApp
app = ClientApp(client_fn)


# def client_fn(context: Context):
#     """Create a Flower client instance using shared model."""
#     partition_id = context.node_config["partition-id"]
#     num_partitions = context.node_config["num-partitions"]
   
#     # Pass context to load_data for flexible partition type handling
#     trainloader, valloader, client_identifier, dataset_size = load_data(
#         partition_id, num_partitions, context
#     )
    
#     return FlowerClient(net, trainloader, valloader, client_identifier, dataset_size).to_client()
# app = ClientApp(client_fn)
# def client_fn(context: Context):
#     """Create a Flower client instance using shared model."""
#     print(f"[CLIENT_FN] Starting client_fn", flush=True)
    
#     try:
#         partition_id = context.node_config["partition-id"]
#         num_partitions = context.node_config["num-partitions"]
#         print(f"[CLIENT_FN] Got partition_id={partition_id}, num_partitions={num_partitions}", flush=True)
        
#         trainloader, valloader, client_class, dataset_size = load_data(partition_id, num_partitions)
#         print(f"[CLIENT_FN] Loaded data for client {client_class}", flush=True)
        
#         client = FlowerClient(net, trainloader, valloader, client_class, dataset_size)
#         print(f"[CLIENT_FN] Created FlowerClient for {client_class}", flush=True)
        
#         return client.to_client()
        
#     except Exception as e:
#         print(f"[CLIENT_FN] ERROR in client_fn: {e}", flush=True)
#         import traceback
#         traceback.print_exc()
#         raise
# app = ClientApp(client_fn)