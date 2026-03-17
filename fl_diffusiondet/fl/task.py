
"""fl: A Flower / PyTorch app - DiffusionDet adaptation."""

import os
import sys
import torch
import json
from collections import OrderedDict
from detectron2.config import get_cfg
from detectron2.engine import DefaultTrainer
from detectron2.utils.logger import setup_logger
from detectron2.modeling import build_model
from detectron2.data import DatasetCatalog, build_detection_train_loader, build_detection_test_loader
from detectron2.checkpoint import DetectionCheckpointer

# DiffusionDet imports
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from diffusiondet import add_diffusiondet_config, DiffusionDetDatasetMapper
from diffusiondet.util.model_ema import add_model_ema_configs, may_build_model_ema, apply_model_ema_and_restore

from fl.config_utils import get_client_train_dataset_name, get_client_val_dataset_name, get_global_val_dataset_name
from register_kitti_fldata import register_kitti_splits
from detectron2.data import DatasetCatalog
    
from train_net import Trainer
from detectron2.utils.events import EventStorage
from fl.evaluation import FastDetectionMetrics, extract_predictions, extract_ground_truth, filter_by_confidence
from flwr.common import Context
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
import tempfile


def get_weights(net):
    """Extract model weights as list of numpy arrays."""
    return [val.cpu().numpy() for _, val in net.state_dict().items()]


def set_weights(net, parameters):
    """Set model weights from list of numpy arrays."""
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)

class Net:
    """DiffusionDet model wrapper with automatic checkpoint loading."""
    
    def __init__(self):
        # Setup detectron2 config
        cfg = get_cfg()
        add_diffusiondet_config(cfg)
        add_model_ema_configs(cfg)
        
        # Load config file
        config_path = os.environ.get("DIFFDET_CONFIG", "configs/diffdet_config.yaml")
        cfg.merge_from_file(config_path)   
        self.cfg = cfg
        
        # Build model
        self.model = build_model(cfg)
        
        # Load pretrained checkpoint
        checkpoint_path = os.environ.get("DIFFDET_CHECKPOINT", "diffdet_initial.pth")
        # checkpoint_path = os.environ.get("DIFFDET_CHECKPOINT", "diffdet_coco_res50.pth")
        if os.path.exists(checkpoint_path):
            checkpointer = DetectionCheckpointer(self.model)
            checkpointer.load(checkpoint_path)
            print(f"[NET] Loaded checkpoint: {checkpoint_path}")
        else:
            print(f"[NET] WARNING: Checkpoint {checkpoint_path} not found, using random initialization")
        
        self.model.train()
        # print(f"[NET] Model initialized: {cfg.MODEL.META_ARCHITECTURE}")
        # print(f"[NET] Num parameters: {sum(p.numel() for p in self.model.parameters()):,}")
    
    def state_dict(self):
        """Return model state dict for parameter exchange."""
        return self.model.state_dict()
    
    def load_state_dict(self, state_dict, strict=True):
        """Load state dict into model."""
        self.model.load_state_dict(state_dict, strict=strict)
    
    def parameters(self):
        """Return model parameters."""
        return self.model.parameters()
    
    def to(self, device):
        """Move model to device."""
        self.model.to(device)
        return self


def load_data(partition_id: int, num_partitions: int, context: Context = None):
    """Load partition-specific KITTI data for federated learning with flexible IID/Non-IID support."""
    
    # Import here to avoid circular imports
    from register_kitti_fldata import register_kitti_splits, register_kitti_iid_splits
    from fl.config_utils import get_data_partition_type, get_dataset_names_for_partition
    
    # Determine partition type from config
    if context is None:
        # Fallback to non-IID if no context provided
        partition_type = 'iid'
        print("[DATA] No context provided, defaulting to non-IID")
    else:
        partition_type = get_data_partition_type(context)
    
    print(f"[DATA] Using partition type: {partition_type}")
    
    # Register appropriate datasets
    if partition_type == 'iid':
        register_kitti_iid_splits()
        print("[DATA] Registered IID datasets")
    else:
        register_kitti_splits()
        print("[DATA] Registered Non-IID datasets")
    
    # Get dataset names based on partition type
    if context:
        train_dataset_name, val_dataset_name, client_identifier = get_dataset_names_for_partition(
            context, partition_id, num_partitions
        )
    else:
        # Fallback to non-IID behavior
        available_classes = ["Car", "Van", "Truck", "Pedestrian", "Tram_Sitting", "Cyclist"]
        client_class = available_classes[partition_id % len(available_classes)]
        train_dataset_name = f"fl_kitti_train_{client_class}"
        val_dataset_name = f"fl_kitti_val_{client_class}"
        client_identifier = client_class
        print("[DATA] Using fallback non-IID assignment")
    
    # debugger
    # print(f"[DATA] Client {partition_id} assigned to: {client_identifier}")
    # print(f"[DATA] Using train dataset: {train_dataset_name}")
    # print(f"[DATA] Using val dataset: {val_dataset_name}")
    
    # Create config for data loading
    cfg = get_cfg()
    add_diffusiondet_config(cfg)
    add_model_ema_configs(cfg)
    cfg.merge_from_file("configs/diffdet_config.yaml")
    
    # Configure datasets in config
    cfg.defrost()
    cfg.DATASETS.TRAIN = (train_dataset_name,)
    cfg.DATASETS.TEST = (val_dataset_name,)  
    cfg.freeze()
    
    # Build data loaders
    train_mapper = DiffusionDetDatasetMapper(cfg, is_train=True)
    train_loader = build_detection_train_loader(cfg, mapper=train_mapper)
    val_mapper = DiffusionDetDatasetMapper(cfg, is_train=True)  
    val_loader = build_detection_test_loader(cfg, val_dataset_name, mapper=val_mapper)
    
    # Get dataset sizes  
    train_dataset_size = len(DatasetCatalog.get(train_dataset_name))
    val_dataset_size = len(DatasetCatalog.get(val_dataset_name))
    
    print(f"[DATA] Train dataset {train_dataset_name} has {train_dataset_size} samples")
    print(f"[DATA] Val dataset {val_dataset_name} has {val_dataset_size} samples")
    
    return train_loader, val_loader, client_identifier, train_dataset_size



def train(net, trainloader, epochs, device, logger=None, server_round=1, lr_config=None):
    """Enhanced FL training with detailed loss tracking and FedProx support.
    some sections of the original centralised training [train_net] were incompatible 
    with the federated learning setup, so this one is custom made.
    """
    # print(f"[TRAIN] Starting training for server round {server_round}")
    
    def log_message(msg):
        if logger:
            logger.info(msg)
        else:
            print(f"[TRAIN] {msg}")
    
    # Create FL config
    cfg = net.cfg
    optimizer = Trainer.build_optimizer(cfg, net.model)
    # scheduler = Trainer.build_lr_scheduler(cfg, optimizer) # done by rounds
    
    # Get strategy type and mu from config
    strategy = lr_config.get("strategy", "fedavg") if lr_config else "fedavg"
    mu = lr_config.get("mu", 0.0) if lr_config else 0.0
    
    # Store initial global model parameters for FedProx
    global_params = None
    if strategy == "fedprox" and mu > 0:
        global_params = [p.clone().detach() for p in net.model.parameters()]
        print(f"[TRAIN] FedProx enabled with mu={mu}")

    # Apply federated LR scheduling
    print(f"[TRAIN] Server round {server_round} - applying LR scheduling")
    if lr_config and lr_config.get("lr_schedule"):
        print(f"[TRAIN] Using LR schedule: {lr_config['lr_schedule']}")
        new_lr = calculate_federated_lr(server_round, lr_config)
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lr
        print(f"[TRAIN] Round {server_round}: LR = {new_lr:.2e}")
    
    net.model.train()
    process_id = os.getpid()
    
    # Gradient accumulation settings - for smaller batch sizes, we accumulate gradients 
    # over multiple steps to simulate a larger effective batch size
    grad_accum_steps = 4
    
    # Loss tracking structure
    loss_tracking = {"iterations": [], "detailed_losses": [], "learning_rates": []}
    
    # Create persistent data iterator
    data_loader_iter = iter(trainloader)
    
    print(f"[TRAIN] Starting training")
    with EventStorage() as storage:
        for iteration in range(cfg.SOLVER.MAX_ITER):
            storage.iter = iteration
            
            # Accumulate gradients over multiple mini-batches
            accumulated_loss = 0.0
            iteration_losses = []
            
            for accum_step in range(grad_accum_steps):
                try:
                    batch = next(data_loader_iter)
                except StopIteration:
                    data_loader_iter = iter(trainloader)
                    batch = next(data_loader_iter)
                
                batch = [{k: v.to(device) if isinstance(v, torch.Tensor) else v 
                         for k, v in b.items()} for b in batch]
                
                # Forward pass
                loss_dict = net.model(batch)
                total_loss = sum(loss_dict.values())
                
                # Add FedProx proximal term
                if strategy == "fedprox" and mu > 0 and global_params is not None:
                    proximal_term = 0.0
                    for local_param, global_param in zip(net.model.parameters(), global_params):
                        proximal_term += torch.norm(local_param - global_param) ** 2
                    fedprox_loss = (mu / 2.0) * proximal_term
                    total_loss += fedprox_loss
                
                # Scale loss by accumulation steps
                scaled_loss = total_loss / grad_accum_steps
                scaled_loss.backward()
                
                # Track individual step losses
                step_losses = {k: v.item() for k, v in loss_dict.items()}
                if strategy == "fedprox" and mu > 0:
                    step_losses['fedprox_term'] = fedprox_loss.item()
                iteration_losses.append(step_losses)
                
                # Track accumulated loss
                accumulated_loss += total_loss.item()
            
            # Apply accumulated gradients
            if cfg.SOLVER.CLIP_GRADIENTS.ENABLED:
                if cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE == "full_model":
                    torch.nn.utils.clip_grad_norm_(
                        net.model.parameters(), 
                        cfg.SOLVER.CLIP_GRADIENTS.CLIP_VALUE
                    )
            
            optimizer.step()
            optimizer.zero_grad()
            # scheduler.step()
            
            # Update EMA
            if cfg.MODEL_EMA.ENABLED and hasattr(net.model, 'ema'):
                net.model.ema.update(net.model)
            
            # Calculate average loss for this iteration
            avg_loss_this_iter = accumulated_loss / grad_accum_steps
            current_lr = optimizer.param_groups[0]['lr']
            
            # Store detailed tracking
            loss_tracking["iterations"].append({
                "iteration": iteration, 
                "avg_loss": avg_loss_this_iter, 
                "step_losses": iteration_losses, 
                "learning_rate": current_lr, 
                "grad_accum_steps": grad_accum_steps
            })
            
            loss_tracking["detailed_losses"].append(avg_loss_this_iter)
            loss_tracking["learning_rates"].append(current_lr)
            
            # Store metrics
            storage.put_scalars(total_loss=avg_loss_this_iter)
            
            if iteration % 20 == 0:
                print(f"[TRAIN-PID-{process_id}] Round {server_round} Iter {iteration}: loss = {avg_loss_this_iter:.4f} (lr={current_lr:.2e})")
        
        # Calculate final statistics
        final_avg_loss = sum(loss_tracking["detailed_losses"]) / len(loss_tracking["detailed_losses"])
        loss_tracking["summary"] = {
            "final_avg_loss": final_avg_loss,
            "total_iterations": len(loss_tracking["detailed_losses"])
        }
        
        log_message(f"Training completed. Average loss: {final_avg_loss:.4f}")
        
        return final_avg_loss, loss_tracking

def test(net, testloader, device, eval_config=None):
    """Enhanced evaluation with detailed loss and metrics tracking."""
    net.to(device)
    
    # Default config if not provided
    if eval_config is None:
        eval_config = {"perform_eval": False, "perform_logging": False}
    
    # Loss computation structure
    loss_tracking = {"batch_losses": [], "detailed_losses": [], "batch_count": 0}
    
    net.model.train()  # For loss computation
    max_eval_batches = 250  # Increased for better statistics
    
    with torch.no_grad():
        for i, batch in enumerate(testloader):
            if i >= max_eval_batches:
                break
            
            batch = [{k: v.to(device) if isinstance(v, torch.Tensor) else v 
                     for k, v in b.items()} for b in batch]
            
            try:
                loss_dict = net.model(batch)
                total_loss = sum(loss_dict.values())
                
                # Track detailed losses
                batch_loss_detail = {k: v.item() for k, v in loss_dict.items()}
                batch_loss_detail['total'] = total_loss.item()
                
                loss_tracking["batch_losses"].append(total_loss.item())
                loss_tracking["detailed_losses"].append(batch_loss_detail)
                loss_tracking["batch_count"] += 1
                
            except Exception as e:
                print(f"[TEST] Warning: Batch {i} failed: {e}")
                continue
    
    # Calculate loss statistics
    if loss_tracking["batch_losses"]:
        avg_loss = sum(loss_tracking["batch_losses"]) / len(loss_tracking["batch_losses"])
        loss_tracking["summary"] = {
            "avg_loss": avg_loss,
            "batches_evaluated": len(loss_tracking["batch_losses"])
        }
    else:
        avg_loss = 0.0
        loss_tracking["summary"] = {"avg_loss": 0.0, "batches_evaluated": 0}
    
    # Detection metrics (only if requested)
    detection_metrics = {}
    if eval_config.get("perform_eval", False):
        try:
            detection_metrics = compute_detection_metrics(net, testloader, device)
        except Exception as e:
            print(f"[TEST] Warning: Detection metrics failed: {e}")
            detection_metrics = {"precision_mean": 0.0, "recall_mean": 0.0, "f1_mean": 0.0, "error": str(e)}

    return avg_loss, detection_metrics, loss_tracking    

def coco_test(net, client_class, context: Context = None):
    """Run COCO evaluation using registered dataset name."""

    if client_class == "global_val":
        # Global evaluation - use the exact registered dataset name
        dataset_name = "global_val"
    elif context is None:
        # Fallback to non-IID if no context provided
        dataset_name = f"fl_kitti_val_{client_class}"
    else:
        # IID client evaluation
        dataset_name = f"fl_kitti_iid_val_{client_class}"

    try:
        # Create fresh data loader using the registered dataset
        cfg = net.cfg
        val_mapper = DiffusionDetDatasetMapper(cfg, is_train=False)
        coco_val_loader = build_detection_test_loader(cfg, dataset_name, mapper=val_mapper)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            evaluator = COCOEvaluator(dataset_name, output_dir=temp_dir, tasks=("bbox",))
            net.model.eval()
            results = inference_on_dataset(net.model, coco_val_loader, evaluator)
            return results
            
    except Exception as e:
        print(f"[COCO_EVAL] Error with dataset {dataset_name}: {e}")
        return {"bbox": {"AP": 0.0, "AP50": 0.0, "AP75": 0.0}}

def calculate_federated_lr(server_round, lr_config):
    """Calculate learning rate for current round-lr scheduling. Supports cosine, exponential, and linear decay."""
    schedule = lr_config.get("lr_schedule", "cosine")
    initial_lr = lr_config.get("initial_lr", 0.000025)  
    final_lr = lr_config.get("final_lr", .0000025)
    decay_start = lr_config.get("lr_decay_start", 50)
    total_rounds = lr_config.get("num_rounds", 100)
    
    if server_round < decay_start:
        return initial_lr
    
    progress = (server_round - decay_start) / (total_rounds - decay_start)
    progress = min(1.0, progress)
    
    if schedule == "cosine":
        import math
        return final_lr + (initial_lr - final_lr) * 0.5 * (1 + math.cos(math.pi * progress))
    elif schedule == "exponential":
        decay_factor = (final_lr / initial_lr) ** progress
        return initial_lr * decay_factor
    else:  # linear
        return initial_lr - (initial_lr - final_lr) * progress

def compute_detection_metrics(net, testloader, device):
    """Compute detection metrics (P, R, F1, confusion matrix) with confidence analysis."""
    
    # Set model to eval mode for predictions
    net.model.eval()
    
    evaluator = FastDetectionMetrics(num_classes=7, confidence_threshold=0.3)
    
    # Collect predictions and ground truth
    all_predictions = []
    all_ground_truth = []

    with torch.no_grad():
        for i, batch in enumerate(testloader):
            if i >= 250:  # Limit for speed
                break
                
            batch = [{k: v.to(device) if isinstance(v, torch.Tensor) else v 
                     for k, v in b.items()} for b in batch]
            
            # Get predictions
            predictions = net.model(batch)
            
            # Extract prediction data
            for pred, gt in zip(predictions, batch):
                all_predictions.append(extract_predictions(pred))
                all_ground_truth.append(extract_ground_truth(gt))
    
    # Compute base metrics (default confidence)
    base_metrics = evaluator.evaluate(all_predictions, all_ground_truth)
    
    # Compute confidence-based metrics
    confidence_thresholds = [round(x * 0.05, 2) for x in range(21)]  # 0.0 to 1.0 in 0.05 steps
    confidence_metrics = {}
    
    for threshold in confidence_thresholds:
        # Filter predictions by confidence
        filtered_preds = filter_by_confidence(all_predictions, threshold)
        conf_metrics = evaluator.evaluate(filtered_preds, all_ground_truth)
        confidence_metrics[f"conf_{threshold}"] = conf_metrics
    
    # Combine base metrics with confidence analysis
    final_metrics = {**base_metrics,  "confidence_analysis": confidence_metrics, "total_predictions": len(all_predictions), "total_ground_truth": len(all_ground_truth)}
    
    return final_metrics


