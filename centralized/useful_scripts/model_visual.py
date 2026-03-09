"""

this code visualizes the architecture of the DiffusionDet model using Detectron2's configuration system. 
higher level structure is given in the thesis report.
"""

import torch
from detectron2.config import get_cfg
from detectron2.modeling import build_model
import os
import sys

# Add paths
sys.path.append('/mimer/NOBACKUP/groups/naiss2024-5-153/old_projects/Berhane/labelled_kitti/centralized')
from diffusiondet import add_diffusiondet_config
from diffusiondet.util.model_ema import add_model_ema_configs

def get_layer_info(module):
    """Extract detailed layer information"""
    info = {}
    if hasattr(module, 'in_channels'):
        info['in_channels'] = module.in_channels
    if hasattr(module, 'out_channels'):
        info['out_channels'] = module.out_channels
    if hasattr(module, 'kernel_size'):
        info['kernel_size'] = module.kernel_size
    if hasattr(module, 'stride'):
        info['stride'] = module.stride
    if hasattr(module, 'padding'):
        info['padding'] = module.padding
    if hasattr(module, 'num_heads'):
        info['num_heads'] = module.num_heads
    if hasattr(module, 'embed_dim'):
        info['embed_dim'] = module.embed_dim
    if hasattr(module, 'output_size'):
        info['output_size'] = module.output_size
    return info

def print_detailed_structure(model, prefix="", max_depth=3, current_depth=0):
    """Print model structure with layer details"""
    if current_depth >= max_depth:
        return
    
    for name, module in model.named_children():
        info = get_layer_info(module)
        info_str = ""
        if info:
            info_parts = []
            if 'out_channels' in info:
                info_parts.append(f"out={info['out_channels']}")
            if 'kernel_size' in info:
                info_parts.append(f"k={info['kernel_size']}")
            if 'stride' in info:
                info_parts.append(f"s={info['stride']}")
            if 'num_heads' in info:
                info_parts.append(f"heads={info['num_heads']}")
            if 'embed_dim' in info:
                info_parts.append(f"dim={info['embed_dim']}")
            if 'output_size' in info:
                info_parts.append(f"size={info['output_size']}")
            if info_parts:
                info_str = f" ({', '.join(info_parts)})"
        
        print(f"{prefix}├── {name}: {type(module).__name__}{info_str}")
        if hasattr(module, 'named_children') and list(module.named_children()):
            print_detailed_structure(module, prefix + "│   ", max_depth, current_depth + 1)

def analyze_model_dimensions(model, cfg):
    """Analyze model with dummy input to get dimensions"""
    print("\nModel Dimension Analysis:")
    print("=" * 50)
    
    # Get input size from config
    input_size = cfg.INPUT.MIN_SIZE_TEST if hasattr(cfg.INPUT, 'MIN_SIZE_TEST') else 800
    print(f"Input size: {input_size} x {input_size}")
    
    # Create dummy input
    dummy_input = torch.randn(1, 3, input_size, input_size)
    
    model.eval()
    with torch.no_grad():
        try:
            # Forward pass to get feature dimensions
            features = model.backbone(dummy_input)
            print(f"\nBackbone output feature maps:")
            for level, feat in features.items():
                print(f"  {level}: {feat.shape}")
            
            # Analyze FPN if available
            if hasattr(model, 'neck'):
                neck_features = model.neck(features)
                print(f"\nFPN output feature maps:")
                for level, feat in neck_features.items():
                    print(f"  {level}: {feat.shape}")
                    
        except Exception as e:
            print(f"Forward pass failed: {e}")
    
    return input_size

def extract_config_info(cfg):
    """Extract relevant configuration information"""
    print("\nConfiguration Details:")
    print("=" * 50)
    
    # Try different possible config paths
    try:
        if hasattr(cfg.MODEL, 'NUM_CLASSES'):
            print(f"Number of classes: {cfg.MODEL.NUM_CLASSES}")
    except:
        pass
    
    # Try different DiffusionDet config paths
    try:
        if hasattr(cfg.MODEL, 'DIFFUSIONDET') and hasattr(cfg.MODEL.DIFFUSIONDET, 'NUM_PROPOSALS'):
            print(f"Number of proposals: {cfg.MODEL.DIFFUSIONDET.NUM_PROPOSALS}")
        elif hasattr(cfg.MODEL, 'NUM_PROPOSALS'):
            print(f"Number of proposals: {cfg.MODEL.NUM_PROPOSALS}")
    except:
        pass
    
    try:
        if hasattr(cfg.MODEL, 'DIFFUSIONDET') and hasattr(cfg.MODEL.DIFFUSIONDET, 'SAMPLING_TIMESTEPS'):
            print(f"Sampling timesteps: {cfg.MODEL.DIFFUSIONDET.SAMPLING_TIMESTEPS}")
        elif hasattr(cfg.MODEL, 'SAMPLING_TIMESTEPS'):
            print(f"Sampling timesteps: {cfg.MODEL.SAMPLING_TIMESTEPS}")
    except:
        pass
    
    try:
        if hasattr(cfg.MODEL, 'ROI_HEADS') and hasattr(cfg.MODEL.ROI_HEADS, 'POOLER_RESOLUTION'):
            print(f"ROI pooler resolution: {cfg.MODEL.ROI_HEADS.POOLER_RESOLUTION}")
    except:
        pass
    
    try:
        if hasattr(cfg.MODEL, 'FPN') and hasattr(cfg.MODEL.FPN, 'OUT_CHANNELS'):
            print(f"FPN output channels: {cfg.MODEL.FPN.OUT_CHANNELS}")
    except:
        pass
    
    # Print all available MODEL attributes to debug
    print(f"\nAvailable MODEL attributes:")
    for attr in dir(cfg.MODEL):
        if not attr.startswith('_'):
            print(f"  {attr}")
            
    # Print input size info
    try:
        if hasattr(cfg.INPUT, 'MIN_SIZE_TEST'):
            print(f"Input min size test: {cfg.INPUT.MIN_SIZE_TEST}")
        if hasattr(cfg.INPUT, 'MAX_SIZE_TEST'):
            print(f"Input max size test: {cfg.INPUT.MAX_SIZE_TEST}")
        if hasattr(cfg.INPUT, 'MIN_SIZE_TRAIN'):
            print(f"Input min size train: {cfg.INPUT.MIN_SIZE_TRAIN}")
    except:
        pass

def main():
    cfg = get_cfg()
    add_diffusiondet_config(cfg)
    add_model_ema_configs(cfg)
    cfg.merge_from_file("diffdet_temp_zod/diffdet.zod.res50.yaml")
    cfg.MODEL.DEVICE = 'cpu'
    
    model = build_model(cfg)
    
    print("DiffusionDet Architecture:")
    print("=" * 50)
    print_detailed_structure(model)
    
    # Extract configuration information
    extract_config_info(cfg)
    
    # Analyze dimensions with dummy input
    input_size = analyze_model_dimensions(model, cfg)
    
    print(f"\nModel Summary:")
    print("=" * 50)
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # Detailed head analysis
    if hasattr(model, 'head'):
        print(f"\nDynamic Head Details:")
        print("-" * 30)
        if hasattr(model.head, 'head_series'):
            print(f"Number of RCNN heads: {len(model.head.head_series)}")
            
            # Analyze first head in detail
            if len(model.head.head_series) > 0:
                first_head = model.head.head_series[0]
                print(f"\nFirst RCNN Head Structure:")
                print_detailed_structure(first_head, max_depth=2)
    
    # Print ROI Pooler details
    if hasattr(model, 'head') and hasattr(model.head, 'box_pooler'):
        print(f"\nROI Pooler Details:")
        pooler = model.head.box_pooler
        if hasattr(pooler, 'output_size'):
            print(f"Output size: {pooler.output_size}")
        if hasattr(pooler, 'scales'):
            print(f"Scales: {pooler.scales}")
        if hasattr(pooler, 'sampling_ratio'):
            print(f"Sampling ratio: {pooler.sampling_ratio}")

if __name__ == "__main__":
    main()