#!/usr/bin/env python3
"""
KITTI Dataset Distribution Analyzer
Comprehensive analysis of bbox distributions, aspect ratios, and labels per image
Optimized for A100 GPU with parallel processing
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import torch
from collections import defaultdict, Counter
import time
import argparse
import multiprocessing as mp

# Fix CUDA multiprocessing issues
mp.set_start_method('spawn', force=True)

def process_annotation_batch_worker(args):
    """Standalone worker function for multiprocessing (no CUDA operations)"""
    annotations_batch, class_names = args
    
    batch_stats = {
        'widths': [], 'heights': [], 'areas': [], 'aspect_ratios': [],
        'class_widths': defaultdict(list), 'class_heights': defaultdict(list),
        'class_areas': defaultdict(list), 'class_aspect_ratios': defaultdict(list),
        'class_counts': defaultdict(int), 'image_labels': defaultdict(int)
    }
    
    for ann in annotations_batch:
        bbox = ann['bbox']  # [x, y, width, height]
        x, y, width, height = bbox
        area = width * height
        aspect_ratio = width / height if height > 0 else 0
        
        # Overall statistics
        batch_stats['widths'].append(width)
        batch_stats['heights'].append(height)
        batch_stats['areas'].append(area)
        batch_stats['aspect_ratios'].append(aspect_ratio)
        
        # Labels per image
        batch_stats['image_labels'][ann['image_id']] += 1
        
        # Class-specific statistics
        class_id = ann['category_id'] - 1  # Convert to 0-based
        if 0 <= class_id < len(class_names):
            class_name = class_names[class_id]
            batch_stats['class_widths'][class_name].append(width)
            batch_stats['class_heights'][class_name].append(height)
            batch_stats['class_areas'][class_name].append(area)
            batch_stats['class_aspect_ratios'][class_name].append(aspect_ratio)
            batch_stats['class_counts'][class_name] += 1
    
    return batch_stats


class KITTIAnalyzer:
    """Main analyzer class for KITTI dataset distributions"""
    
    def __init__(self, output_dir="./kitti_analysis", max_workers=8):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        self.max_workers = max_workers
        
        # Don't setup GPU in main process for multiprocessing compatibility
        self.workspace = None
        
        # KITTI-specific configurations
        self.kitti_classes = ["Car", "Van", "Truck", "Pedestrian", "Person_sitting", "Cyclist", "Tram"]
        self.anchor_ratios = [0.1, 0.5, 1.0, 2.0, 4.0]  # Current DiffusionDet anchors
        self.anchor_sizes = [32, 64, 128, 256, 512, 1024]  # Current anchor sizes
        
        # Training config reference
        self.train_config = {
            'min_sizes': [512, 576, 640, 704, 768, 832, 896, 960, 1024],
            'max_size': 1500,
            'test_size': 800,
            'detections_per_image': 120
        }
    
    def setup_gpu(self):
        """Setup GPU workspace to maintain utilization - called only when needed"""
        if torch.cuda.is_available():
            device = torch.device("cuda")
            workspace = torch.randn(4000, 4000, dtype=torch.float32).to(device)
            print(f"[GPU] Using CUDA device: {torch.cuda.get_device_name()}")
            print(f"[GPU] Memory allocated: {torch.cuda.memory_allocated()/1024**3:.1f}GB")
            return workspace
        return None
    
    def merge_batch_stats(self, all_batch_stats):
        """Merge statistics from all batches"""
        merged = {
            'widths': [], 'heights': [], 'areas': [], 'aspect_ratios': [],
            'class_widths': defaultdict(list), 'class_heights': defaultdict(list),
            'class_areas': defaultdict(list), 'class_aspect_ratios': defaultdict(list),
            'class_counts': defaultdict(int), 'image_labels': defaultdict(int)
        }
        
        for batch_stats in all_batch_stats:
            # Merge simple lists
            for key in ['widths', 'heights', 'areas', 'aspect_ratios']:
                merged[key].extend(batch_stats[key])
            
            # Merge class-specific data
            for class_key in ['class_widths', 'class_heights', 'class_areas', 'class_aspect_ratios']:
                for class_name, values in batch_stats[class_key].items():
                    merged[class_key][class_name].extend(values)
            
            # Merge counts
            for class_name, count in batch_stats['class_counts'].items():
                merged['class_counts'][class_name] += count
            
            for img_id, count in batch_stats['image_labels'].items():
                merged['image_labels'][img_id] += count
        
        return merged
    
    def analyze_coco_file(self, coco_file, batch_size=10000):
        """Analyze COCO file with parallel processing"""
        print(f"[ANALYZE] Loading {coco_file}...")
        
        with open(coco_file, 'r') as f:
            data = json.load(f)
        
        annotations = data['annotations']
        categories = data['categories']
        images = data['images']
        
        # Create class name mapping
        class_names = [cat['name'] for cat in sorted(categories, key=lambda x: x['id'])]
        
        print(f"[ANALYZE] Processing {len(annotations):,} annotations, {len(images):,} images")
        print(f"[ANALYZE] Classes: {class_names}")
        
        # Split into batches for parallel processing
        annotation_batches = []
        for i in range(0, len(annotations), batch_size):
            batch = annotations[i:i + batch_size]
            annotation_batches.append((batch, class_names))
        
        # Process batches in parallel using the standalone worker
        all_batch_stats = []
        start_time = time.time()
        
        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_batch = {executor.submit(process_annotation_batch_worker, batch): i 
                              for i, batch in enumerate(annotation_batches)}
            
            for i, future in enumerate(as_completed(future_to_batch)):
                batch_stats = future.result()
                all_batch_stats.append(batch_stats)
                
                if (i + 1) % 5 == 0:
                    elapsed = time.time() - start_time
                    print(f"[PROGRESS] Processed {i + 1}/{len(annotation_batches)} batches ({elapsed:.1f}s)")
        
        # Merge results
        final_stats = self.merge_batch_stats(all_batch_stats)
        
        # Process labels per image (include images with 0 labels)
        image_ids = {img['id'] for img in images}
        labels_per_image = np.array([final_stats['image_labels'].get(img_id, 0) for img_id in image_ids])
        final_stats['labels_per_image'] = labels_per_image
        final_stats['total_images'] = len(images)
        
        processing_time = time.time() - start_time
        print(f"[COMPLETE] Analysis finished in {processing_time:.1f}s")
        
        return final_stats, class_names
    
    def create_overall_distributions(self, train_stats, val_stats):
        """Create overall bbox distribution plots"""
        print(f"[PLOTS] Creating overall distribution plots...")
        
        # Calculate totals
        total_train_ann = len(train_stats['widths'])
        total_val_ann = len(val_stats['widths'])
        total_annotations = total_train_ann + total_val_ann
        
        # Combine all data
        all_widths = train_stats['widths'] + val_stats['widths']
        all_heights = train_stats['heights'] + val_stats['heights']
        all_ratios = train_stats['aspect_ratios'] + val_stats['aspect_ratios']
        all_diagonals = np.sqrt(np.array(all_widths)**2 + np.array(all_heights)**2)
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. Width distribution
        ax1.hist(all_widths, bins=80, alpha=0.7, color='blue', density=True)
        mean_w = np.mean(all_widths)
        median_w = np.median(all_widths)
        p95_w = np.percentile(all_widths, 95)
        
        ax1.axvline(mean_w, color='red', linestyle='--', label=f'Mean: {mean_w:.1f}')
        ax1.axvline(median_w, color='green', linestyle='--', label=f'Median: {median_w:.1f}')
        ax1.axvline(p95_w, color='orange', linestyle='--', label=f'95%: {p95_w:.1f}')
        
        ax1.set_xlabel('Bbox Width (pixels)')
        ax1.set_ylabel('Density')
        ax1.set_title(f'KITTI: Width Distribution ({total_annotations:,} annotations)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Height distribution
        ax2.hist(all_heights, bins=80, alpha=0.7, color='orange', density=True)
        mean_h = np.mean(all_heights)
        median_h = np.median(all_heights)
        p95_h = np.percentile(all_heights, 95)
        
        ax2.axvline(mean_h, color='red', linestyle='--', label=f'Mean: {mean_h:.1f}')
        ax2.axvline(median_h, color='green', linestyle='--', label=f'Median: {median_h:.1f}')
        ax2.axvline(p95_h, color='orange', linestyle='--', label=f'95%: {p95_h:.1f}')
        
        ax2.set_xlabel('Bbox Height (pixels)')
        ax2.set_ylabel('Density')
        ax2.set_title(f'KITTI: Height Distribution ({total_annotations:,} annotations)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Aspect ratio distribution
        ax3.hist(all_ratios, bins=100, alpha=0.7, color='purple', density=True)
        
        # Add current anchor ratios
        for ratio in self.anchor_ratios:
            ax3.axvline(ratio, color='red', linestyle='--', alpha=0.7, linewidth=2)
            ax3.text(ratio, ax3.get_ylim()[1] * 0.9, f'{ratio}', ha='center', fontsize=10, 
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        ax3.set_xlabel('Aspect Ratio (width/height)')
        ax3.set_ylabel('Density')
        ax3.set_title(f'KITTI: Aspect Ratio Distribution ({total_annotations:,} annotations)')
        ax3.grid(True, alpha=0.3)
        ax3.set_xlim(0, min(10, np.percentile(all_ratios, 99)))
        
        # 4. Diagonal distribution
        ax4.hist(all_diagonals, bins=100, alpha=0.7, color='green', density=True)
        
        # Add current anchor sizes
        for size in self.anchor_sizes:
            ax4.axvline(size, color='red', linestyle='--', alpha=0.7, linewidth=2)
            ax4.text(size, ax4.get_ylim()[1] * 0.9, f'{size}', rotation=90, 
                    ha='right', fontsize=10, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        ax4.set_xlabel('Diagonal Size (pixels)')
        ax4.set_ylabel('Density')
        ax4.set_title(f'KITTI: Diagonal Size Distribution ({total_annotations:,} annotations)')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'kitti_overall_distributions.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def create_class_wise_plots(self, train_stats, val_stats, train_classes):
        """Create class-wise distribution plots (separate subplots)"""
        print(f"[PLOTS] Creating class-wise distribution plots...")
        
        total_annotations = len(train_stats['widths']) + len(val_stats['widths'])
        n_classes = len(train_classes)
        
        # 1. Width distributions by class
        fig, axes = plt.subplots((n_classes + 1) // 2, 2, figsize=(16, 4 * ((n_classes + 1) // 2)))
        axes = axes.flatten() if n_classes > 2 else [axes] if n_classes == 1 else axes
        
        for i, class_name in enumerate(train_classes):
            if i >= len(axes):
                break
                
            ax = axes[i]
            class_widths = (train_stats['class_widths'].get(class_name, []) + 
                           val_stats['class_widths'].get(class_name, []))
            
            if class_widths:
                ax.hist(class_widths, bins=50, alpha=0.7, color=plt.cm.Set3(i), density=True)
                mean_w = np.mean(class_widths)
                median_w = np.median(class_widths)
                ax.axvline(mean_w, color='red', linestyle='--', alpha=0.8)
                ax.axvline(median_w, color='green', linestyle='--', alpha=0.8)
                
                stats_text = f'Count: {len(class_widths):,}\nMean: {mean_w:.1f}\nMedian: {median_w:.1f}'
                ax.text(0.98, 0.98, stats_text, transform=ax.transAxes, 
                       verticalalignment='top', horizontalalignment='right', fontsize=9,
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            ax.set_xlabel('Width (pixels)')
            ax.set_ylabel('Density')
            ax.set_title(f'{class_name}: Width ({len(class_widths):,} annotations)')
            ax.grid(True, alpha=0.3)
        
        for i in range(n_classes, len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'kitti_width_by_class.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Height distributions by class
        fig, axes = plt.subplots((n_classes + 1) // 2, 2, figsize=(16, 4 * ((n_classes + 1) // 2)))
        axes = axes.flatten() if n_classes > 2 else [axes] if n_classes == 1 else axes
        
        for i, class_name in enumerate(train_classes):
            if i >= len(axes):
                break
                
            ax = axes[i]
            class_heights = (train_stats['class_heights'].get(class_name, []) + 
                            val_stats['class_heights'].get(class_name, []))
            
            if class_heights:
                ax.hist(class_heights, bins=50, alpha=0.7, color=plt.cm.Set3(i), density=True)
                mean_h = np.mean(class_heights)
                median_h = np.median(class_heights)
                ax.axvline(mean_h, color='red', linestyle='--', alpha=0.8)
                ax.axvline(median_h, color='green', linestyle='--', alpha=0.8)
                
                stats_text = f'Count: {len(class_heights):,}\nMean: {mean_h:.1f}\nMedian: {median_h:.1f}'
                ax.text(0.98, 0.98, stats_text, transform=ax.transAxes, 
                       verticalalignment='top', horizontalalignment='right', fontsize=9,
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            ax.set_xlabel('Height (pixels)')
            ax.set_ylabel('Density')
            ax.set_title(f'{class_name}: Height ({len(class_heights):,} annotations)')
            ax.grid(True, alpha=0.3)
        
        for i in range(n_classes, len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'kitti_height_by_class.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. Aspect ratio distributions by class
        fig, axes = plt.subplots((n_classes + 1) // 2, 2, figsize=(16, 4 * ((n_classes + 1) // 2)))
        axes = axes.flatten() if n_classes > 2 else [axes] if n_classes == 1 else axes
        
        for i, class_name in enumerate(train_classes):
            if i >= len(axes):
                break
                
            ax = axes[i]
            class_ratios = (train_stats['class_aspect_ratios'].get(class_name, []) + 
                           val_stats['class_aspect_ratios'].get(class_name, []))
            
            if class_ratios:
                ax.hist(class_ratios, bins=50, alpha=0.7, color=plt.cm.Set3(i), density=True)
                
                # Add anchor ratio lines
                for ratio in self.anchor_ratios:
                    ax.axvline(ratio, color='red', linestyle='--', alpha=0.5, linewidth=1)
                
                thin_count = np.sum(np.array(class_ratios) < 0.5)
                wide_count = np.sum(np.array(class_ratios) > 2.0)
                stats_text = f'Count: {len(class_ratios):,}\nThin(<0.5): {thin_count}\nWide(>2.0): {wide_count}'
                ax.text(0.98, 0.98, stats_text, transform=ax.transAxes, 
                       verticalalignment='top', horizontalalignment='right', fontsize=9,
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            ax.set_xlabel('Aspect Ratio (width/height)')
            ax.set_ylabel('Density')
            ax.set_title(f'{class_name}: Aspect Ratio ({len(class_ratios):,} annotations)')
            ax.grid(True, alpha=0.3)
            ax.set_xlim(0, min(8, np.percentile(class_ratios, 99) if class_ratios else 8))
        
        for i in range(n_classes, len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'kitti_aspect_ratio_by_class.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 4. Diagonal distributions by class
        fig, axes = plt.subplots((n_classes + 1) // 2, 2, figsize=(16, 4 * ((n_classes + 1) // 2)))
        axes = axes.flatten() if n_classes > 2 else [axes] if n_classes == 1 else axes
        
        for i, class_name in enumerate(train_classes):
            if i >= len(axes):
                break
                
            ax = axes[i]
            class_widths = (train_stats['class_widths'].get(class_name, []) + 
                           val_stats['class_widths'].get(class_name, []))
            class_heights = (train_stats['class_heights'].get(class_name, []) + 
                            val_stats['class_heights'].get(class_name, []))
            
            if class_widths and class_heights:
                class_diagonals = np.sqrt(np.array(class_widths)**2 + np.array(class_heights)**2)
                ax.hist(class_diagonals, bins=50, alpha=0.7, color=plt.cm.Set3(i), density=True)
                
                # Add anchor size lines
                for size in self.anchor_sizes:
                    ax.axvline(size, color='red', linestyle='--', alpha=0.5, linewidth=1)
                
                mean_d = np.mean(class_diagonals)
                median_d = np.median(class_diagonals)
                p95_d = np.percentile(class_diagonals, 95)
                stats_text = f'Count: {len(class_diagonals):,}\nMean: {mean_d:.1f}\nMedian: {median_d:.1f}\n95%: {p95_d:.1f}'
                ax.text(0.98, 0.98, stats_text, transform=ax.transAxes, 
                       verticalalignment='top', horizontalalignment='right', fontsize=9,
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            ax.set_xlabel('Diagonal Size (pixels)')
            ax.set_ylabel('Density')
            ax.set_title(f'{class_name}: Diagonal ({len(class_widths):,} annotations)')
            ax.grid(True, alpha=0.3)
        
        for i in range(n_classes, len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'kitti_diagonal_by_class.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def create_overlayed_plots(self, train_stats, val_stats, train_classes):
        """Create overlayed distribution plots (all classes on same axes)"""
        print(f"[PLOTS] Creating overlayed distribution plots...")
        
        total_annotations = len(train_stats['widths']) + len(val_stats['widths'])
        colors = plt.cm.Set3(np.linspace(0, 1, len(train_classes)))
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 16))
        
        # 1. Width overlay
        for i, (class_name, color) in enumerate(zip(train_classes, colors)):
            class_widths = (train_stats['class_widths'].get(class_name, []) + 
                           val_stats['class_widths'].get(class_name, []))
            if len(class_widths) > 10:
                ax1.hist(class_widths, bins=80, alpha=0.6, label=f'{class_name} ({len(class_widths):,})', 
                        density=True, color=color)
        
        ax1.set_xlabel('Width (pixels)')
        ax1.set_ylabel('Density')
        ax1.set_title(f'KITTI: Width Distributions by Class (Overlayed)')
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax1.grid(True, alpha=0.3)
        
        # 2. Height overlay
        for i, (class_name, color) in enumerate(zip(train_classes, colors)):
            class_heights = (train_stats['class_heights'].get(class_name, []) + 
                            val_stats['class_heights'].get(class_name, []))
            if len(class_heights) > 10:
                ax2.hist(class_heights, bins=80, alpha=0.6, label=f'{class_name} ({len(class_heights):,})', 
                        density=True, color=color)
        
        ax2.set_xlabel('Height (pixels)')
        ax2.set_ylabel('Density')
        ax2.set_title(f'KITTI: Height Distributions by Class (Overlayed)')
        ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax2.grid(True, alpha=0.3)
        
        # 3. Aspect ratio overlay
        for i, (class_name, color) in enumerate(zip(train_classes, colors)):
            class_ratios = (train_stats['class_aspect_ratios'].get(class_name, []) + 
                           val_stats['class_aspect_ratios'].get(class_name, []))
            if len(class_ratios) > 10:
                ax3.hist(class_ratios, bins=80, alpha=0.6, label=f'{class_name} ({len(class_ratios):,})', 
                        density=True, color=color)
        
        # Add anchor ratio lines
        for ratio in self.anchor_ratios:
            ax3.axvline(ratio, color='black', linestyle='--', alpha=0.8, linewidth=2)
            ax3.text(ratio, ax3.get_ylim()[1] * 0.9, f'{ratio}', ha='center', fontweight='bold')
        
        ax3.set_xlabel('Aspect Ratio (width/height)')
        ax3.set_ylabel('Density')
        ax3.set_title(f'KITTI: Aspect Ratio Distributions by Class (Overlayed)')
        ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax3.grid(True, alpha=0.3)
        ax3.set_xlim(0, 8)
        
        # 4. Diagonal overlay
        for i, (class_name, color) in enumerate(zip(train_classes, colors)):
            class_widths = (train_stats['class_widths'].get(class_name, []) + 
                           val_stats['class_widths'].get(class_name, []))
            class_heights = (train_stats['class_heights'].get(class_name, []) + 
                            val_stats['class_heights'].get(class_name, []))
            if len(class_widths) > 10 and len(class_heights) > 10:
                class_diagonals = np.sqrt(np.array(class_widths)**2 + np.array(class_heights)**2)
                ax4.hist(class_diagonals, bins=80, alpha=0.6, label=f'{class_name} ({len(class_diagonals):,})', 
                        density=True, color=color)
        
        # Add anchor size lines
        for size in self.anchor_sizes:
            ax4.axvline(size, color='black', linestyle='--', alpha=0.8, linewidth=2)
            ax4.text(size, ax4.get_ylim()[1] * 0.9, f'{size}', rotation=90, ha='right', fontweight='bold')
        
        ax4.set_xlabel('Diagonal Size (pixels)')
        ax4.set_ylabel('Density')
        ax4.set_title(f'KITTI: Diagonal Size Distributions by Class (Overlayed)')
        ax4.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'kitti_distributions_overlayed.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def create_stacked_absolute_density_plots(self, train_stats, val_stats, train_classes):
        """Create stacked absolute density plots"""
        print(f"[PLOTS] Creating stacked absolute density plots...")
        
        total_annotations = len(train_stats['widths']) + len(val_stats['widths'])
        colors = plt.cm.Set3(np.linspace(0, 1, len(train_classes)))
        
        # Prepare data
        all_widths = train_stats['widths'] + val_stats['widths']
        all_heights = train_stats['heights'] + val_stats['heights']
        all_ratios = train_stats['aspect_ratios'] + val_stats['aspect_ratios']
        all_diagonals = np.sqrt(np.array(all_widths)**2 + np.array(all_heights)**2)
        
        # Create bins
        width_bins = np.linspace(0, np.percentile(all_widths, 99), 80)
        height_bins = np.linspace(0, np.percentile(all_heights, 99), 80)
        ratio_bins = np.linspace(0, min(8, np.percentile(all_ratios, 99)), 80)
        diagonal_bins = np.linspace(0, np.percentile(all_diagonals, 99), 80)
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 16))
        
        # 1. Width absolute density stacked
        bottom_w = np.zeros(len(width_bins)-1)
        for i, (class_name, color) in enumerate(zip(train_classes, colors)):
            class_widths = (train_stats['class_widths'].get(class_name, []) + 
                           val_stats['class_widths'].get(class_name, []))
            if len(class_widths) > 10:
                counts, _ = np.histogram(class_widths, bins=width_bins)
                bin_widths = np.diff(width_bins)
                absolute_density = counts / (total_annotations * bin_widths)

                bin_centers = (width_bins[:-1] + width_bins[1:]) / 2
                ax1.bar(bin_centers, absolute_density, bottom=bottom_w, 
                       label=f'{class_name} ({len(class_widths):,})', color=color, alpha=0.8, 
                       width=bin_widths, align='center')
                bottom_w += absolute_density
        
        ax1.set_xlabel('Width (pixels)')
        ax1.set_ylabel('Absolute Density (relative to total dataset)')
        ax1.set_title(f'KITTI: Width Absolute Density (Stacked)\nTotal: {total_annotations:,} annotations')
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax1.grid(True, alpha=0.3)
        
        # 2. Height absolute density stacked
        bottom_h = np.zeros(len(height_bins)-1)
        for i, (class_name, color) in enumerate(zip(train_classes, colors)):
            class_heights = (train_stats['class_heights'].get(class_name, []) + 
                            val_stats['class_heights'].get(class_name, []))
            if len(class_heights) > 10:
                counts, _ = np.histogram(class_heights, bins=height_bins)
                bin_widths = np.diff(height_bins)
                absolute_density = counts / (total_annotations * bin_widths)
                
                bin_centers = (height_bins[:-1] + height_bins[1:]) / 2
                ax2.bar(bin_centers, absolute_density, bottom=bottom_h, 
                       label=f'{class_name} ({len(class_heights):,})', color=color, alpha=0.8, 
                       width=bin_widths, align='center')
                bottom_h += absolute_density
        
        ax2.set_xlabel('Height (pixels)')
        ax2.set_ylabel('Absolute Density (relative to total dataset)')
        ax2.set_title(f'KITTI: Height Absolute Density (Stacked)\nTotal: {total_annotations:,} annotations')
        ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax2.grid(True, alpha=0.3)
        
        # 3. Aspect ratio absolute density stacked
        bottom_r = np.zeros(len(ratio_bins)-1)
        for i, (class_name, color) in enumerate(zip(train_classes, colors)):
            class_ratios = (train_stats['class_aspect_ratios'].get(class_name, []) + 
                           val_stats['class_aspect_ratios'].get(class_name, []))
            if len(class_ratios) > 10:
                counts, _ = np.histogram(class_ratios, bins=ratio_bins)
                bin_widths = np.diff(ratio_bins)
                absolute_density = counts / (total_annotations * bin_widths)
                
                bin_centers = (ratio_bins[:-1] + ratio_bins[1:]) / 2
                ax3.bar(bin_centers, absolute_density, bottom=bottom_r, 
                       label=f'{class_name} ({len(class_ratios):,})', color=color, alpha=0.8, 
                       width=bin_widths, align='center')
                bottom_r += absolute_density
        
        # Add anchor ratio lines
        for ratio in self.anchor_ratios:
            ax3.axvline(ratio, color='black', linestyle='--', alpha=0.8, linewidth=2)
            ax3.text(ratio, ax3.get_ylim()[1] * 0.95, f'{ratio}', ha='center', fontweight='bold')
        
        ax3.set_xlabel('Aspect Ratio (width/height)')
        ax3.set_ylabel('Absolute Density (relative to total dataset)')
        ax3.set_title(f'KITTI: Aspect Ratio Absolute Density (Stacked)\nTotal: {total_annotations:,} annotations')
        ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax3.grid(True, alpha=0.3)
        
        # 4. Diagonal absolute density stacked
        bottom_d = np.zeros(len(diagonal_bins)-1)
        for i, (class_name, color) in enumerate(zip(train_classes, colors)):
            class_widths = (train_stats['class_widths'].get(class_name, []) + 
                           val_stats['class_widths'].get(class_name, []))
            class_heights = (train_stats['class_heights'].get(class_name, []) + 
                            val_stats['class_heights'].get(class_name, []))
            if len(class_widths) > 10 and len(class_heights) > 10:
                class_diagonals = np.sqrt(np.array(class_widths)**2 + np.array(class_heights)**2)
                counts, _ = np.histogram(class_diagonals, bins=diagonal_bins)
                bin_widths = np.diff(diagonal_bins)
                absolute_density = counts / (total_annotations * bin_widths)
                
                bin_centers = (diagonal_bins[:-1] + diagonal_bins[1:]) / 2
                ax4.bar(bin_centers, absolute_density, bottom=bottom_d, 
                       label=f'{class_name} ({len(class_diagonals):,})', color=color, alpha=0.8, 
                       width=bin_widths, align='center')
                bottom_d += absolute_density
        
        # Add anchor size lines
        for size in self.anchor_sizes:
            ax4.axvline(size, color='black', linestyle='--', alpha=0.8, linewidth=2)
            ax4.text(size, ax4.get_ylim()[1] * 0.95, f'{size}', rotation=90, ha='right', fontweight='bold')
        
        ax4.set_xlabel('Diagonal Size (pixels)')
        ax4.set_ylabel('Absolute Density (relative to total dataset)')
        ax4.set_title(f'KITTI: Diagonal Absolute Density (Stacked)\nTotal: {total_annotations:,} annotations')
        ax4.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'kitti_stacked_absolute_density.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def create_image_size_analysis(self, train_file, val_file):
        """Create image size distribution analysis for input size optimization"""
        print(f"[PLOTS] Creating image size analysis...")
        
        # Load image dimensions from both files
        all_image_dims = []
        
        for file_path, split_name in [(train_file, 'train'), (val_file, 'val')]:
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            for img in data['images']:
                all_image_dims.append({
                    'width': img['width'],
                    'height': img['height'],
                    'split': split_name
                })
        
        widths = np.array([img['width'] for img in all_image_dims])
        heights = np.array([img['height'] for img in all_image_dims])
        
        # Calculate edges and aspect ratios
        shortest_edges = np.minimum(widths, heights)
        longest_edges = np.maximum(widths, heights)
        aspect_ratios = widths / heights
        
        print(f"[IMAGE_SIZE] Analyzing {len(all_image_dims):,} images")
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. Shortest edge distribution (for MIN_SIZE_TRAIN)
        ax1.hist(shortest_edges, bins=50, alpha=0.7, color='blue', density=True)
        
        # Add training size references
        for size in self.train_config['min_sizes'][::2]:  # Show every other
            ax1.axvline(size, color='red', linestyle='--', alpha=0.6, linewidth=1)
        ax1.axvline(self.train_config['test_size'], color='green', linestyle='-', linewidth=2, 
                   label=f'TEST: {self.train_config["test_size"]}')
        
        ax1.set_xlabel('Shortest Edge (pixels)')
        ax1.set_ylabel('Density')
        ax1.set_title(f'KITTI: Image Shortest Edge vs MIN_SIZE_TRAIN\n{len(all_image_dims):,} images')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Add statistics
        mean_short = shortest_edges.mean()
        median_short = np.median(shortest_edges)
        ax1.text(0.02, 0.98, f'Mean: {mean_short:.0f}\nMedian: {median_short:.0f}', 
                transform=ax1.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # 2. Longest edge distribution (for MAX_SIZE_TRAIN)
        ax2.hist(longest_edges, bins=50, alpha=0.7, color='orange', density=True)
        ax2.axvline(self.train_config['max_size'], color='red', linestyle='-', linewidth=2, 
                   label=f'MAX_TRAIN: {self.train_config["max_size"]}')
        
        ax2.set_xlabel('Longest Edge (pixels)')
        ax2.set_ylabel('Density')
        ax2.set_title(f'KITTI: Image Longest Edge vs MAX_SIZE_TRAIN')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Add statistics
        mean_long = longest_edges.mean()
        median_long = np.median(longest_edges)
        ax2.text(0.02, 0.98, f'Mean: {mean_long:.0f}\nMedian: {median_long:.0f}', 
                transform=ax2.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # 3. Image aspect ratio distribution
        ax3.hist(aspect_ratios, bins=50, alpha=0.7, color='purple', density=True)
        ax3.set_xlabel('Image Aspect Ratio (width/height)')
        ax3.set_ylabel('Density')
        ax3.set_title(f'KITTI: Image Aspect Ratios')
        ax3.grid(True, alpha=0.3)
        
        # Add statistics
        mean_ar = aspect_ratios.mean()
        median_ar = np.median(aspect_ratios)
        ax3.text(0.02, 0.98, f'Mean: {mean_ar:.2f}\nMedian: {median_ar:.2f}', 
                transform=ax3.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # 4. Training size coverage analysis
        too_small = np.sum(shortest_edges < self.train_config['min_sizes'][0])
        well_covered = np.sum((shortest_edges >= self.train_config['min_sizes'][0]) & 
                             (longest_edges <= self.train_config['max_size']))
        too_large = np.sum(longest_edges > self.train_config['max_size'])
        
        coverage_data = [too_small, well_covered, too_large]
        labels = ['Too Small\n(<512px)', 'Well Covered', 'Too Large\n(>1500px)']
        colors = ['red', 'green', 'orange']
        
        bars = ax4.bar(labels, coverage_data, color=colors, alpha=0.7)
        ax4.set_ylabel('Image Count')
        ax4.set_title('KITTI: Training Size Coverage Analysis')
        
        # Add percentage labels
        total_images = len(all_image_dims)
        for bar, count in zip(bars, coverage_data):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + total_images*0.01,
                    f'{count:,}\n({count/total_images*100:.1f}%)', 
                    ha='center', va='bottom', fontweight='bold')
        
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'kitti_image_size_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        return {
            'shortest_edges': shortest_edges,
            'longest_edges': longest_edges,
            'aspect_ratios': aspect_ratios,
            'coverage': {'too_small': too_small, 'well_covered': well_covered, 'too_large': too_large}
        }
    
    def create_labels_per_image_plots(self, train_stats, val_stats):
        """Create labels per image analysis"""
        print(f"[PLOTS] Creating labels per image plots...")
        
        train_labels = train_stats['labels_per_image']
        val_labels = val_stats['labels_per_image']
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 10))
        
        # 1. Distribution comparison
        max_labels = max(train_labels.max(), val_labels.max())
        bins = np.arange(0, min(max_labels + 1, 101))
        
        ax1.hist(train_labels, bins=bins, alpha=0.7, label=f'Train ({len(train_labels):,} images)', 
                density=True, color='blue')
        ax1.hist(val_labels, bins=bins, alpha=0.7, label=f'Val ({len(val_labels):,} images)', 
                density=True, color='orange')
        
        # Add current DETECTIONS_PER_IMAGE
        current_limit = self.train_config['detections_per_image']
        ax1.axvline(current_limit, color='red', linestyle='--', linewidth=2, 
                   label=f'Current limit: {current_limit}')
        
        ax1.set_xlabel('Labels per Image')
        ax1.set_ylabel('Density')
        ax1.set_title('KITTI: Labels per Image Distribution')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Cumulative distribution
        ax2.hist(train_labels, bins=50, alpha=0.7, cumulative=True, density=True, 
                label='Train', color='blue')
        ax2.hist(val_labels, bins=50, alpha=0.7, cumulative=True, density=True, 
                label='Val', color='orange')
        ax2.axvline(current_limit, color='red', linestyle='--', linewidth=2)
        ax2.set_xlabel('Labels per Image')
        ax2.set_ylabel('Cumulative Density')
        ax2.set_title('KITTI: Cumulative Labels Distribution')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. High-count images
        high_threshold = 50
        train_high = train_labels[train_labels > high_threshold]
        val_high = val_labels[val_labels > high_threshold]
        
        if len(train_high) > 0 or len(val_high) > 0:
            ax3.hist(train_high, bins=20, alpha=0.7, label=f'Train high (>{high_threshold})', color='blue')
            ax3.hist(val_high, bins=20, alpha=0.7, label=f'Val high (>{high_threshold})', color='orange')
            ax3.set_xlabel('Labels per Image')
            ax3.set_ylabel('Count')
            ax3.set_title(f'KITTI: Images with >{high_threshold} Labels')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
        else:
            ax3.text(0.5, 0.5, f'No images with >{high_threshold} labels', 
                    ha='center', va='center', transform=ax3.transAxes)
            ax3.set_title(f'KITTI: Images with >{high_threshold} Labels')
        
        # 4. Coverage analysis
        detection_limits = [20, 30, 40, 50, 75, 100, 120, 150, 200]
        train_coverage = [np.mean(train_labels <= limit) * 100 for limit in detection_limits]
        val_coverage = [np.mean(val_labels <= limit) * 100 for limit in detection_limits]
        
        ax4.plot(detection_limits, train_coverage, 'bo-', label='Train Coverage', linewidth=2)
        ax4.plot(detection_limits, val_coverage, 'ro-', label='Val Coverage', linewidth=2)
        ax4.axvline(current_limit, color='red', linestyle='--', alpha=0.7, 
                   label=f'Current: {current_limit}')
        ax4.set_xlabel('DETECTIONS_PER_IMAGE Limit')
        ax4.set_ylabel('Coverage (%)')
        ax4.set_title('KITTI: Coverage vs DETECTIONS_PER_IMAGE Setting')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'kitti_labels_per_image_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def print_comprehensive_statistics(self, train_stats, val_stats, train_classes):
        """Print comprehensive analysis summary with DiffusionDet recommendations"""
        
        # Calculate totals
        total_train_ann = len(train_stats['widths'])
        total_val_ann = len(val_stats['widths'])
        total_annotations = total_train_ann + total_val_ann
        total_train_img = train_stats['total_images']
        total_val_img = val_stats['total_images']
        
        print(f"\n📊 COMPREHENSIVE KITTI ANALYSIS SUMMARY:")
        print(f"  Total annotations: Train={total_train_ann:,}, Val={total_val_ann:,}, Total={total_annotations:,}")
        print(f"  Total images: Train={total_train_img:,}, Val={total_val_img:,}, Total={total_train_img + total_val_img:,}")
        
        # Class distribution
        print(f"\n🏷️  CLASS ANNOTATION DISTRIBUTION:")
        for class_name in train_classes:
            train_count = train_stats['class_counts'].get(class_name, 0)
            val_count = val_stats['class_counts'].get(class_name, 0)
            total_count = train_count + val_count
            percentage = total_count / total_annotations * 100
            
            print(f"  {class_name:15} - Train: {train_count:6,}, Val: {val_count:5,}, "
                  f"Total: {total_count:7,} ({percentage:5.1f}%)")
        
        # Calculate combined statistics
        all_widths = np.array(train_stats['widths'] + val_stats['widths'])
        all_heights = np.array(train_stats['heights'] + val_stats['heights'])
        all_diagonals = np.sqrt(all_widths**2 + all_heights**2)
        all_ratios = np.array(train_stats['aspect_ratios'] + val_stats['aspect_ratios'])
        
        print(f"\n📐 BBOX SIZE STATISTICS FOR ANCHOR OPTIMIZATION:")
        print(f"  Width  - Mean: {all_widths.mean():.1f}, Median: {np.median(all_widths):.1f}, "
              f"Range: [{all_widths.min():.0f}, {all_widths.max():.0f}]")
        print(f"  Height - Mean: {all_heights.mean():.1f}, Median: {np.median(all_heights):.1f}, "
              f"Range: [{all_heights.min():.0f}, {all_heights.max():.0f}]")
        print(f"  Diagonal - Mean: {all_diagonals.mean():.1f}, Median: {np.median(all_diagonals):.1f}, "
              f"Range: [{all_diagonals.min():.0f}, {all_diagonals.max():.0f}]")
        
        # Aspect ratio coverage analysis
        print(f"\n🎯 ASPECT RATIO COVERAGE (Current anchors: {self.anchor_ratios}):")
        for i, ratio in enumerate(self.anchor_ratios):
            if i == 0:
                covered = np.sum(all_ratios <= ratio)
                range_desc = f"≤{ratio}"
            elif i == len(self.anchor_ratios) - 1:
                covered = np.sum(all_ratios > self.anchor_ratios[i-1])
                range_desc = f">{self.anchor_ratios[i-1]}"
            else:
                covered = np.sum((all_ratios > self.anchor_ratios[i-1]) & (all_ratios <= ratio))
                range_desc = f"{self.anchor_ratios[i-1]}-{ratio}"
            
            coverage_pct = covered / len(all_ratios) * 100
            print(f"  {range_desc:>8}: {covered:7,} annotations ({coverage_pct:5.1f}%)")
        
        # Diagonal size anchor coverage
        print(f"\n📏 DIAGONAL SIZE ANCHOR COVERAGE (Current: {self.anchor_sizes}):")
        for i, size in enumerate(self.anchor_sizes):
            if i == 0:
                covered = np.sum(all_diagonals <= size)
                range_desc = f"≤{size}"
            else:
                covered = np.sum((all_diagonals > self.anchor_sizes[i-1]) & (all_diagonals <= size))
                range_desc = f"{self.anchor_sizes[i-1]}-{size}"
            
            coverage_pct = covered / len(all_diagonals) * 100
            print(f"  {range_desc:>10}: {covered:7,} annotations ({coverage_pct:5.1f}%)")
        
        # Uncovered large objects
        uncovered = np.sum(all_diagonals > self.anchor_sizes[-1])
        uncovered_pct = uncovered / len(all_diagonals) * 100
        print(f"  {'Uncovered':>10}: {uncovered:7,} annotations ({uncovered_pct:5.1f}%)")
        
        # Labels per image analysis
        train_labels = train_stats['labels_per_image']
        val_labels = val_stats['labels_per_image']
        all_labels = np.concatenate([train_labels, val_labels])
        
        print(f"\n📊 LABELS PER IMAGE ANALYSIS:")
        print(f"  Mean: {all_labels.mean():.1f}, Median: {np.median(all_labels):.1f}, "
              f"Max: {all_labels.max()}, 95%: {np.percentile(all_labels, 95):.0f}")
        
        current_limit = self.train_config['detections_per_image']
        exceeded = np.sum(all_labels > current_limit)
        print(f"  Current DETECTIONS_PER_IMAGE: {current_limit}")
        print(f"  Images exceeding limit: {exceeded:,} ({exceeded/len(all_labels)*100:.1f}%)")
        
        # Recommendations
        print(f"\n💡 DIFFUSIONDET OPTIMIZATION RECOMMENDATIONS:")
        
        # Aspect ratio recommendations
        ultra_thin = np.sum(all_ratios < 0.1)
        ultra_wide = np.sum(all_ratios > 4.0)
        if ultra_thin > total_annotations * 0.01:
            print(f"  📐 Add aspect ratio 0.05: {ultra_thin:,} thin objects ({ultra_thin/total_annotations*100:.1f}%)")
        if ultra_wide > total_annotations * 0.01:
            print(f"  📐 Add aspect ratio 6.0: {ultra_wide:,} wide objects ({ultra_wide/total_annotations*100:.1f}%)")
        
        # Anchor size recommendations
        optimal_sizes = [int(np.percentile(all_diagonals, p)) for p in [10, 30, 50, 70, 85, 95]]
        print(f"  📏 Optimal anchor sizes (percentile-based): {optimal_sizes}")
        
        if uncovered_pct > 5:
            rec_large_size = int(np.percentile(all_diagonals, 98))
            print(f"  📏 Add larger anchor: {rec_large_size} (covers {uncovered_pct:.1f}% uncovered objects)")
        
        # DETECTIONS_PER_IMAGE recommendation
        if exceeded > len(all_labels) * 0.05:  # More than 5% exceed
            recommended_limit = int(np.percentile(all_labels, 95))
            print(f"  📊 Recommended DETECTIONS_PER_IMAGE: {max(recommended_limit, 150)}")
        else:
            print(f"  ✅ Current DETECTIONS_PER_IMAGE setting is adequate")
    
    def save_detailed_statistics(self, train_stats, val_stats, train_classes):
        """Save detailed statistics to JSON file"""
        print(f"[SAVE] Saving detailed statistics...")
        
        # Calculate additional metrics
        train_diag = np.sqrt(np.array(train_stats['widths'])**2 + np.array(train_stats['heights'])**2)
        val_diag = np.sqrt(np.array(val_stats['widths'])**2 + np.array(val_stats['heights'])**2)
        all_diag = np.concatenate([train_diag, val_diag])
        
        detailed_stats = {
            "dataset": "KITTI",
            "analysis_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "train": {
                "total_annotations": len(train_stats['widths']),
                "total_images": train_stats['total_images'],
                "bbox_stats": {
                    "width": {
                        "mean": float(np.mean(train_stats['widths'])),
                        "median": float(np.median(train_stats['widths'])),
                        "std": float(np.std(train_stats['widths'])),
                        "min": float(np.min(train_stats['widths'])),
                        "max": float(np.max(train_stats['widths'])),
                        "percentiles": {str(p): float(np.percentile(train_stats['widths'], p)) 
                                      for p in [25, 50, 75, 90, 95, 99]}
                    },
                    "height": {
                        "mean": float(np.mean(train_stats['heights'])),
                        "median": float(np.median(train_stats['heights'])),
                        "std": float(np.std(train_stats['heights'])),
                        "min": float(np.min(train_stats['heights'])),
                        "max": float(np.max(train_stats['heights'])),
                        "percentiles": {str(p): float(np.percentile(train_stats['heights'], p)) 
                                      for p in [25, 50, 75, 90, 95, 99]}
                    },
                    "diagonal": {
                        "mean": float(np.mean(train_diag)),
                        "median": float(np.median(train_diag)),
                        "percentiles": {str(p): float(np.percentile(train_diag, p)) 
                                      for p in [10, 30, 50, 70, 85, 95]}
                    }
                },
                "labels_per_image": {
                    "mean": float(np.mean(train_stats['labels_per_image'])),
                    "median": float(np.median(train_stats['labels_per_image'])),
                    "max": int(np.max(train_stats['labels_per_image'])),
                    "percentiles": {str(p): float(np.percentile(train_stats['labels_per_image'], p)) 
                                  for p in [90, 95, 99]}
                },
                "class_counts": dict(train_stats['class_counts'])
            },
            "val": {
                "total_annotations": len(val_stats['widths']),
                "total_images": val_stats['total_images'],
                "class_counts": dict(val_stats['class_counts'])
            },
            "analysis_summary": {
                "classes": train_classes,
                "current_anchor_ratios": self.anchor_ratios,
                "current_anchor_sizes": self.anchor_sizes,
                "recommended_detections_per_image": int(np.percentile(
                    np.concatenate([train_stats['labels_per_image'], val_stats['labels_per_image']]), 95)),
                "recommended_anchor_sizes": [int(np.percentile(all_diag, p)) for p in [10, 30, 50, 70, 85, 95]]
            }
        }
        
        stats_file = self.output_dir / "kitti_detailed_statistics.json"
        with open(stats_file, 'w') as f:
            json.dump(detailed_stats, f, indent=2)
        
        print(f"[SAVE] Detailed statistics saved to: {stats_file}")
        return stats_file
    
    def analyze_datasets(self, train_file, val_file):
        """Main analysis workflow"""
        print("🚀 KITTI Dataset Distribution Analyzer (A100 Optimized)")
        print("="*60)
        
        # Setup GPU after multiprocessing is done
        self.workspace = self.setup_gpu()
        
        # Analyze both datasets
        print(f"\n[TRAIN] Analyzing training dataset...")
        train_stats, train_classes = self.analyze_coco_file(train_file)
        
        # Keep GPU active
        if self.workspace is not None:
            self.workspace = self.workspace * 1.0
        
        print(f"\n[VAL] Analyzing validation dataset...")
        val_stats, val_classes = self.analyze_coco_file(val_file)
        
        # Create all plots
        print(f"\n[PLOTS] Creating comprehensive visualization plots...")
        self.create_overall_distributions(train_stats, val_stats)
        self.create_class_wise_plots(train_stats, val_stats, train_classes)
        self.create_overlayed_plots(train_stats, val_stats, train_classes)
        self.create_stacked_absolute_density_plots(train_stats, val_stats, train_classes)
        self.create_image_size_analysis(train_file, val_file)
        self.create_labels_per_image_plots(train_stats, val_stats)
        
        # Print comprehensive statistics
        self.print_comprehensive_statistics(train_stats, val_stats, train_classes)
        
        # Save detailed statistics
        stats_file = self.save_detailed_statistics(train_stats, val_stats, train_classes)
        
        # Final summary
        print(f"\n💾 ANALYSIS COMPLETE:")
        print(f"  📊 Plots saved to: {self.output_dir}/")
        print(f"     • kitti_overall_distributions.png - Overall width/height/aspect/diagonal")
        print(f"     • kitti_width_by_class.png - Width distributions per class")
        print(f"     • kitti_height_by_class.png - Height distributions per class")
        print(f"     • kitti_aspect_ratio_by_class.png - Aspect ratios per class")
        print(f"     • kitti_diagonal_by_class.png - Diagonal sizes per class")
        print(f"     • kitti_distributions_overlayed.png - All classes overlayed")
        print(f"     • kitti_stacked_absolute_density.png - Stacked absolute density")
        print(f"     • kitti_image_size_analysis.png - Image size vs training config")
        print(f"     • kitti_labels_per_image_analysis.png - Labels per image analysis")
        print(f"  📄 Statistics: {stats_file}")
        
        if self.workspace is not None:
            final_gpu_mem = torch.cuda.memory_allocated() / 1024**3
            print(f"  🖥️  GPU memory used: {final_gpu_mem:.1f}GB")
        
        return train_stats, val_stats, train_classes


def main():
    """Main function with argument parsing"""
    parser = argparse.ArgumentParser(description='KITTI Dataset Distribution Analyzer')
    parser.add_argument('--dataset-root', type=str, 
                       default='/cephyr/users/kiross/Alvis/Desktop/mimer_naiss2024-5-153/Berhane/labelled_kitti/centralized/datasets/kitti_coco',
                       help='Root directory of KITTI COCO dataset')
    parser.add_argument('--output-dir', type=str, default='./kitti_analysis',
                       help='Output directory for analysis results')
    parser.add_argument('--max-workers', type=int, default=8,
                       help='Maximum number of parallel workers (reduced for stability)')
    parser.add_argument('--train-file', type=str, default=None,
                       help='Custom train annotation file (overrides dataset-root)')
    parser.add_argument('--val-file', type=str, default=None,
                       help='Custom val annotation file (overrides dataset-root)')
    
    args = parser.parse_args()
    
    # Determine file paths
    if args.train_file and args.val_file:
        train_file = Path(args.train_file)
        val_file = Path(args.val_file)
    else:
        dataset_root = Path(args.dataset_root)
        train_file = dataset_root / "annotations/train2017_no_misc_train_fixed_dimensions.json"
        val_file = dataset_root / "annotations/train2017_no_misc_val_fixed_dimensions.json"
    
    # Verify files exist
    if not train_file.exists():
        print(f"[ERROR] Train file not found: {train_file}")
        return 1
    if not val_file.exists():
        print(f"[ERROR] Val file not found: {val_file}")
        return 1
    
    print(f"[CONFIG] Dataset files:")
    print(f"  Train: {train_file}")
    print(f"  Val: {val_file}")
    print(f"  Output: {args.output_dir}")
    print(f"  Workers: {args.max_workers}")
    
    # Run analysis
    analyzer = KITTIAnalyzer(output_dir=args.output_dir, max_workers=args.max_workers)
    train_stats, val_stats, train_classes = analyzer.analyze_datasets(train_file, val_file)
    
    return 0


if __name__ == "__main__":
    exit(main())