#!/usr/bin/env python3
"""
Analyze and visualize class distribution in IID partitions.
Creates heatmaps showing both raw counts and proportions.
"""

import json
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict, Counter
import pandas as pd

def load_json(file_path: str) -> dict:
    """Load JSON file."""
    with open(file_path, 'r') as f:
        return json.load(f)

def get_class_distribution(annotations_dir: str, num_clients: int = 6) -> tuple:
    """Extract class distribution from IID training partitions."""
    print("[ANALYZE] Extracting class distributions from IID partitions...")
    
    # Get category mapping from any file
    sample_file = os.path.join(annotations_dir, "train_client_0.json")
    sample_data = load_json(sample_file)
    
    # Create category mapping
    categories = {cat['id']: cat['name'] for cat in sample_data['categories']}
    class_names = [categories[i] for i in sorted(categories.keys())]
    
    print(f"[ANALYZE] Found classes: {class_names}")
    
    # Initialize distribution matrices
    counts_matrix = np.zeros((num_clients, len(class_names)), dtype=int)
    
    # Process each client's training data
    for client_id in range(num_clients):
        train_file = os.path.join(annotations_dir, f"train_client_{client_id}.json")
        
        if not os.path.exists(train_file):
            print(f"[WARNING] File not found: {train_file}")
            continue
            
        data = load_json(train_file)
        
        # Count annotations per category
        class_counts = Counter()
        for ann in data['annotations']:
            category_id = ann['category_id']
            if category_id in categories:
                class_name = categories[category_id]
                class_counts[class_name] += 1
        
        # Fill matrix row for this client
        for class_idx, class_name in enumerate(class_names):
            counts_matrix[client_id, class_idx] = class_counts[class_name]
        
        total_annotations = sum(class_counts.values())
        print(f"[ANALYZE] Client {client_id}: {total_annotations} total annotations")
        for class_name, count in class_counts.most_common():
            print(f"  {class_name}: {count}")
    
    # Calculate proportions matrix (per-client normalization)
    proportions_matrix = np.zeros_like(counts_matrix, dtype=float)
    for client_id in range(num_clients):
        client_total = counts_matrix[client_id].sum()
        if client_total > 0:
            proportions_matrix[client_id] = counts_matrix[client_id] / client_total
    
    return counts_matrix, proportions_matrix, class_names

def create_heatmaps(counts_matrix: np.ndarray, proportions_matrix: np.ndarray, 
                   class_names: list, output_dir: str = "analysis_output"):
    """Create heatmap visualizations."""
    print("[PLOT] Creating heatmap visualizations...")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Create client labels
    client_labels = [f"Client {i}" for i in range(counts_matrix.shape[0])]
    
    # Set up the plotting style
    plt.style.use('default')
    sns.set_palette("viridis")
    
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    # 1. Raw Counts Heatmap
    sns.heatmap(counts_matrix, 
                annot=True, 
                fmt='d',  # Integer format
                cmap='Reds',
                xticklabels=class_names,
                yticklabels=client_labels,
                cbar_kws={'label': 'Number of Annotations'},
                ax=ax1)
    
    ax1.set_title('IID FL Client Class Distribution - Raw Counts', 
                  fontsize=16, fontweight='bold', pad=20)
    ax1.set_xlabel('Object Classes', fontsize=14)
    ax1.set_ylabel('FL Clients', fontsize=14)
    ax1.tick_params(axis='x', rotation=45)
    
    # 2. Proportions Heatmap
    sns.heatmap(proportions_matrix, 
                annot=True, 
                fmt='.3f',  # 3 decimal places
                cmap='Blues',
                xticklabels=class_names,
                yticklabels=client_labels,
                cbar_kws={'label': 'Class Probability'},
                vmin=0, vmax=1,
                ax=ax2)
    
    ax2.set_title('IID FL Client Class Distribution - Proportions', 
                  fontsize=16, fontweight='bold', pad=20)
    ax2.set_xlabel('Object Classes', fontsize=14)
    ax2.set_ylabel('FL Clients', fontsize=14)
    ax2.tick_params(axis='x', rotation=45)
    
    # Adjust layout and save
    plt.tight_layout()
    
    # Save combined plot
    combined_path = os.path.join(output_dir, 'iid_class_distribution_heatmaps.png')
    plt.savefig(combined_path, dpi=300, bbox_inches='tight')
    print(f"[PLOT] Saved combined heatmaps: {combined_path}")
    
    # Save individual plots
    # Raw counts only
    plt.figure(figsize=(12, 8))
    sns.heatmap(counts_matrix, 
                annot=True, 
                fmt='d',
                cmap='Reds',
                xticklabels=class_names,
                yticklabels=client_labels,
                cbar_kws={'label': 'Number of Annotations'})
    
    plt.title('IID FL Client Class Distribution - Raw Counts', 
              fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Object Classes', fontsize=14)
    plt.ylabel('FL Clients', fontsize=14)
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    counts_path = os.path.join(output_dir, 'iid_raw_counts_heatmap.png')
    plt.savefig(counts_path, dpi=300, bbox_inches='tight')
    print(f"[PLOT] Saved raw counts heatmap: {counts_path}")
    plt.close()
    
    # Proportions only
    plt.figure(figsize=(12, 8))
    sns.heatmap(proportions_matrix, 
                annot=True, 
                fmt='.3f',
                cmap='Blues',
                xticklabels=class_names,
                yticklabels=client_labels,
                cbar_kws={'label': 'Class Probability'},
                vmin=0, vmax=1)
    
    plt.title('IID FL Client Class Distribution - Proportions', 
              fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Object Classes', fontsize=14)
    plt.ylabel('FL Clients', fontsize=14)
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    proportions_path = os.path.join(output_dir, 'iid_proportions_heatmap.png')
    plt.savefig(proportions_path, dpi=300, bbox_inches='tight')
    print(f"[PLOT] Saved proportions heatmap: {proportions_path}")
    plt.close()

def print_distribution_stats(counts_matrix: np.ndarray, proportions_matrix: np.ndarray, 
                           class_names: list):
    """Print detailed distribution statistics."""
    print("\n" + "="*80)
    print("IID DISTRIBUTION ANALYSIS")
    print("="*80)
    
    num_clients, num_classes = counts_matrix.shape
    
    # Overall statistics
    total_annotations = counts_matrix.sum()
    print(f"Total annotations across all clients: {total_annotations:,}")
    print(f"Average annotations per client: {total_annotations/num_clients:.1f}")
    
    # Per-class statistics
    print(f"\nPER-CLASS DISTRIBUTION:")
    print("-" * 50)
    for class_idx, class_name in enumerate(class_names):
        class_total = counts_matrix[:, class_idx].sum()
        class_mean = counts_matrix[:, class_idx].mean()
        class_std = counts_matrix[:, class_idx].std()
        class_min = counts_matrix[:, class_idx].min()
        class_max = counts_matrix[:, class_idx].max()
        
        print(f"{class_name:15} | Total: {class_total:5d} | "
              f"Mean: {class_mean:6.1f} | Std: {class_std:6.1f} | "
              f"Range: [{class_min:4d}, {class_max:4d}]")
    
    # Per-client statistics
    print(f"\nPER-CLIENT DISTRIBUTION:")
    print("-" * 50)
    for client_id in range(num_clients):
        client_total = counts_matrix[client_id].sum()
        dominant_class = class_names[np.argmax(counts_matrix[client_id])]
        dominant_count = counts_matrix[client_id].max()
        dominant_prop = proportions_matrix[client_id].max()
        
        print(f"Client {client_id} | Total: {client_total:5d} | "
              f"Dominant: {dominant_class} ({dominant_count}, {dominant_prop:.3f})")
    
    # Balance metrics
    print(f"\nBALANCE METRICS:")
    print("-" * 30)
    
    # Coefficient of variation for each class across clients
    print("Class balance across clients (lower CV = more balanced):")
    for class_idx, class_name in enumerate(class_names):
        cv = counts_matrix[:, class_idx].std() / counts_matrix[:, class_idx].mean()
        print(f"  {class_name:15}: CV = {cv:.3f}")
    
    # Client balance (how similar are client distributions)
    client_similarities = []
    for i in range(num_clients):
        for j in range(i+1, num_clients):
            # Cosine similarity between client distributions
            dot_product = np.dot(proportions_matrix[i], proportions_matrix[j])
            norm_i = np.linalg.norm(proportions_matrix[i])
            norm_j = np.linalg.norm(proportions_matrix[j])
            similarity = dot_product / (norm_i * norm_j)
            client_similarities.append(similarity)
    
    avg_similarity = np.mean(client_similarities)
    print(f"Average client similarity: {avg_similarity:.3f} (1.0 = identical distributions)")
    
    print("="*80)

def save_distribution_data(counts_matrix: np.ndarray, proportions_matrix: np.ndarray,
                          class_names: list, output_dir: str):
    """Save distribution data as CSV files."""
    print("[SAVE] Saving distribution data as CSV...")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Create client labels
    client_labels = [f"Client_{i}" for i in range(counts_matrix.shape[0])]
    
    # Save raw counts
    counts_df = pd.DataFrame(counts_matrix, 
                           index=client_labels, 
                           columns=class_names)
    counts_path = os.path.join(output_dir, 'iid_class_counts.csv')
    counts_df.to_csv(counts_path)
    print(f"[SAVE] Raw counts saved: {counts_path}")
    
    # Save proportions
    proportions_df = pd.DataFrame(proportions_matrix, 
                                index=client_labels, 
                                columns=class_names)
    proportions_path = os.path.join(output_dir, 'iid_class_proportions.csv')
    proportions_df.to_csv(proportions_path)
    print(f"[SAVE] Proportions saved: {proportions_path}")

def main():
    # Configuration
    FL_DATASET_DIR = "fl_dataset"
    ANNOTATIONS_IID_DIR = os.path.join(FL_DATASET_DIR, "annotations_iid")
    OUTPUT_DIR = "iid_analysis"
    NUM_CLIENTS = 6
    
    print(f"[START] Analyzing IID class distribution")
    print(f"[CONFIG] Source: {ANNOTATIONS_IID_DIR}")
    print(f"[CONFIG] Output: {OUTPUT_DIR}")
    print(f"[CONFIG] Number of clients: {NUM_CLIENTS}")
    
    # Check if IID annotations exist
    if not os.path.exists(ANNOTATIONS_IID_DIR):
        print(f"[ERROR] IID annotations directory not found: {ANNOTATIONS_IID_DIR}")
        print("[ERROR] Please run create_iid_partitions.py first")
        return
    
    # Extract class distributions
    counts_matrix, proportions_matrix, class_names = get_class_distribution(
        ANNOTATIONS_IID_DIR, NUM_CLIENTS
    )
    
    # Create visualizations
    create_heatmaps(counts_matrix, proportions_matrix, class_names, OUTPUT_DIR)
    
    # Print statistics
    print_distribution_stats(counts_matrix, proportions_matrix, class_names)
    
    # Save data
    save_distribution_data(counts_matrix, proportions_matrix, class_names, OUTPUT_DIR)
    
    print(f"\n[COMPLETE] Analysis complete. Check {OUTPUT_DIR}/ for outputs.")

if __name__ == "__main__":
    main()