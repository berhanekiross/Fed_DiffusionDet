#!/usr/bin/env python3
"""
Complete script to plot class distribution of KITTI dataset labels with dual axis
"""

import os
import glob
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns

def parse_kitti_labels(labels_dir):
    """Parse KITTI label files and count class occurrences AND image appearances."""
    
    class_counts = Counter()  # Count total objects per class
    class_image_counts = Counter()  # Count images containing each class
    total_objects = 0
    
    # Get all .txt files in the labels directory
    label_files = glob.glob(os.path.join(labels_dir, "*.txt"))
    
    print(f"Found {len(label_files)} label files")
    
    for label_file in label_files:
        classes_in_this_image = set()  # Track unique classes in this specific image
        
        with open(label_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line:  # Skip empty lines
                    parts = line.split()
                    if len(parts) >= 1:
                        class_id = parts[0]  # First element is class ID
                        class_counts[class_id] += 1  # Count this object
                        classes_in_this_image.add(class_id)  # Remember this class appears in this image
                        total_objects += 1
        
        # Now count this image for each class that appeared in it
        for class_id in classes_in_this_image:
            class_image_counts[class_id] += 1
    
    print(f"DEBUG - Object counts: {dict(class_counts)}")
    print(f"DEBUG - Image counts: {dict(class_image_counts)}")
    
    return class_counts, class_image_counts, total_objects

def plot_class_distribution(class_counts, class_image_counts, total_objects, save_path=None):
    """Plot the class distribution with dual axis."""
    
    # Define KITTI class mapping (class_id -> class_name)
    id_to_class_name = {
        '0': 'Car',
        '1': 'Van', 
        '2': 'Truck',
        '3': 'Pedestrian',
        '4': 'Person_sitting',
        '5': 'Cyclist',
        '6': 'Tram',
        # '7': 'Misc',
        # '-1': 'DontCare'
    }
    
    # Sort classes by count in descending order
    sorted_items = sorted(class_counts.items(), key=lambda x: x[1], reverse=True)
    class_ids = [item[0] for item in sorted_items]  # These are string IDs like '0', '3', etc.
    counts = [item[1] for item in sorted_items]
    
    # Get image counts in the same order
    image_counts = [class_image_counts[class_id] for class_id in class_ids]
    
    print(f"Actual class IDs found: {class_ids}")
    print(f"Object counts: {counts}")
    print(f"Image counts: {image_counts}")
    
    # Create x-axis labels with class name and ID
    x_labels = []
    for class_id in class_ids:
        class_name = id_to_class_name.get(class_id, f'Unknown_{class_id}')
        x_labels.append(f'{class_name}({class_id})')
        print(f"Class ID: '{class_id}' -> Name: {class_name}")  # Debug print
    
    # Calculate percentages
    percentages = [count/total_objects * 100 for count in counts]
    
    # Create figure with dual y-axis
    fig, ax1 = plt.subplots(1, 1, figsize=(10, 6))
    
    # Primary axis: Bar chart with object counts
    bars = ax1.bar(x_labels, counts, color='skyblue', alpha=0.7, label='Object Count', width=0.6)
    ax1.set_xlabel('Object Classes', fontsize=12)
    ax1.set_ylabel('Number of Objects', color='blue', fontsize=12)
    ax1.set_title('KITTI Dataset - Object Class Distribution', fontsize=14)
    ax1.tick_params(axis='x', rotation=45)
    ax1.tick_params(axis='y', labelcolor='blue')
    
    # Add count and percentage labels on bars
    for bar, count, pct in zip(bars, counts, percentages):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(counts)*0.01, 
                f'{count}({pct:.1f}%)', ha='center', va='bottom', fontsize=9)
    
    # Secondary axis: Line plot with image counts
    ax2 = ax1.twinx()
    x_positions = range(len(x_labels))
    line = ax2.plot(x_positions, image_counts, color='red', marker='o', 
                    linewidth=0.5, markersize=8, label='Images Containing Class', zorder=10)
    ax2.set_ylabel('Number of Images', color='red', fontsize=12)
    ax2.tick_params(axis='y', labelcolor='red')
    
    # Set the secondary axis limits to make the line more visible
    ax2.set_ylim(0, max(image_counts) * 1.1)
    
    # Add image count labels on line points
    for i, img_count in enumerate(image_counts):
        ax2.annotate(f'{img_count}', (i, img_count), textcoords="offset points", 
                    xytext=(0,15), ha='center', fontsize=9, color='red', weight='bold',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor='white', edgecolor='red', alpha=0.8))
    
    # Add legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
    
    # Add grid for better readability
    ax1.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    
    plt.show()
    
    # Print summary statistics
    print(f"\nSummary:")
    print(f"Total objects: {total_objects}")
    print(f"Number of classes: {len(class_ids)}")
    print(f"\nClass distribution:")
    for class_id, count in sorted_items:
        class_name = id_to_class_name.get(class_id, f'Unknown_{class_id}')
        img_count = class_image_counts[class_id]
        percentage = count/total_objects * 100
        print(f"  {class_name}({class_id}): {count} objects ({percentage:.1f}%) in {img_count} images")

def main():
    # Set the path to your KITTI labels directory
    labels_dir = "/mimer/NOBACKUP/groups/naiss2024-5-153/old_projects/Berhane/labelled_kitti/training/labels"
    
    # Check if directory exists
    if not os.path.exists(labels_dir):
        print(f"Error: Directory {labels_dir} does not exist!")
        return
    
    print(f"Analyzing KITTI labels in: {labels_dir}")
    
    # Parse labels and count classes
    class_counts, class_image_counts, total_objects = parse_kitti_labels(labels_dir)
    
    if total_objects == 0:
        print("No objects found in label files!")
        return
    
    # Plot the distribution
    save_path = "kitti_class_distribution.png"  # Optional: save the plot
    plot_class_distribution(class_counts, class_image_counts, total_objects, save_path)

if __name__ == "__main__":
    main()