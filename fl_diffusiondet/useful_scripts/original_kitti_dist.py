#!/usr/bin/env python3
"""
Original KITTI Label_2 Directory Analyzer
Analyzes the original KITTI label_2 directory including DontCare, Misc, etc.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict, Counter
import seaborn as sns

# ALL KITTI classes including ignored ones
ALL_KITTI_CLASSES = {
    'Car': 0,
    'Van': 1,
    'Truck': 2,
    'Pedestrian': 3,
    'Person_sitting': 4,
    'Cyclist': 5,
    'Tram': 6,
    'Misc': 7,
    'DontCare': 8
}

def parse_original_kitti_label(label_path):
    """Parse original KITTI label file and return all objects including DontCare."""
    objects = []
    
    if not os.path.exists(label_path):
        return objects
    
    with open(label_path, 'r') as f:
        for line_num, line in enumerate(f.readlines(), 1):
            parts = line.strip().split()
            if len(parts) < 15:
                continue
                
            class_name = parts[0]
            
            # Parse all fields
            try:
                truncated = float(parts[1])
                occluded = int(parts[2])
                alpha = float(parts[3])
                
                # Bounding box
                left = float(parts[4])
                top = float(parts[5])
                right = float(parts[6])
                bottom = float(parts[7])
                
                # 3D dimensions
                height_3d = float(parts[8])
                width_3d = float(parts[9])
                length_3d = float(parts[10])
                
                # 3D location
                x_3d = float(parts[11])
                y_3d = float(parts[12])
                z_3d = float(parts[13])
                
                # Rotation
                rotation_y = float(parts[14])
                
                # Calculate bbox area and aspect ratio
                bbox_width = right - left
                bbox_height = bottom - top
                bbox_area = bbox_width * bbox_height
                aspect_ratio = bbox_width / bbox_height if bbox_height > 0 else 1.0
                
                objects.append({
                    'class_name': class_name,
                    'truncated': truncated,
                    'occluded': occluded,
                    'alpha': alpha,
                    'bbox': [left, top, right, bottom],
                    'bbox_area': bbox_area,
                    'bbox_width': bbox_width,
                    'bbox_height': bbox_height,
                    'aspect_ratio': aspect_ratio,
                    'dimensions_3d': [height_3d, width_3d, length_3d],
                    'location_3d': [x_3d, y_3d, z_3d],
                    'rotation_y': rotation_y,
                    'distance': np.sqrt(x_3d**2 + z_3d**2)  # Distance from camera
                })
                
            except (ValueError, IndexError) as e:
                print(f"Warning: Could not parse line {line_num} in {label_path}: {line.strip()}")
                continue
    
    return objects

def analyze_original_kitti_labels(labels_dir):
    """Analyze all original KITTI label files."""
    
    labels_path = Path(labels_dir)
    if not labels_path.exists():
        print(f"Error: Directory {labels_dir} does not exist!")
        return None
    
    # Find all label files
    label_files = list(labels_path.glob("*.txt"))
    
    if not label_files:
        print(f"No label files found in {labels_dir}")
        return None
    
    print(f"Found {len(label_files)} label files")
    
    # Parse all labels
    all_objects = []
    images_with_objects = []
    
    for label_file in label_files:
        objects = parse_original_kitti_label(label_file)
        if objects:
            all_objects.extend(objects)
            images_with_objects.append({
                'filename': label_file.name,
                'objects': objects,
                'num_objects': len(objects)
            })
    
    return {
        'all_objects': all_objects,
        'images_with_objects': images_with_objects,
        'total_images': len(label_files),
        'images_with_annotations': len(images_with_objects)
    }

def analyze_class_distribution(data):
    """Analyze class distribution including DontCare and Misc."""
    
    print("\n" + "="*60)
    print("ORIGINAL KITTI CLASS DISTRIBUTION")
    print("="*60)
    
    all_objects = data['all_objects']
    
    # Count by class
    class_counts = Counter()
    for obj in all_objects:
        class_counts[obj['class_name']] += 1
    
    # Print distribution
    total_objects = len(all_objects)
    print(f"Total objects: {total_objects}")
    print(f"Total images: {data['total_images']}")
    print(f"Images with annotations: {data['images_with_annotations']}")
    print(f"Images without annotations: {data['total_images'] - data['images_with_annotations']}")
    
    print(f"\nClass Distribution:")
    print("-" * 40)
    
    for class_name in sorted(class_counts.keys()):
        count = class_counts[class_name]
        percentage = (count / total_objects) * 100
        print(f"{class_name:15}: {count:6d} ({percentage:6.2f}%)")
    
    return class_counts

def analyze_difficulty_levels(data):
    """Analyze KITTI difficulty levels based on official criteria."""
    
    print("\n" + "="*60)
    print("KITTI DIFFICULTY ANALYSIS")
    print("="*60)
    
    all_objects = data['all_objects']
    
    # Filter out DontCare and Misc for difficulty analysis
    valid_objects = [obj for obj in all_objects 
                    if obj['class_name'] not in ['DontCare', 'Misc']]
    
    print(f"Analyzing {len(valid_objects)} valid objects (excluding DontCare/Misc)")
    
    # KITTI difficulty criteria:
    # Easy: height >= 40px, occlusion <= 0, truncation <= 0.15
    # Moderate: height >= 25px, occlusion <= 1, truncation <= 0.3
    # Hard: height >= 25px, occlusion <= 2, truncation <= 0.5
    
    easy_objects = []
    moderate_objects = []
    hard_objects = []
    
    for obj in valid_objects:
        height = obj['bbox_height']
        occlusion = obj['occluded']
        truncation = obj['truncated']
        
        if height >= 40 and occlusion <= 0 and truncation <= 0.15:
            easy_objects.append(obj)
        elif height >= 25 and occlusion <= 1 and truncation <= 0.3:
            moderate_objects.append(obj)
        elif height >= 25 and occlusion <= 2 and truncation <= 0.5:
            hard_objects.append(obj)
    
    total_valid = len(valid_objects)
    
    print(f"\nKITTI Official Difficulty Distribution:")
    print("-" * 40)
    print(f"Easy objects:     {len(easy_objects):5d} ({len(easy_objects)/total_valid*100:5.1f}%)")
    print(f"Moderate objects: {len(moderate_objects):5d} ({len(moderate_objects)/total_valid*100:5.1f}%)")
    print(f"Hard objects:     {len(hard_objects):5d} ({len(hard_objects)/total_valid*100:5.1f}%)")
    
    # Analyze what makes objects difficult
    print(f"\nDifficulty Factor Analysis:")
    print("-" * 40)
    
    # Height distribution
    heights = [obj['bbox_height'] for obj in valid_objects]
    print(f"Object heights - Min: {min(heights):.1f}, Max: {max(heights):.1f}, Mean: {np.mean(heights):.1f}")
    
    small_objects = sum(1 for h in heights if h < 25)
    medium_objects = sum(1 for h in heights if 25 <= h < 40)
    large_objects = sum(1 for h in heights if h >= 40)
    
    print(f"  Small (< 25px):   {small_objects:5d} ({small_objects/total_valid*100:5.1f}%)")
    print(f"  Medium (25-40px): {medium_objects:5d} ({medium_objects/total_valid*100:5.1f}%)")
    print(f"  Large (>= 40px):  {large_objects:5d} ({large_objects/total_valid*100:5.1f}%)")
    
    # Occlusion distribution
    occlusion_counts = Counter(obj['occluded'] for obj in valid_objects)
    print(f"\nOcclusion levels:")
    for level in sorted(occlusion_counts.keys()):
        count = occlusion_counts[level]
        print(f"  Level {level}: {count:5d} ({count/total_valid*100:5.1f}%)")
    
    # Truncation distribution
    truncations = [obj['truncated'] for obj in valid_objects]
    print(f"\nTruncation levels:")
    print(f"  Min: {min(truncations):.3f}, Max: {max(truncations):.3f}, Mean: {np.mean(truncations):.3f}")
    
    low_trunc = sum(1 for t in truncations if t <= 0.15)
    med_trunc = sum(1 for t in truncations if 0.15 < t <= 0.3)
    high_trunc = sum(1 for t in truncations if t > 0.3)
    
    print(f"  Low (<= 0.15):    {low_trunc:5d} ({low_trunc/total_valid*100:5.1f}%)")
    print(f"  Medium (0.15-0.3): {med_trunc:5d} ({med_trunc/total_valid*100:5.1f}%)")
    print(f"  High (> 0.3):     {high_trunc:5d} ({high_trunc/total_valid*100:5.1f}%)")
    
    return {
        'easy_objects': easy_objects,
        'moderate_objects': moderate_objects,
        'hard_objects': hard_objects,
        'heights': heights,
        'occlusion_counts': occlusion_counts,
        'truncations': truncations
    }

def analyze_objects_per_image(data):
    """Analyze objects per image distribution."""
    
    print("\n" + "="*60)
    print("OBJECTS PER IMAGE ANALYSIS")
    print("="*60)
    
    images_with_objects = data['images_with_objects']
    total_images = data['total_images']
    
    # Count objects per image
    objects_per_image = [img['num_objects'] for img in images_with_objects]
    
    # Add zeros for images without objects
    images_without_objects = total_images - len(images_with_objects)
    objects_per_image.extend([0] * images_without_objects)
    
    print(f"Total images: {total_images}")
    print(f"Images with objects: {len(images_with_objects)}")
    print(f"Images without objects: {images_without_objects}")
    
    if objects_per_image:
        print(f"\nObjects per image statistics:")
        print(f"  Min: {min(objects_per_image)}")
        print(f"  Max: {max(objects_per_image)}")
        print(f"  Mean: {np.mean(objects_per_image):.2f}")
        print(f"  Median: {np.median(objects_per_image):.2f}")
        
        # Distribution
        count_dist = Counter(objects_per_image)
        print(f"\nObject count distribution:")
        for count in sorted(count_dist.keys())[:15]:  # Show first 15
            images = count_dist[count]
            percentage = (images / total_images) * 100
            print(f"  {count:2d} objects: {images:5d} images ({percentage:5.1f}%)")
        
        if max(objects_per_image) > 15:
            many_objects = sum(count_dist[i] for i in range(16, max(objects_per_image) + 1))
            print(f"  >15 objects: {many_objects:5d} images ({many_objects/total_images*100:5.1f}%)")
    
    return objects_per_image

def analyze_distance_and_size(data):
    """Analyze object distance and size relationships."""
    
    print("\n" + "="*60)
    print("DISTANCE AND SIZE ANALYSIS")
    print("="*60)
    
    all_objects = data['all_objects']
    
    # Filter valid objects
    valid_objects = [obj for obj in all_objects 
                    if obj['class_name'] not in ['DontCare', 'Misc']]
    
    # Distance analysis
    distances = [obj['distance'] for obj in valid_objects]
    areas = [obj['bbox_area'] for obj in valid_objects]
    
    print(f"Distance from camera:")
    print(f"  Min: {min(distances):.1f}m, Max: {max(distances):.1f}m, Mean: {np.mean(distances):.1f}m")
    
    # Size analysis
    print(f"\nBounding box areas:")
    print(f"  Min: {min(areas):.1f}px², Max: {max(areas):.1f}px², Mean: {np.mean(areas):.1f}px²")
    
    # Size categories
    tiny_objects = sum(1 for area in areas if area < 400)  # < 20x20
    small_objects = sum(1 for area in areas if 400 <= area < 1024)  # 20x20 to 32x32
    medium_objects = sum(1 for area in areas if 1024 <= area < 9216)  # 32x32 to 96x96
    large_objects = sum(1 for area in areas if area >= 9216)  # > 96x96
    
    total_valid = len(valid_objects)
    
    print(f"\nSize distribution:")
    print(f"  Tiny (< 400px²):     {tiny_objects:5d} ({tiny_objects/total_valid*100:5.1f}%)")
    print(f"  Small (400-1024px²): {small_objects:5d} ({small_objects/total_valid*100:5.1f}%)")
    print(f"  Medium (1k-9k px²):  {medium_objects:5d} ({medium_objects/total_valid*100:5.1f}%)")
    print(f"  Large (> 9k px²):    {large_objects:5d} ({large_objects/total_valid*100:5.1f}%)")
    
    return {
        'distances': distances,
        'areas': areas,
        'size_distribution': {
            'tiny': tiny_objects,
            'small': small_objects,
            'medium': medium_objects,
            'large': large_objects
        }
    }

def create_visualizations(data, class_counts, difficulty_data, objects_per_image, distance_data, output_dir="./quality_analysis"):
    """Create comprehensive visualizations."""
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Set up plotting
    plt.style.use('seaborn-v0_8')
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # 1. Class distribution
    ax1 = axes[0, 0]
    classes = list(class_counts.keys())
    counts = list(class_counts.values())
    
    bars = ax1.bar(classes, counts, alpha=0.8, color='skyblue')
    ax1.set_xlabel('Object Classes')
    ax1.set_ylabel('Number of Annotations')
    ax1.set_title('Original KITTI Class Distribution')
    ax1.tick_params(axis='x', rotation=45)
    
    # Add value labels on bars
    for bar, count in zip(bars, counts):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{count}', ha='center', va='bottom', fontsize=9)
    
    ax1.grid(True, alpha=0.3)
    
    # 2. Difficulty distribution
    ax2 = axes[0, 1]
    difficulty_counts = [
        len(difficulty_data['easy_objects']),
        len(difficulty_data['moderate_objects']),
        len(difficulty_data['hard_objects'])
    ]
    difficulty_labels = ['Easy', 'Moderate', 'Hard']
    
    bars = ax2.bar(difficulty_labels, difficulty_counts, alpha=0.8, color=['green', 'orange', 'red'])
    ax2.set_xlabel('Difficulty Level')
    ax2.set_ylabel('Number of Objects')
    ax2.set_title('KITTI Official Difficulty Distribution')
    
    for bar, count in zip(bars, difficulty_counts):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{count}', ha='center', va='bottom', fontsize=10)
    
    ax2.grid(True, alpha=0.3)
    
    # 3. Objects per image
    ax3 = axes[0, 2]
    ax3.hist(objects_per_image, bins=range(0, min(max(objects_per_image)+2, 21)), 
             alpha=0.7, color='lightgreen', edgecolor='black')
    ax3.set_xlabel('Objects per Image')
    ax3.set_ylabel('Number of Images')
    ax3.set_title('Objects per Image Distribution')
    ax3.grid(True, alpha=0.3)
    
    # 4. Object height distribution
    ax4 = axes[1, 0]
    heights = difficulty_data['heights']
    ax4.hist(heights, bins=50, alpha=0.7, color='purple', edgecolor='black')
    ax4.axvline(x=25, color='red', linestyle='--', label='Min for Hard (25px)')
    ax4.axvline(x=40, color='green', linestyle='--', label='Min for Easy (40px)')
    ax4.set_xlabel('Object Height (pixels)')
    ax4.set_ylabel('Number of Objects')
    ax4.set_title('Object Height Distribution')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # 5. Distance vs Size scatter
    ax5 = axes[1, 1]
    distances = distance_data['distances']
    areas = distance_data['areas']
    
    # Sample for visibility if too many points
    if len(distances) > 5000:
        indices = np.random.choice(len(distances), 5000, replace=False)
        distances = [distances[i] for i in indices]
        areas = [areas[i] for i in indices]
    
    ax5.scatter(distances, areas, alpha=0.5, s=10)
    ax5.set_xlabel('Distance from Camera (m)')
    ax5.set_ylabel('Bounding Box Area (px²)')
    ax5.set_title('Distance vs Object Size')
    ax5.set_yscale('log')
    ax5.grid(True, alpha=0.3)
    
    # 6. Occlusion levels
    ax6 = axes[1, 2]
    occlusion_counts = difficulty_data['occlusion_counts']
    levels = list(occlusion_counts.keys())
    counts = list(occlusion_counts.values())
    
    bars = ax6.bar(levels, counts, alpha=0.8, color='coral')
    ax6.set_xlabel('Occlusion Level')
    ax6.set_ylabel('Number of Objects')
    ax6.set_title('Occlusion Level Distribution')
    
    for bar, count in zip(bars, counts):
        height = bar.get_height()
        ax6.text(bar.get_x() + bar.get_width()/2., height,
                f'{count}', ha='center', va='bottom', fontsize=10)
    
    ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/original_kitti_analysis.png", dpi=300, bbox_inches='tight')
    print(f"\nVisualizations saved to {output_dir}/original_kitti_analysis.png")

def main():
    """Main analysis function."""
    
    # Configuration
    LABELS_DIR = "/mimer/NOBACKUP/groups/naiss2024-5-153/old_projects/Berhane/labelled_kitti/training/label_2/"
    
    print("=" * 80)
    print("ORIGINAL KITTI LABEL_2 ANALYSIS")
    print("=" * 80)
    print(f"Analyzing labels in: {LABELS_DIR}")
    
    # Parse all labels
    data = analyze_original_kitti_labels(LABELS_DIR)
    
    if not data:
        print("Failed to analyze labels!")
        return
    
    # Perform analyses
    class_counts = analyze_class_distribution(data)
    difficulty_data = analyze_difficulty_levels(data)
    objects_per_image = analyze_objects_per_image(data)
    distance_data = analyze_distance_and_size(data)
    
    # Create visualizations
    create_visualizations(data, class_counts, difficulty_data, objects_per_image, distance_data)
    
    # Summary comparison with your processed data
    print("\n" + "="*60)
    print("COMPARISON WITH YOUR PROCESSED DATA")
    print("="*60)
    
    total_valid_objects = len([obj for obj in data['all_objects'] 
                              if obj['class_name'] not in ['DontCare', 'Misc']])
    
    print(f"\nOriginal KITTI characteristics:")
    print(f"  Total images: {data['total_images']}")
    print(f"  Total objects (all): {len(data['all_objects'])}")
    print(f"  Valid objects (excluding DontCare/Misc): {total_valid_objects}")
    print(f"  DontCare objects: {class_counts.get('DontCare', 0)}")
    print(f"  Misc objects: {class_counts.get('Misc', 0)}")
    
    # Check if your processed data is missing difficult cases
    easy_pct = len(difficulty_data['easy_objects']) / total_valid_objects * 100
    moderate_pct = len(difficulty_data['moderate_objects']) / total_valid_objects * 100
    hard_pct = len(difficulty_data['hard_objects']) / total_valid_objects * 100
    
    print(f"\nDifficulty distribution:")
    print(f"  Easy: {easy_pct:.1f}%")
    print(f"  Moderate: {moderate_pct:.1f}%") 
    print(f"  Hard: {hard_pct:.1f}%")
    
    if hard_pct < 20:
        print("  ⚠️  WARNING: Low percentage of hard objects in original data")
    
    print(f"\nIf your processed data shows < 20% small objects,")
    print(f"but original has {hard_pct:.1f}% hard objects,")
    print(f"then your partitioning is filtering out difficult cases!")

if __name__ == "__main__":
    main()