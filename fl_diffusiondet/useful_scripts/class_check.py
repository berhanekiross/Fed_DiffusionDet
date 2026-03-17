#!/usr/bin/env python3
"""
COCO JSON Class Distribution Checker
Analyzes object class distribution across all COCO JSON files in fl_dataset/annotations/
"""

import os
import json
from pathlib import Path
from collections import defaultdict, Counter
import pandas as pd

# KITTI class mapping (COCO uses 1-based IDs)
KITTI_CLASSES = {
    1: "Car",
    2: "Van", 
    3: "Truck",
    4: "Pedestrian",
    5: "Person_sitting", 
    6: "Cyclist",
    7: "Tram"
}

def load_coco_json(json_path):
    """Load and return COCO JSON data."""
    try:
        with open(json_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading {json_path}: {e}")
        return None

def analyze_single_json(json_path):
    """Analyze class distribution in a single COCO JSON file."""
    data = load_coco_json(json_path)
    if not data:
        return None
    
    # Count annotations by category
    class_counts = Counter()
    
    for annotation in data.get('annotations', []):
        category_id = annotation.get('category_id')
        if category_id:
            class_counts[category_id] += 1
    
    # Get basic stats
    total_images = len(data.get('images', []))
    total_annotations = len(data.get('annotations', []))
    
    return {
        'file': os.path.basename(json_path),
        'total_images': total_images,
        'total_annotations': total_annotations,
        'class_counts': dict(class_counts),
        'unique_classes': len(class_counts)
    }

def create_distribution_table(results):
    """Create a comprehensive distribution table."""
    
    # Extract all unique class IDs
    all_class_ids = set()
    for result in results.values():
        if result:
            all_class_ids.update(result['class_counts'].keys())
    
    all_class_ids = sorted(all_class_ids)
    
    # Create table data
    table_data = []
    
    for json_file, result in results.items():
        if not result:
            continue
            
        row = {
            'JSON_File': result['file'],
            'Total_Images': result['total_images'],
            'Total_Annotations': result['total_annotations'],
            'Unique_Classes': result['unique_classes']
        }
        
        # Add class counts
        for class_id in all_class_ids:
            class_name = KITTI_CLASSES.get(class_id, f"Unknown_{class_id}")
            count = result['class_counts'].get(class_id, 0)
            row[f"{class_name}_{class_id}"] = count
        
        table_data.append(row)
    
    return pd.DataFrame(table_data)

def print_summary_statistics(results):
    """Print summary statistics."""
    
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)
    
    # Overall stats
    total_files = len([r for r in results.values() if r])
    total_images = sum(r['total_images'] for r in results.values() if r)
    total_annotations = sum(r['total_annotations'] for r in results.values() if r)
    
    print(f"Total JSON files analyzed: {total_files}")
    print(f"Total images across all files: {total_images}")
    print(f"Total annotations across all files: {total_annotations}")
    
    # Class distribution across all files
    overall_class_counts = Counter()
    for result in results.values():
        if result:
            for class_id, count in result['class_counts'].items():
                overall_class_counts[class_id] += count
    
    print(f"\nOverall Class Distribution:")
    print("-" * 40)
    for class_id in sorted(overall_class_counts.keys()):
        class_name = KITTI_CLASSES.get(class_id, f"Unknown_{class_id}")
        count = overall_class_counts[class_id]
        percentage = (count / total_annotations) * 100 if total_annotations > 0 else 0
        print(f"{class_name:15} (ID {class_id}): {count:6d} ({percentage:5.1f}%)")

def check_missing_classes(results):
    """Check for missing classes in each client."""
    
    print("\n" + "="*80)
    print("MISSING CLASS ANALYSIS")
    print("="*80)
    
    expected_classes = set(KITTI_CLASSES.keys())
    
    for json_file, result in results.items():
        if not result:
            continue
            
        present_classes = set(result['class_counts'].keys())
        missing_classes = expected_classes - present_classes
        
        if missing_classes:
            print(f"\n{result['file']}:")
            print(f"  Missing classes: {[KITTI_CLASSES[cid] for cid in sorted(missing_classes)]}")
            print(f"  Present classes: {[KITTI_CLASSES[cid] for cid in sorted(present_classes)]}")
        else:
            print(f"\n{result['file']}: All classes present ✓")

def analyze_client_patterns(results):
    """Analyze patterns by client type."""
    
    print("\n" + "="*80)
    print("CLIENT PATTERN ANALYSIS")
    print("="*80)
    
    # Group by client
    clients = defaultdict(list)
    
    for json_file, result in results.items():
        if not result:
            continue
            
        # Extract client name from filename
        filename = result['file']
        if '_' in filename:
            parts = filename.replace('.json', '').split('_')
            if len(parts) >= 2:
                client_name = '_'.join(parts[1:])  # Everything after first underscore
                clients[client_name].append(result)
    
    for client_name, client_results in clients.items():
        print(f"\nClient: {client_name}")
        print("-" * 30)
        
        total_images = sum(r['total_images'] for r in client_results)
        total_annotations = sum(r['total_annotations'] for r in client_results)
        
        print(f"  Total files: {len(client_results)}")
        print(f"  Total images: {total_images}")
        print(f"  Total annotations: {total_annotations}")
        
        # Aggregate class counts for this client
        client_class_counts = Counter()
        for result in client_results:
            for class_id, count in result['class_counts'].items():
                client_class_counts[class_id] += count
        
        print(f"  Classes present: {len(client_class_counts)}")
        for class_id in sorted(client_class_counts.keys()):
            class_name = KITTI_CLASSES.get(class_id, f"Unknown_{class_id}")
            count = client_class_counts[class_id]
            print(f"    {class_name}: {count}")

def main():
    # Configuration
    ANNOTATIONS_DIR = "./fl_dataset/annotations/"
    
    print("=== COCO JSON Class Distribution Checker ===")
    print(f"Analyzing JSON files in: {ANNOTATIONS_DIR}")
    
    # Check if directory exists
    annotations_path = Path(ANNOTATIONS_DIR)
    if not annotations_path.exists():
        print(f"Error: Directory {ANNOTATIONS_DIR} does not exist!")
        return
    
    # Find all JSON files
    json_files = list(annotations_path.glob("*.json"))
    
    if not json_files:
        print(f"No JSON files found in {ANNOTATIONS_DIR}")
        return
    
    print(f"Found {len(json_files)} JSON files")
    
    # Analyze each JSON file
    results = {}
    
    for json_path in sorted(json_files):
        print(f"Analyzing {json_path.name}...")
        result = analyze_single_json(json_path)
        results[str(json_path)] = result
    
    # Create and display distribution table
    print("\n" + "="*80)
    print("CLASS DISTRIBUTION TABLE")
    print("="*80)
    
    df = create_distribution_table(results)
    
    # Display table with proper formatting
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', 20)
    
    print(df.to_string(index=False))
    
    # Print detailed analysis
    print_summary_statistics(results)
    check_missing_classes(results)
    analyze_client_patterns(results)
    
    # Save detailed report to file
    output_file = "class_distribution_report.txt"
    with open(output_file, 'w') as f:
        f.write("COCO JSON Class Distribution Report\n")
        f.write("="*50 + "\n\n")
        f.write(df.to_string(index=False))
        f.write("\n\n")
    
    print(f"\n=== Analysis Complete ===")
    print(f"Detailed report saved to: {output_file}")
    
    # Quick summary
    print(f"\nQuick Summary:")
    valid_results = [r for r in results.values() if r]
    if valid_results:
        total_files = len(valid_results)
        files_with_missing = sum(1 for r in valid_results if r['unique_classes'] < 7)
        print(f"- {total_files} JSON files analyzed")
        print(f"- {files_with_missing} files have missing classes")
        print(f"- {total_files - files_with_missing} files have all 7 classes")

if __name__ == "__main__":
    main()