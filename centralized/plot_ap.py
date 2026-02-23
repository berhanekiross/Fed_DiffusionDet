import json
import matplotlib.pyplot as plt

def plot_diffdet_metrics_json(json_file, output_file='diffdet_ap_ap5_95.png'):
    """Plot AP50 and mAP@0.5:0.95 from DiffusionDet metrics.json"""
    
    # Read JSON file line by line
    iterations = []
    ap50_values = []
    map50_95_values = []
    
    with open(json_file, 'r') as f:
        for line in f:
            try:
                data = json.loads(line.strip())
                # Only process lines with bbox metrics (evaluation results)
                if 'bbox/AP' in data and 'bbox/AP50' in data:
                    iterations.append(data['iteration'])
                    ap50_values.append(data['bbox/AP50'])
                    map50_95_values.append(data['bbox/AP'])
            except json.JSONDecodeError:
                continue

    print(f"Sample AP50 values: {ap50_values[:5]}")
    print(f"Sample mAP values: {map50_95_values[:5]}")

    if not iterations:
        print("No evaluation metrics found in the JSON file")
        return
    
    plt.plot(iterations, ap50_values, '-', color='#e74c3c', linewidth=4, 
            label='AP@50', alpha=0.8)
    plt.plot(iterations, map50_95_values, '-', color='#f39c12', linewidth=4, 
            label='mAP@50-95', alpha=0.8)
    
    # Formatting
    plt.xlabel('Iteration', fontsize=18)
    plt.ylabel('Average Precision', fontsize=18)
    # plt.title('DiffusionDet Performance Progress', fontsize=14, fontweight='bold')
    plt.legend(fontsize=18)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.grid(True, alpha=0.3)
    
    # Set y-axis limits
    plt.ylim(0, 100)
    
    # Improve layout
    plt.tight_layout()
    
    # Save plot
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print final metrics
    print(f"Final AP50: {ap50_values[-1]:.3f}")
    print(f"Final mAP@50-95: {map50_95_values[-1]:.3f}")
    print(f"Total evaluation points: {len(iterations)}")

def plot_diffdet_loss(json_file, output_file='diffdet_cl_kitti_loss.png'):
    """Plot training losses vs iterations from DiffusionDet metrics.json"""
    
    # Read JSON file line by line
    iterations = []
    total_loss = []
    # bbox_loss = []
    # ce_loss = []
    # giou_loss = []
    
    with open(json_file, 'r') as f:
        for line in f:
            try:
                data = json.loads(line.strip())
                # Only process lines with loss metrics (training results)
                if 'total_loss' in data and 'loss_bbox' in data:
                    iterations.append(data['iteration'])
                    total_loss.append(data['total_loss'])
                    # bbox_loss.append(data['loss_bbox'])
                    # ce_loss.append(data['loss_ce'])
                    # giou_loss.append(data['loss_giou'])
            except json.JSONDecodeError:
                continue
    
    if not iterations:
        print("No training loss metrics found in the JSON file")
        return
    
    # Create the plot
    plt.figure(figsize=(12, 8))
    
    plt.plot(iterations, total_loss, '-', color='#2c3e50', linewidth=3, 
            label='Total Loss', alpha=0.9)
    # plt.plot(iterations, bbox_loss, '-', color='#3498db', linewidth=2, 
    #         label='BBox Loss', alpha=0.8)
    # plt.plot(iterations, ce_loss, '-', color='#e74c3c', linewidth=2, 
    #         label='Classification Loss', alpha=0.8)
    # plt.plot(iterations, giou_loss, '-', color='#f39c12', linewidth=2, 
    #         label='GIoU Loss', alpha=0.8)
    
    # Formatting
    plt.xlabel('Iteration', fontsize=18)
    plt.ylabel('Loss', fontsize=18)
    plt.legend(fontsize=16)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.grid(True, alpha=0.3)
    
    # Improve layout
    plt.tight_layout()
    
    # Save plot
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print final metrics
    print(f"Final Total Loss: {total_loss[-1]:.4f}")
    # print(f"Final BBox Loss: {bbox_loss[-1]:.4f}")
    # print(f"Final Classification Loss: {ce_loss[-1]:.4f}")
    # print(f"Final GIoU Loss: {giou_loss[-1]:.4f}")
    # print(f"Total training points: {len(iterations)}")

def plot_all_ap_trends(json_file, output_file='diffdet_cl_kitti_perclassap.png'):
    """Plot AP50-95 trends for overall and all classes vs iterations"""
    
    # Read JSON file line by line
    iterations = []
    overall_ap = []
    class_aps = {
        'Car': [], 'Cyclist': [], 'Pedestrian': [], 'Person_sitting': [], 
        'Tram': [], 'Truck': [], 'Van': []
    }
    
    with open(json_file, 'r') as f:
        for line in f:
            try:
                data = json.loads(line.strip())
                if 'bbox/AP' in data and 'bbox/AP50' in data:
                    iterations.append(data['iteration'])
                    overall_ap.append(data['bbox/AP'])
                    
                    # Extract per-class AP values
                    for class_name in class_aps.keys():
                        key = f'bbox/AP-{class_name}'
                        if key in data:
                            class_aps[class_name].append(data[key])
                        else:
                            class_aps[class_name].append(0)
            except json.JSONDecodeError:
                continue
    
    if not iterations:
        print("No evaluation metrics found")
        return
    
    # Create the plot
    plt.figure(figsize=(14, 10))
    
    # Plot overall AP
    # plt.plot(iterations, overall_ap, '-', color='black', linewidth=4, 
    #          label='Overall mAP@50-95', alpha=0.9)
    
    # Define colors for each class
    colors = ['#3498db', '#e74c3c', '#f39c12', '#2ecc71', 
              '#9b59b6', '#1abc9c', '#e67e22']
    
    # Plot per-class AP
    for i, (class_name, ap_values) in enumerate(class_aps.items()):
        if ap_values:  # Only plot if we have data
            plt.plot(iterations, ap_values, '-', color=colors[i], 
                    linewidth=2, label=class_name, alpha=0.8)
    
    # Formatting
    plt.xlabel('Iteration', fontsize=16)
    plt.ylabel('Average Precision (AP@0.5-0.95)', fontsize=16)
    # plt.title('DiffusionDet Per-Class AP Trends (KITTI)', fontsize=14, fontweight='bold')
    plt.legend(fontsize=16, loc='center left', bbox_to_anchor=(1, 0.5))
    plt.grid(True, alpha=0.3)
    
    # Adjust y-axis based on data range
    if max(overall_ap) > 10:  # If values are percentages
        plt.ylim(0, 100)
    else:  # If values are decimals
        plt.ylim(0, 1.0)
    
    # Improve layout
    plt.tight_layout()
    
    # Save plot
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print final results
    print(f"\nFinal Results:")
    print(f"Overall mAP@50-95: {overall_ap[-1]:.1f}")
    for class_name, ap_values in class_aps.items():
        if ap_values:
            print(f"{class_name:15s}: {ap_values[-1]:.1f}")

def plot_zod_ap_trends(json_file, output_file='diffdet_zod_perclassap.png'):
    """Plot AP50-95 trends for overall and all ZOD classes vs iterations"""
    
    # Read JSON file line by line
    iterations = []
    overall_ap = []
    ap50 = []
    class_aps = {
        'Vehicle': [], 'TrafficSign': [], 'PoleObject': [], 'TrafficGuide': [], 
        'TrafficSignal': [], 'Pedestrian': [], 'VulnerableVehicle': [], 
        'TrafficBeacon': [], 'DynamicBarrier': [], 'Animal': []
    }
    
    with open(json_file, 'r') as f:
        for line in f:
            try:
                data = json.loads(line.strip())
                if 'bbox/AP' in data and 'bbox/AP50' in data:
                    iterations.append(data['iteration'])
                    overall_ap.append(data['bbox/AP'])
                    ap50.append(data['bbox/AP50'])
                    
                    # Extract per-class AP values
                    for class_name in class_aps.keys():
                        key = f'bbox/AP-{class_name}'
                        if key in data:
                            class_aps[class_name].append(data[key])
                        else:
                            class_aps[class_name].append(0)
            except json.JSONDecodeError:
                continue
    
    if not iterations:
        print("No evaluation metrics found")
        return
    
    # Make all lines start from zero
    if iterations:
        # Create a complete iteration range starting from 0
        start_iter = 0
        end_iter = max(iterations)
        step = iterations[1] - iterations[0] if len(iterations) > 1 else 5000  # Your eval frequency
        
        # Create full iteration range
        full_iterations = list(range(start_iter, iterations[0], step)) + iterations
        
        # Pad all data with zeros for missing early iterations
        padding_length = len(full_iterations) - len(iterations)
        
        overall_ap = [0] * padding_length + overall_ap
        ap50 = [0] * padding_length + ap50
        
        for class_name in class_aps:
            class_aps[class_name] = [0] * padding_length + class_aps[class_name]
        
        # Update iterations to use full range
        iterations = full_iterations
    
    # Create the plot
    plt.figure(figsize=(14, 10))
    
    # Plot overall AP
    plt.plot(iterations, overall_ap, '-', color='black', linewidth=4, 
             label='Overall mAP@50-95', alpha=0.9)
    
    # Plot AP50 with bold pink color
    plt.plot(iterations, ap50, '-', color='#FF1493', linewidth=4, 
             label='mAP@50', alpha=0.9)
    
    # Define colors for each class (10 colors for ZOD classes)
    colors = ['#3498db', '#e74c3c', '#f39c12', '#2ecc71', '#9b59b6', 
              '#1abc9c', '#e67e22', '#34495e', '#f1c40f', '#8e44ad']
    
    # Plot per-class AP
    for i, (class_name, ap_values) in enumerate(class_aps.items()):
        if ap_values and max(ap_values) > 0:  # Only plot if we have meaningful data
            plt.plot(iterations, ap_values, '-', color=colors[i], 
                    linewidth=2, label=class_name, alpha=0.8)
    
    # Formatting with larger fonts
    plt.xlabel('Iteration', fontsize=16)
    plt.ylabel('Average Precision (AP@0.5-0.95)', fontsize=16)
    plt.title('DiffusionDet Per-Class AP Trends (ZOD)', fontsize=18, fontweight='bold')
    plt.legend(fontsize=16, loc='center left', bbox_to_anchor=(1, 0.5))
    plt.grid(True, alpha=0.3)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    
    # Set y-axis limits
    plt.ylim(0, max(max(overall_ap + ap50), 65))
    
    # Improve layout
    plt.tight_layout()
    
    # Save plot
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print final results
    print(f"\nFinal ZOD Results:")
    print(f"Overall mAP@50-95: {overall_ap[-1]:.1f}")
    print(f"Overall mAP@50: {ap50[-1]:.1f}")
    for class_name, ap_values in class_aps.items():
        if ap_values:
            print(f"{class_name:18s}: {ap_values[-1]:.1f}")

# Usage
# plot_zod_ap_trends('metrics.json')
# # Usage
plot_all_ap_trends('metrics.json')
# Usage
plot_diffdet_metrics_json('metrics.json')

plot_diffdet_loss('metrics.json')