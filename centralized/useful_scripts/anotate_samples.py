import cv2
import os
from pathlib import Path

# Paths
LABELS_DIR = "/mimer/NOBACKUP/groups/naiss2024-5-153/Berhane/labelled_kitti/training/labels"
IMAGES_DIR = "samples_per_class"
OUTPUT_DIR = "labelled_images"

# YOLO format: class_id x_center y_center width height (all normalized 0-1)
# KITTI classes
CLASSES = ['Car', 'Van', 'Truck', 'Pedestrian', 'Person_sitting', 'Cyclist', 'Tram', 'Misc', 'DontCare']

# Colors for each class (BGR format for OpenCV)
COLORS = {
    'Car': (0, 255, 0),           # Green
    'Van': (255, 0, 0),            # Blue
    'Truck': (0, 0, 255),          # Red
    'Pedestrian': (255, 255, 0),   # Cyan
    'Person_sitting': (255, 0, 255), # Magenta
    'Cyclist': (0, 255, 255),      # Yellow
    'Tram': (128, 0, 128),         # Purple
    'Misc': (128, 128, 128),       # Gray
    'DontCare': (64, 64, 64)       # Dark Gray
}

def yolo_to_bbox(x_center, y_center, width, height, img_width, img_height):
    """Convert YOLO format to pixel coordinates"""
    x_center *= img_width
    y_center *= img_height
    width *= img_width
    height *= img_height
    
    x1 = int(x_center - width / 2)
    y1 = int(y_center - height / 2)
    x2 = int(x_center + width / 2)
    y2 = int(y_center + height / 2)
    
    return x1, y1, x2, y2

def annotate_image(image_path, label_path, output_path):
    """Draw bounding boxes and labels on image"""
    # Read image
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"Failed to read image: {image_path}")
        return
    
    img_height, img_width = img.shape[:2]
    
    # Read label file
    if not os.path.exists(label_path):
        print(f"Label file not found: {label_path}")
        # Save image without annotations
        cv2.imwrite(str(output_path), img)
        return
    
    with open(label_path, 'r') as f:
        lines = f.readlines()
    
    # Draw each bounding box
    for line in lines:
        parts = line.strip().split()
        if len(parts) < 5:
            continue
            
        class_id = int(parts[0])
        if class_id >= len(CLASSES):
            continue
            
        class_name = CLASSES[class_id]
        x_center, y_center, width, height = map(float, parts[1:5])
        
        # Convert to pixel coordinates
        x1, y1, x2, y2 = yolo_to_bbox(x_center, y_center, width, height, img_width, img_height)
        
        # Get color for this class
        color = COLORS.get(class_name, (255, 255, 255))
        
        # Draw bounding box
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        
        # Draw label background
        label = class_name
        (text_width, text_height), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
        )
        cv2.rectangle(img, (x1, y1 - text_height - 5), 
                     (x1 + text_width, y1), color, -1)
        
        # Draw label text
        cv2.putText(img, label, (x1, y1 - 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # Save annotated image
    cv2.imwrite(str(output_path), img)
    print(f"Saved: {output_path}")

def main():
    # Create output directory
    Path(OUTPUT_DIR).mkdir(exist_ok=True)
    
    # Process each class folder
    for class_name in os.listdir(IMAGES_DIR):
        class_dir = Path(IMAGES_DIR) / class_name
        if not class_dir.is_dir():
            continue
        
        # Create output subdirectory for this class
        output_class_dir = Path(OUTPUT_DIR) / class_name
        output_class_dir.mkdir(exist_ok=True)
        
        print(f"\nProcessing class: {class_name}")
        
        # Process each image
        for img_file in class_dir.glob("*.png"):
            # Get corresponding label file
            label_file = Path(LABELS_DIR) / f"{img_file.stem}.txt"
            
            # Output path
            output_path = output_class_dir / img_file.name
            
            # Annotate and save
            annotate_image(img_file, label_file, output_path)
    
    print(f"\n✓ All annotated images saved to: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()