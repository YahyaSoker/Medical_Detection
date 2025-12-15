"""
Prediction Script for Bone Fracture Segmentation Model
Predicts fractures on images in target folder and saves annotated images to pred folder
"""

import os
from pathlib import Path
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
import yaml

# Paths
BASE_DIR = Path(__file__).parent
DATA_YAML = BASE_DIR / "BoneFractureYolo8" / "data.yaml"
BEST_MODEL_PATH = BASE_DIR / "output" / "best.pt"
TARGET_DIR = BASE_DIR / "target"
PRED_DIR = BASE_DIR / "pred"
PRED_DIR.mkdir(parents=True, exist_ok=True)

def load_data_config():
    """Load data.yaml configuration"""
    with open(DATA_YAML, 'r') as f:
        config = yaml.safe_load(f)
    return config

def draw_predictions(image, results, class_names):
    """Draw bounding boxes and segmentation masks on image"""
    # Convert PIL to numpy if needed
    if isinstance(image, Image.Image):
        img_array = np.array(image)
    else:
        img_array = image.copy()
    
    # YOLOv8 returns a list of Results objects (one per image)
    # Get the first result (since we process one image at a time)
    if len(results) > 0:
        result = results[0]
        boxes = result.boxes
        masks = result.masks
        
        # Draw masks first (so boxes appear on top)
        if masks is not None:
            for i, mask in enumerate(masks.data):
                if boxes is not None and i < len(boxes):
                    class_id = int(boxes.cls[i].cpu().numpy())
                    color = get_color(class_id)
                    
                    # Get mask in original image coordinates
                    mask_np = mask.cpu().numpy()
                    h, w = img_array.shape[:2]
                    
                    # Resize mask to image size
                    mask_resized = cv2.resize(mask_np, (w, h))
                    mask_binary = (mask_resized > 0.5).astype(np.uint8)
                    
                    # Create colored mask overlay
                    mask_colored = np.zeros_like(img_array)
                    mask_colored[mask_binary > 0] = color
                    
                    # Blend mask with image (semi-transparent)
                    img_array = cv2.addWeighted(img_array, 0.7, mask_colored, 0.3, 0)
        
        # Draw bounding boxes and labels
        if boxes is not None:
            for i in range(len(boxes)):
                # Get box coordinates
                x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy()
                confidence = float(boxes.conf[i].cpu().numpy())
                class_id = int(boxes.cls[i].cpu().numpy())
                
                # Draw bounding box
                color = get_color(class_id)
                cv2.rectangle(img_array, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                
                # Draw label
                label = f"{class_names[class_id]}: {confidence:.2f}"
                label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(img_array, (int(x1), int(y1) - label_size[1] - 10),
                             (int(x1) + label_size[0], int(y1)), color, -1)
                cv2.putText(img_array, label, (int(x1), int(y1) - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    return img_array

def get_color(class_id):
    """Get color for a class ID"""
    colors = [
        (255, 0, 0),      # Red
        (0, 255, 0),      # Green
        (0, 0, 255),      # Blue
        (255, 255, 0),    # Yellow
        (255, 0, 255),    # Magenta
        (0, 255, 255),    # Cyan
        (128, 0, 128),    # Purple
    ]
    return colors[class_id % len(colors)]

def predict_images():
    """Predict fractures on images in target folder"""
    print("=" * 60)
    print("Bone Fracture Segmentation Prediction")
    print("=" * 60)
    
    # Check if best model exists
    if not BEST_MODEL_PATH.exists():
        print(f"\nError: Best model not found at {BEST_MODEL_PATH}")
        print("Please train the model first using train.py")
        return
    
    # Check if target directory exists and has images
    if not TARGET_DIR.exists():
        print(f"\nError: Target directory not found at {TARGET_DIR}")
        print("Please create the target directory and add images to predict")
        return
    
    # Find all images in target directory
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']
    image_files = []
    for ext in image_extensions:
        image_files.extend(list(TARGET_DIR.glob(f"*{ext}")))
        image_files.extend(list(TARGET_DIR.glob(f"*{ext.upper()}")))
    
    if not image_files:
        print(f"\nNo images found in {TARGET_DIR}")
        print("Supported formats: jpg, jpeg, png, bmp, tiff, tif")
        return
    
    print(f"\nFound {len(image_files)} image(s) to process")
    
    # Load data config for class names
    config = load_data_config()
    class_names = config['names']
    
    # Load model
    print(f"\nLoading best model from: {BEST_MODEL_PATH}")
    model = YOLO(str(BEST_MODEL_PATH))
    
    # Process each image
    print("\nProcessing images...")
    print("=" * 60)
    
    successful = 0
    failed = 0
    
    for img_path in image_files:
        try:
            print(f"\nProcessing: {img_path.name}")
            
            # Run prediction
            results = model.predict(
                source=str(img_path),
                imgsz=640,
                conf=0.25,
                iou=0.45,
                save=False,  # We'll save manually with custom visualization
                show=False,
                verbose=False
            )
            
            # Load original image
            image = Image.open(img_path).convert('RGB')
            img_array = np.array(image)
            
            # Convert BGR to RGB if needed (for cv2 compatibility)
            if len(img_array.shape) == 3:
                img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
            
            # Draw predictions
            annotated_img = draw_predictions(img_array, results, class_names)
            
            # Convert back to RGB for saving
            annotated_img = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)
            
            # Save annotated image
            output_path = PRED_DIR / img_path.name
            Image.fromarray(annotated_img).save(output_path)
            
            # Count detections
            num_detections = 0
            if len(results) > 0 and results[0].boxes is not None:
                num_detections = len(results[0].boxes)
            
            print(f"  Detections: {num_detections}")
            print(f"  Saved to: {output_path}")
            
            successful += 1
            
        except Exception as e:
            print(f"  Error processing {img_path.name}: {e}")
            failed += 1
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 60)
    print("Prediction completed!")
    print(f"  Successful: {successful}")
    print(f"  Failed: {failed}")
    print(f"\nAll annotated images saved to: {PRED_DIR}")
    print("=" * 60)

if __name__ == "__main__":
    predict_images()

