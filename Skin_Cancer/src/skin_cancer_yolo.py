"""
YOLO-based Skin Cancer Detection
Note: This approach treats skin cancer detection as object detection,
which requires lesion segmentation/annotation. This is a demonstration
of how YOLO could be adapted for medical imaging.
"""

import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ultralytics import YOLO
import torch
from pathlib import Path
import json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import shutil

class SkinCancerYOLO:
    def __init__(self, data_path=".", model_size="yolov8n.pt"):
        self.data_path = Path(data_path)
        self.model = YOLO(model_size)
        self.label_encoder = LabelEncoder()
        self.classes = ['nv', 'mel', 'bkl', 'bcc', 'akiec', 'vasc', 'df']
        
    def prepare_yolo_dataset(self):
        """
        Prepare dataset in YOLO format
        Note: This creates dummy bounding boxes since HAM10000 doesn't have annotations
        In real medical applications, you'd need expert annotations
        """
        print("Preparing YOLO dataset...")
        
        # Create YOLO directory structure
        yolo_dir = self.data_path / "yolo_dataset"
        yolo_dir.mkdir(exist_ok=True)
        
        for split in ['train', 'val', 'test']:
            (yolo_dir / split / 'images').mkdir(parents=True, exist_ok=True)
            (yolo_dir / split / 'labels').mkdir(parents=True, exist_ok=True)
        
        # Load metadata
        df = pd.read_csv(self.data_path / "HAM10000_metadata.csv")
        
        # Encode labels
        df['label_encoded'] = self.label_encoder.fit_transform(df['dx'])
        
        # Split data
        train_df, temp_df = train_test_split(df, test_size=0.3, random_state=42, stratify=df['dx'])
        val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42, stratify=temp_df['dx'])
        
        # Process each split
        for split_name, split_df in [('train', train_df), ('val', val_df), ('test', test_df)]:
            self._process_split(split_df, yolo_dir / split_name)
        
        # Create dataset.yaml
        self._create_dataset_yaml(yolo_dir)
        
        return yolo_dir
    
    def _process_split(self, df, split_dir):
        """Process a data split for YOLO format"""
        for _, row in df.iterrows():
            image_id = row['image_id']
            label = row['label_encoded']
            
            # Find image file
            image_path = None
            for part in ['HAM10000_images_part_1', 'HAM10000_images_part_2']:
                potential_path = self.data_path / part / f"{image_id}.jpg"
                if potential_path.exists():
                    image_path = potential_path
                    break
            
            if image_path is None:
                continue
            
            # Copy image
            dest_image = split_dir / 'images' / f"{image_id}.jpg"
            shutil.copy2(image_path, dest_image)
            
            # Create dummy label file (full image as bounding box)
            # In real medical applications, you'd need expert annotations
            label_file = split_dir / 'labels' / f"{image_id}.txt"
            with open(label_file, 'w') as f:
                # Format: class_id center_x center_y width height (normalized)
                f.write(f"{label} 0.5 0.5 1.0 1.0\n")
    
    def _create_dataset_yaml(self, yolo_dir):
        """Create dataset configuration file"""
        yaml_content = f"""
path: {yolo_dir.absolute()}
train: train/images
val: val/images
test: test/images

nc: {len(self.classes)}
names: {self.classes}
"""
        with open(yolo_dir / 'dataset.yaml', 'w') as f:
            f.write(yaml_content)
    
    def train_model(self, yolo_dir, epochs=50):
        """Train YOLO model"""
        print("Training YOLO model...")
        results = self.model.train(
            data=str(yolo_dir / 'dataset.yaml'),
            epochs=epochs,
            imgsz=640,
            batch=16,
            device='cuda' if torch.cuda.is_available() else 'cpu'
        )
        return results
    
    def predict_and_visualize(self, image_path, save_path=None):
        """Make prediction and visualize results"""
        results = self.model(image_path)
        
        # Plot results
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Original image
        img = cv2.imread(str(image_path))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        axes[0].imshow(img_rgb)
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        # Prediction with bounding box
        annotated_img = results[0].plot()
        axes[1].imshow(annotated_img)
        axes[1].set_title('YOLO Prediction')
        axes[1].axis('off')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        return results

def main():
    """Main function to demonstrate YOLO approach"""
    print("=== YOLO-based Skin Cancer Detection ===")
    print("Note: This is a demonstration. Real medical applications require expert annotations.")
    
    # Initialize YOLO detector
    detector = SkinCancerYOLO()
    
    # Prepare dataset (this will take some time)
    print("Preparing dataset...")
    yolo_dir = detector.prepare_yolo_dataset()
    
    # Train model (uncomment to train)
    # print("Training model...")
    # detector.train_model(yolo_dir, epochs=10)
    
    # Example prediction (using a sample image)
    sample_image = "HAM10000_images_part_1/ISIC_0024306.jpg"
    if os.path.exists(sample_image):
        print("Making prediction on sample image...")
        detector.predict_and_visualize(sample_image, "yolo_prediction.png")
    
    print("YOLO setup complete. Check 'yolo_dataset' folder for prepared data.")

if __name__ == "__main__":
    main()
