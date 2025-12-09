"""
Main Script for Skin Cancer Detection
This script provides a comprehensive solution for skin cancer detection using multiple approaches.
"""

import os
import sys
import argparse
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description='Skin Cancer Detection System')
    parser.add_argument('--mode', choices=['yolo', 'cnn', 'visualize', 'all'], 
                       default='cnn', help='Choose the detection mode')
    parser.add_argument('--train', action='store_true', help='Train the model')
    parser.add_argument('--predict', type=str, help='Path to image for prediction')
    parser.add_argument('--epochs', type=int, default=20, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    
    args = parser.parse_args()
    
    print("=== Skin Cancer Detection System ===")
    print(f"Mode: {args.mode}")
    print(f"Training: {args.train}")
    print(f"Prediction image: {args.predict}")
    
    if args.mode == 'yolo' or args.mode == 'all':
        print("\n--- YOLO Approach ---")
        print("Note: YOLO is not ideal for skin cancer detection as it requires bounding box annotations.")
        print("This approach creates dummy bounding boxes for demonstration purposes.")
        
        try:
            from skin_cancer_yolo import SkinCancerYOLO
            
            detector = SkinCancerYOLO()
            
            if args.train:
                print("Preparing YOLO dataset...")
                yolo_dir = detector.prepare_yolo_dataset()
                print("Training YOLO model...")
                detector.train_model(yolo_dir, epochs=args.epochs)
            
            if args.predict and os.path.exists(args.predict):
                print(f"Making YOLO prediction on {args.predict}")
                detector.predict_and_visualize(args.predict, "yolo_prediction.png")
            
        except ImportError as e:
            print(f"Error importing YOLO module: {e}")
            print("Make sure you have installed ultralytics: pip install ultralytics")
    
    if args.mode == 'cnn' or args.mode == 'all':
        print("\n--- CNN with GradCAM Approach (Recommended) ---")
        print("This approach provides interpretable heatmaps showing which parts of the image")
        print("the model focuses on for making predictions.")
        
        try:
            from skin_cancer_cnn import SkinCancerDetector
            
            detector = SkinCancerDetector()
            
            if args.train:
                print("Preparing CNN dataset...")
                train_df, val_df, test_df = detector.prepare_data()
                train_loader, val_loader, test_loader = detector.create_dataloaders(
                    train_df, val_df, test_df, batch_size=args.batch_size
                )
                print("Training CNN model...")
                detector.train_model(train_loader, val_loader, epochs=args.epochs)
            
            if args.predict and os.path.exists(args.predict):
                print(f"Making CNN prediction with heatmap on {args.predict}")
                result = detector.predict_with_heatmap(args.predict, "cnn_prediction_heatmap.png")
                print(f"Prediction: {result['class_description']}")
                print(f"Confidence: {result['confidence']:.2%}")
            
        except ImportError as e:
            print(f"Error importing CNN module: {e}")
            print("Make sure you have installed all required packages: pip install -r requirements.txt")
    
    if args.mode == 'visualize' or args.mode == 'all':
        print("\n--- Visualization and Analysis ---")
        
        try:
            from skin_cancer_visualizer import SkinCancerVisualizer
            
            visualizer = SkinCancerVisualizer()
            
            print("Creating dataset analysis...")
            visualizer.plot_dataset_distribution("dataset_analysis.png")
            visualizer.plot_sample_images(samples_per_class=2, save_path="sample_images.png")
            
            # Example model results for comparison
            example_results = {
                'CNN + GradCAM': {'accuracy': 0.85, 'precision': 0.83, 'recall': 0.82, 'f1_score': 0.82},
                'YOLO': {'accuracy': 0.72, 'precision': 0.70, 'recall': 0.68, 'f1_score': 0.69},
                'Vision Transformer': {'accuracy': 0.88, 'precision': 0.86, 'recall': 0.85, 'f1_score': 0.85}
            }
            
            visualizer.plot_model_comparison(example_results, "model_comparison.png")
            visualizer.generate_report(example_results)
            
        except ImportError as e:
            print(f"Error importing visualization module: {e}")
            print("Make sure you have installed plotly: pip install plotly")
    
    print("\n=== Summary ===")
    print("For skin cancer detection with heatmap visualization, the CNN + GradCAM approach is recommended.")
    print("YOLO is not ideal for this task as it requires bounding box annotations.")
    print("Alternative models to consider:")
    print("1. Vision Transformers (ViT) - Excellent for medical imaging")
    print("2. EfficientNet - State-of-the-art for medical classification")
    print("3. ResNet with attention mechanisms - Good balance of performance and interpretability")
    print("4. Ensemble methods - Combine multiple models for improved accuracy")

if __name__ == "__main__":
    main()

