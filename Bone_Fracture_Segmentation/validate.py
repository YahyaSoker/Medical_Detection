"""
Validation Script for Bone Fracture Segmentation Model
Validates the trained model and generates comprehensive metrics and visualizations
"""

import os
import shutil
from pathlib import Path
from ultralytics import YOLO
import yaml
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import json

# Paths
BASE_DIR = Path(__file__).parent
DATA_YAML = BASE_DIR / "BoneFractureYolo8" / "data.yaml"
BEST_MODEL_PATH = BASE_DIR / "output" / "best.pt"
OUTPUT_DIR = BASE_DIR / "output" / "validation"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

def load_data_config():
    """Load data.yaml configuration"""
    with open(DATA_YAML, 'r') as f:
        config = yaml.safe_load(f)
    return config

def validate_model():
    """Validate the trained model"""
    print("=" * 60)
    print("Bone Fracture Segmentation Model Validation")
    print("=" * 60)
    
    # Check if best model exists
    if not BEST_MODEL_PATH.exists():
        print(f"\nError: Best model not found at {BEST_MODEL_PATH}")
        print("Please train the model first using train.py")
        return
    
    # Load data config
    config = load_data_config()
    class_names = config['names']
    num_classes = config['nc']
    
    print(f"\nDataset Configuration:")
    print(f"  Classes: {num_classes}")
    print(f"  Class names: {class_names}")
    
    print(f"\nLoading best model from: {BEST_MODEL_PATH}")
    model = YOLO(str(BEST_MODEL_PATH))
    
    # Run validation
    print("\nRunning validation on validation set...")
    print("=" * 60)
    
    try:
        # Validate the model
        metrics = model.val(
            data=str(DATA_YAML),
            split='val',
            imgsz=640,
            conf=0.25,
            iou=0.45,
            plots=True,
            save_json=True,
            save_hybrid=False,
            verbose=True
        )
        
        print("\n" + "=" * 60)
        print("Validation completed!")
        print("=" * 60)
        
        # Extract metrics
        results_dict = {}
        if hasattr(metrics, 'results_dict'):
            results_dict = metrics.results_dict
        elif hasattr(metrics, 'box'):
            # Extract from metrics object
            results_dict = {
                'metrics/mAP50(B)': getattr(metrics.box, 'map50', None),
                'metrics/mAP50-95(B)': getattr(metrics.box, 'map', None),
                'metrics/precision(B)': getattr(metrics.box, 'mp', None),
                'metrics/recall(B)': getattr(metrics.box, 'mr', None),
            }
        
        # Print metrics
        print("\nValidation Metrics:")
        print("-" * 60)
        for key, value in results_dict.items():
            if isinstance(value, (int, float)):
                print(f"  {key}: {value:.4f}")
        
        # Get per-class metrics if available
        if hasattr(metrics, 'box') and hasattr(metrics.box, 'maps'):
            print("\nPer-Class mAP50-95:")
            print("-" * 60)
            for i, class_name in enumerate(class_names):
                if i < len(metrics.box.maps):
                    print(f"  {class_name}: {metrics.box.maps[i]:.4f}")
        
        # Save metrics to JSON
        metrics_file = OUTPUT_DIR / "validation_metrics.json"
        with open(metrics_file, 'w') as f:
            json.dump(results_dict, f, indent=2)
        print(f"\nMetrics saved to: {metrics_file}")
        
        # Create visualizations
        create_confusion_matrix(model, class_names, OUTPUT_DIR)
        create_per_class_metrics(metrics, class_names, OUTPUT_DIR)
        create_precision_recall_curves(model, class_names, OUTPUT_DIR)
        
        # Copy validation plots from YOLOv8 output
        copy_validation_plots(OUTPUT_DIR)
        
        print(f"\nAll validation outputs saved to: {OUTPUT_DIR}")
        print("=" * 60)
        
    except Exception as e:
        print(f"\nError during validation: {e}")
        import traceback
        traceback.print_exc()
        raise

def create_confusion_matrix(model, class_names, output_dir):
    """Create confusion matrix visualization"""
    try:
        # YOLOv8 saves confusion matrix, try to find it
        confusion_paths = [
            Path("runs") / "seg" / "val" / "confusion_matrix.png",
            BASE_DIR / "output" / "validation" / "confusion_matrix.png",
        ]
        
        # Also search recursively
        for path in BASE_DIR.rglob("confusion_matrix.png"):
            if path.exists():
                shutil.copy(path, output_dir / "confusion_matrix.png")
                print(f"Confusion matrix saved to: {output_dir / 'confusion_matrix.png'}")
                return
        
        print("Note: Confusion matrix not found in YOLOv8 output")
    except Exception as e:
        print(f"Warning: Could not create confusion matrix: {e}")

def create_per_class_metrics(metrics, class_names, output_dir):
    """Create per-class performance bar chart"""
    try:
        if hasattr(metrics, 'box') and hasattr(metrics.box, 'maps'):
            per_class_map = metrics.box.maps
            
            plt.figure(figsize=(14, 8))
            bars = plt.barh(class_names, per_class_map, color=sns.color_palette("husl", len(class_names)))
            plt.xlabel('mAP50-95', fontsize=12, fontweight='bold')
            plt.ylabel('Class', fontsize=12, fontweight='bold')
            plt.title('Per-Class mAP50-95 Performance', fontsize=16, fontweight='bold')
            plt.grid(axis='x', alpha=0.3)
            
            # Add value labels
            for i, bar in enumerate(bars):
                width = bar.get_width()
                plt.text(width, bar.get_y() + bar.get_height()/2.,
                        f'{width:.3f}',
                        ha='left', va='center', fontsize=10, fontweight='bold')
            
            plt.tight_layout()
            plt.savefig(output_dir / 'per_class_map.png', dpi=300, bbox_inches='tight')
            plt.close()
            print(f"Per-class metrics chart saved to: {output_dir / 'per_class_map.png'}")
    except Exception as e:
        print(f"Warning: Could not create per-class metrics: {e}")

def create_precision_recall_curves(model, class_names, output_dir):
    """Create precision-recall curves"""
    try:
        # YOLOv8 saves PR curves, try to find them
        pr_paths = [
            Path("runs") / "seg" / "val" / "PR_curve.png",
            BASE_DIR / "output" / "validation" / "PR_curve.png",
        ]
        
        # Search recursively
        for path in BASE_DIR.rglob("PR_curve.png"):
            if path.exists():
                shutil.copy(path, output_dir / "PR_curve.png")
                print(f"Precision-Recall curves saved to: {output_dir / 'PR_curve.png'}")
                return
        
        print("Note: PR curves not found in YOLOv8 output")
    except Exception as e:
        print(f"Warning: Could not create PR curves: {e}")

def copy_validation_plots(output_dir):
    """Copy validation plots from YOLOv8 output"""
    try:
        # Find validation results directory
        val_dirs = list(BASE_DIR.rglob("**/val*"))
        
        for val_dir in val_dirs:
            if val_dir.is_dir():
                # Copy relevant plots
                plots_to_copy = [
                    "confusion_matrix.png",
                    "PR_curve.png",
                    "F1_curve.png",
                    "results.png",
                ]
                
                for plot_name in plots_to_copy:
                    plot_path = val_dir / plot_name
                    if plot_path.exists():
                        shutil.copy(plot_path, output_dir / plot_name)
                        print(f"Copied {plot_name} to validation output")
    except Exception as e:
        print(f"Warning: Could not copy validation plots: {e}")

if __name__ == "__main__":
    validate_model()

