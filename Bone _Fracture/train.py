"""
YOLO Training Script for Bone Fracture Detection
This script trains a YOLO model and saves training/validation metrics, graphs, and results
All outputs are saved to results/model_history/
"""

from ultralytics import YOLO
import os
import yaml
import json
import pandas as pd
from pathlib import Path
# Note: matplotlib and seaborn removed - ultralytics handles plotting internally

def load_config(config_path='data.yaml'):
    """Load and validate the data configuration file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def save_metrics_to_csv(results, save_path):
    """Save training metrics to CSV file"""
    try:
        # Extract metrics from results
        metrics_dict = {}
        
        # Get results directory
        if hasattr(results, 'save_dir'):
            results_dir = Path(results.save_dir)
        else:
            results_dir = Path(save_path)
        
        # Try to read results.csv if it exists
        results_csv = results_dir / 'results.csv'
        if results_csv.exists():
            df = pd.read_csv(results_csv)
            df.to_csv(os.path.join(save_path, 'training_metrics.csv'), index=False)
            print(f"Training metrics saved to: {os.path.join(save_path, 'training_metrics.csv')}")
        
        return True
    except Exception as e:
        print(f"Warning: Could not save metrics to CSV: {e}")
        return False

def save_config_summary(config, save_path):
    """Save configuration summary to JSON"""
    config_summary = {
        'number_of_classes': config.get('nc', 0),
        'class_names': config.get('names', []),
        'train_path': config.get('train', ''),
        'val_path': config.get('val', ''),
        'test_path': config.get('test', ''),
        'roboflow': config.get('roboflow', {})
    }
    
    config_file = os.path.join(save_path, 'config_summary.json')
    with open(config_file, 'w') as f:
        json.dump(config_summary, f, indent=4)
    print(f"Configuration summary saved to: {config_file}")

def print_training_summary(results, save_path):
    """Print and save training summary"""
    summary = {
        'best_epoch': None,
        'best_map50': None,
        'best_map50_95': None,
        'best_precision': None,
        'best_recall': None,
        'final_epoch': None,
        'total_epochs': None
    }
    
    try:
        # Try to read results.csv
        results_csv = Path(save_path) / 'results.csv'
        if results_csv.exists():
            df = pd.read_csv(results_csv)
            if not df.empty:
                # Get best metrics
                best_idx = df['metrics/mAP50(B)'].idxmax() if 'metrics/mAP50(B)' in df.columns else 0
                summary['best_epoch'] = int(df.iloc[best_idx]['epoch']) if 'epoch' in df.columns else None
                summary['best_map50'] = float(df.iloc[best_idx]['metrics/mAP50(B)']) if 'metrics/mAP50(B)' in df.columns else None
                summary['best_map50_95'] = float(df.iloc[best_idx]['metrics/mAP50-95(B)']) if 'metrics/mAP50-95(B)' in df.columns else None
                summary['best_precision'] = float(df.iloc[best_idx]['metrics/precision(B)']) if 'metrics/precision(B)' in df.columns else None
                summary['best_recall'] = float(df.iloc[best_idx]['metrics/recall(B)']) if 'metrics/recall(B)' in df.columns else None
                summary['final_epoch'] = int(df.iloc[-1]['epoch']) if 'epoch' in df.columns else None
                summary['total_epochs'] = len(df)
        
        # Print summary
        print("\n" + "="*60)
        print("TRAINING SUMMARY")
        print("="*60)
        if summary['best_epoch'] is not None:
            print(f"Best Epoch: {summary['best_epoch']}")
        if summary['best_map50'] is not None:
            print(f"Best mAP50: {summary['best_map50']:.4f}")
        if summary['best_map50_95'] is not None:
            print(f"Best mAP50-95: {summary['best_map50_95']:.4f}")
        if summary['best_precision'] is not None:
            print(f"Best Precision: {summary['best_precision']:.4f}")
        if summary['best_recall'] is not None:
            print(f"Best Recall: {summary['best_recall']:.4f}")
        if summary['total_epochs'] is not None:
            print(f"Total Epochs Completed: {summary['total_epochs']}")
        print("="*60)
        
        # Save summary to JSON
        summary_file = os.path.join(save_path, 'training_summary.json')
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=4)
        print(f"\nTraining summary saved to: {summary_file}")
        
    except Exception as e:
        print(f"Warning: Could not generate training summary: {e}")

def train_model(
    model_size='yolo12s',  # Options: yolo12n, yolo12s, yolo12m, yolo12l, yolo12x (or yolov8n, yolov8s, etc.)
    data_yaml='data.yaml',
    epochs=100,
    imgsz=416,  # Reduced from 640 for faster training (416 is still good for detection)
    batch=32,  # Increased batch size for faster training (adjust based on GPU memory)
    device='cpu',  # 'cpu' or 'cuda' or '0' for GPU
    project='results',
    name='model_history',
    patience=50,
    save=True,
    plots=True,
    val=True,  # Run validation during training
    amp=True,  # Mixed precision training for faster training
    workers=8  # Optimize data loading workers
):
    """
    Train a YOLO model for bone fracture detection with comprehensive metrics tracking
    Optimized for faster training
    
    Args:
        model_size: YOLO model size (yolo12n, yolo12s, yolo12m, yolo12l, yolo12x for YOLOv12,
                   or yolov8n, yolov8s, yolov8m, yolov8l, yolov8x for YOLOv8)
        data_yaml: Path to data configuration file
        epochs: Number of training epochs
        imgsz: Image size for training (smaller = faster, default 416 instead of 640)
        batch: Batch size (larger = faster, adjust based on GPU memory)
        device: Device to use for training ('cpu', 'cuda', or GPU number)
        project: Project directory name
        name: Experiment name (will be saved to results/model_history)
        patience: Early stopping patience
        save: Whether to save checkpoints
        plots: Whether to generate training plots
        val: Whether to run validation during training
        amp: Mixed precision training (faster, uses less memory)
        workers: Number of data loading workers
    """
    
    # Load configuration
    config = load_config(data_yaml)
    print("="*60)
    print("BONE FRACTURE DETECTION - YOLO TRAINING")
    print("="*60)
    print(f"Loaded configuration from {data_yaml}")
    print(f"Number of classes: {config['nc']}")
    print(f"Classes: {config['names']}")
    
    # Set output directory
    output_dir = os.path.join(project, name)
    os.makedirs(output_dir, exist_ok=True)
    
    # Save configuration summary
    save_config_summary(config, output_dir)
    
    # Initialize YOLO model
    print(f"\nInitializing {model_size} model...")
    model = YOLO(f'{model_size}.pt')  # Load pretrained weights
    
    # Train the model
    print(f"\nStarting training...")
    print(f"Training parameters (optimized for speed):")
    print(f"  - Model: {model_size}")
    print(f"  - Epochs: {epochs}")
    print(f"  - Image size: {imgsz} (reduced from 640 for faster training)")
    print(f"  - Batch size: {batch} (increased for faster training)")
    print(f"  - Device: {device}")
    print(f"  - Mixed precision (AMP): {amp} (faster training)")
    print(f"  - Workers: {workers} (data loading)")
    print(f"  - Output directory: {output_dir}")
    print(f"  - Early stopping patience: {patience}")
    print("\nSpeed optimizations applied:")
    print("  ✓ Reduced image size (416 vs 640)")
    print("  ✓ Increased batch size")
    print("  ✓ Mixed precision training enabled")
    print("  ✓ Reduced augmentation intensity")
    print("  ✓ Reduced mosaic augmentation")
    print("="*60)
    
    # Train with optimized settings for faster training
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        project=project,
        name=name,
        patience=patience,
        save=save,
        plots=plots,
        val=val,
        verbose=True,
        # Speed optimizations
        amp=amp,  # Mixed precision training (faster, uses less memory)
        workers=workers,  # Optimize data loading
        # Additional metrics and visualization options
        save_json=True,  # Save results as JSON
        save_hybrid=False,  # Save hybrid labels
        show=False,  # Don't show plots during training
        # Validation settings - optimized for speed
        conf=0.25,  # Higher confidence threshold for faster validation (was 0.001)
        iou=0.6,  # IoU threshold for NMS
        # Reduced augmentation for faster training (can be increased if needed)
        hsv_h=0.01,  # Reduced augmentation
        hsv_s=0.5,  # Reduced augmentation
        hsv_v=0.3,  # Reduced augmentation
        degrees=0.0,  # No rotation (faster)
        translate=0.05,  # Reduced translation
        scale=0.3,  # Reduced scale
        shear=0.0,  # No shear
        perspective=0.0,  # No perspective
        flipud=0.0,  # No vertical flip
        fliplr=0.5,  # Keep horizontal flip (important for medical images)
        mosaic=0.5,  # Reduced mosaic (was 1.0) - mosaic is slow
        mixup=0.0,  # No mixup (faster)
        # Additional speed optimizations
        close_mosaic=10,  # Disable mosaic in last 10 epochs for faster training
        cache=False,  # Don't cache images (saves memory, but can be slower - set True if you have RAM)
    )
    
    print("\n" + "="*60)
    print("Training completed!")
    print("="*60)
    
    # Save metrics to CSV
    save_metrics_to_csv(results, output_dir)
    
    # Print and save training summary
    print_training_summary(results, output_dir)
    
    # Print information about saved files
    print(f"\nAll results saved to: {output_dir}")
    print("\nGenerated files:")
    print(f"  - Model weights: {output_dir}/weights/")
    print(f"  - Training plots: {output_dir}/")
    print(f"  - Metrics CSV: {output_dir}/training_metrics.csv")
    print(f"  - Configuration summary: {output_dir}/config_summary.json")
    print(f"  - Training summary: {output_dir}/training_summary.json")
    
    # Print best model path
    best_model_path = os.path.join(output_dir, 'weights', 'best.pt')
    last_model_path = os.path.join(output_dir, 'weights', 'last.pt')
    
    if os.path.exists(best_model_path):
        print(f"\n✓ Best model saved at: {best_model_path}")
    if os.path.exists(last_model_path):
        print(f"✓ Last model saved at: {last_model_path}")
    
    return model, results

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='YOLO Training Script for Bone Fracture Detection')
    parser.add_argument('--model', type=str, default='yolo12s',
                        help='Model size: yolo12n, yolo12s, yolo12m, yolo12l, yolo12x (or yolov8n, yolov8s, etc.)')
    parser.add_argument('--data', type=str, default='data.yaml',
                        help='Path to data.yaml configuration file')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--batch', type=int, default=32,
                        help='Batch size (larger = faster, adjust based on GPU memory)')
    parser.add_argument('--imgsz', type=int, default=416,
                        help='Image size (smaller = faster, default 416 instead of 640)')
    parser.add_argument('--amp', action='store_true', default=True,
                        help='Enable mixed precision training (faster)')
    parser.add_argument('--workers', type=int, default=8,
                        help='Number of data loading workers')
    parser.add_argument('--no-amp', action='store_true',
                        help='Disable mixed precision training (slower but more stable)')
    parser.add_argument('--device', type=str, default='cpu',
                        help='Device: cpu, cuda, or GPU number')
    parser.add_argument('--project', type=str, default='results',
                        help='Project directory')
    parser.add_argument('--name', type=str, default='model_history',
                        help='Experiment name (saved to results/model_history)')
    parser.add_argument('--patience', type=int, default=50,
                        help='Early stopping patience')
    
    args = parser.parse_args()
    
    # Training mode
    model, results = train_model(
        model_size=args.model,
        data_yaml=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=args.project,
        name=args.name,
        patience=args.patience,
        amp=not args.no_amp,  # Enable AMP by default unless --no-amp is specified
        workers=args.workers
    )
    
    print("\n" + "="*60)
    print("Training process completed successfully!")
    print("="*60)
