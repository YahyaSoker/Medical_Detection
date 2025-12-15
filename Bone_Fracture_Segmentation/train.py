"""
Training Script for Bone Fracture Segmentation Model
Uses YOLOv8-seg for instance segmentation with accuracy optimization
"""

import os
import shutil
from pathlib import Path
from ultralytics import YOLO
import yaml

# Paths
BASE_DIR = Path(__file__).parent
DATA_YAML = BASE_DIR / "BoneFractureYolo8" / "data.yaml"
OUTPUT_DIR = BASE_DIR / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
TRAINING_OUTPUT_DIR = OUTPUT_DIR / "training"
TRAINING_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Model files
BEST_MODEL_PATH = OUTPUT_DIR / "best.pt"
LAST_MODEL_PATH = OUTPUT_DIR / "last.pt"

class BestModelCallback:
    """Callback to manage best model saving"""
    def __init__(self, best_model_path):
        self.best_model_path = best_model_path
        self.best_metric = -1.0
        self.best_epoch = 0
    
    def on_train_epoch_end(self, trainer):
        """Called at the end of each training epoch"""
        # Get current metrics
        metrics = trainer.metrics
        if metrics is None:
            return
        
        # Track mAP50-95 (segmentation metric)
        current_metric = getattr(metrics, 'metrics/mAP50-95(B)', None)
        if current_metric is None:
            # Fallback to mAP50 if mAP50-95 not available
            current_metric = getattr(metrics, 'metrics/mAP50(B)', None)
        
        if current_metric is not None and current_metric > self.best_metric:
            self.best_metric = current_metric
            self.best_epoch = trainer.epoch
            
            # Delete old best model if exists
            if self.best_model_path.exists():
                try:
                    self.best_model_path.unlink()
                    print(f"\n[Epoch {trainer.epoch}] Deleted old best model")
                except Exception as e:
                    print(f"Warning: Could not delete old best model: {e}")
            
            # Save new best model
            try:
                shutil.copy(trainer.last, self.best_model_path)
                print(f"[Epoch {trainer.epoch}] New best model saved! mAP50-95: {current_metric:.4f}")
            except Exception as e:
                print(f"Warning: Could not save best model: {e}")

def train_model():
    """Train YOLOv8 segmentation model"""
    print("=" * 60)
    print("Bone Fracture Segmentation Model Training")
    print("=" * 60)
    
    # Load data config
    with open(DATA_YAML, 'r') as f:
        config = yaml.safe_load(f)
    
    print(f"\nDataset Configuration:")
    print(f"  Classes: {config['nc']}")
    print(f"  Class names: {config['names']}")
    print(f"  Data YAML: {DATA_YAML}")
    
    # Initialize model - using yolov8s-seg for better accuracy
    # Can use yolov8m-seg.pt or yolov8l-seg.pt for even better accuracy
    model_name = "yolov8s-seg.pt"  # Small model for good balance
    print(f"\nLoading model: {model_name}")
    model = YOLO(model_name)
    
    # Training parameters optimized for accuracy
    print("\nTraining Parameters:")
    print("  Model: YOLOv8-seg (Segmentation)")
    print("  Epochs: 150")
    print("  Image size: 640")
    print("  Batch size: Auto")
    print("  Learning rate: 0.01")
    print("  Augmentation: Enabled")
    print("  Early stopping: Enabled (patience=50)")
    
    # Delete old best model if exists
    if BEST_MODEL_PATH.exists():
        try:
            BEST_MODEL_PATH.unlink()
            print(f"\nDeleted existing best model at {BEST_MODEL_PATH}")
        except Exception as e:
            print(f"Warning: Could not delete old best model: {e}")
    
    # Train the model
    try:
        results = model.train(
            data=str(DATA_YAML),
            epochs=150,
            imgsz=640,
            batch=-1,  # Auto batch size
            lr0=0.01,
            lrf=0.01,
            momentum=0.937,
            weight_decay=0.0005,
            warmup_epochs=3.0,
            warmup_momentum=0.8,
            warmup_bias_lr=0.1,
            box=7.5,
            cls=0.5,
            dfl=1.5,
            pose=12.0,
            kobj=1.0,
            label_smoothing=0.0,
            nbs=64,
            hsv_h=0.015,
            hsv_s=0.7,
            hsv_v=0.4,
            degrees=0.0,
            translate=0.1,
            scale=0.5,
            shear=0.0,
            perspective=0.0,
            flipud=0.0,
            fliplr=0.5,
            mosaic=1.0,
            mixup=0.0,
            copy_paste=0.0,
            auto_augment='randaugment',
            erasing=0.4,
            crop_fraction=1.0,
            patience=50,  # Early stopping patience
            save=True,
            save_period=10,
            val=True,
            plots=True,
            project=str(OUTPUT_DIR),
            name="training",
            exist_ok=True,
            pretrained=True,
            optimizer='auto',
            verbose=True,
            seed=0,
            deterministic=True,
            single_cls=False,
            rect=False,
            cos_lr=False,
            close_mosaic=10,
            resume=False,
            amp=True,
            fraction=1.0,
            profile=False,
            freeze=None,
            # Segmentation specific
            overlap_mask=True,
            mask_ratio=4,
            dropout=0.0,
        )
        
        print("\n" + "=" * 60)
        print("Training completed successfully!")
        print("=" * 60)
        
        # YOLOv8 saves models in the project/name/weights/ directory
        # Since we set project=OUTPUT_DIR and name="training", models should be in:
        # OUTPUT_DIR/training/weights/
        weights_dir = OUTPUT_DIR / "training" / "weights"
        
        # Also check runs directory (default YOLOv8 location)
        if not weights_dir.exists():
            weights_dir = Path("runs") / "seg" / "training" / "weights"
        
        # If still not found, search recursively
        if not weights_dir.exists():
            training_dirs = list(BASE_DIR.rglob("**/weights"))
            if training_dirs:
                # Get the most recently modified weights directory
                weights_dir = max(training_dirs, key=lambda x: x.stat().st_mtime if x.exists() else 0)
        
        if weights_dir.exists():
            best_in_runs = weights_dir / "best.pt"
            last_in_runs = weights_dir / "last.pt"
            
            # Copy best model (delete old first)
            if best_in_runs.exists():
                if BEST_MODEL_PATH.exists():
                    BEST_MODEL_PATH.unlink()
                shutil.copy(best_in_runs, BEST_MODEL_PATH)
                print(f"\nBest model saved to: {BEST_MODEL_PATH}")
            else:
                print(f"\nWarning: best.pt not found in {weights_dir}")
            
            # Copy last model
            if last_in_runs.exists():
                if LAST_MODEL_PATH.exists():
                    LAST_MODEL_PATH.unlink()
                shutil.copy(last_in_runs, LAST_MODEL_PATH)
                print(f"Last model saved to: {LAST_MODEL_PATH}")
            else:
                print(f"Warning: last.pt not found in {weights_dir}")
        else:
            print(f"\nWarning: Could not find weights directory. Models may be in a different location.")
            print(f"Searched in: {OUTPUT_DIR / 'training' / 'weights'}")
            print(f"Please check the YOLOv8 output directory for model files.")
        
        # Copy training plots
        # Find results.png in training directories
        possible_plot_paths = [
            Path("runs") / "seg" / "training" / "results.png",
            OUTPUT_DIR / "training" / "results.png",
            OUTPUT_DIR / "training" / "train" / "results.png",
        ]
        
        for plot_path in possible_plot_paths:
            if plot_path.exists():
                shutil.copy(plot_path, TRAINING_OUTPUT_DIR / "training_results.png")
                print(f"Training plots saved to: {TRAINING_OUTPUT_DIR / 'training_results.png'}")
                break
        
        # Also search recursively
        if not (TRAINING_OUTPUT_DIR / "training_results.png").exists():
            for results_file in OUTPUT_DIR.rglob("results.png"):
                shutil.copy(results_file, TRAINING_OUTPUT_DIR / "training_results.png")
                print(f"Training plots saved to: {TRAINING_OUTPUT_DIR / 'training_results.png'}")
                break
        
        # Print final metrics
        if hasattr(results, 'results_dict'):
            print("\nFinal Training Metrics:")
            for key, value in results.results_dict.items():
                if isinstance(value, (int, float)):
                    print(f"  {key}: {value:.4f}")
        
        print(f"\nAll outputs saved to: {OUTPUT_DIR}")
        print("=" * 60)
        
    except Exception as e:
        print(f"\nError during training: {e}")
        import traceback
        traceback.print_exc()
        raise

if __name__ == "__main__":
    train_model()

