"""
YOLO11 Bone Fracture Detection Training Script

This script trains a YOLO11 model for bone fracture detection with:
- Automatic best model tracking and saving to models/ directory
- Timestamped results directories
- Optimized parameters for medical imaging
"""

from ultralytics import YOLO
import os
import shutil
import time
from datetime import datetime
from pathlib import Path
import threading


class BestModelTracker:
    """Callback class to track and save best models on validation improvement."""
    
    def __init__(self, models_dir="models", results_dir=None):
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(exist_ok=True)
        self.results_dir = Path(results_dir) if results_dir else None
        self.best_map50_95 = -1.0
        self.best_epoch = 0
        self.last_best_mtime = 0
        self.monitoring = False
        
    def start_monitoring(self, results_dir):
        """Start monitoring the results directory for best.pt updates."""
        self.results_dir = Path(results_dir)
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_best_model, daemon=True)
        self.monitor_thread.start()
        
    def _monitor_best_model(self):
        """Monitor best.pt file for changes and copy to models/ directory."""
        weights_dir = self.results_dir / "weights"
        best_model_path = weights_dir / "best.pt"
        
        while self.monitoring:
            try:
                if best_model_path.exists():
                    # Check if file was modified
                    current_mtime = best_model_path.stat().st_mtime
                    
                    if current_mtime > self.last_best_mtime:
                        self.last_best_mtime = current_mtime
                        
                        # Try to read epoch from results.csv if available
                        results_csv = self.results_dir / "results.csv"
                        epoch = self._get_current_epoch(results_csv)
                        
                        if epoch > 0:
                            # Copy best model to models directory with epoch number
                            dest_path = self.models_dir / f"best_epoch{epoch}.pt"
                            try:
                                shutil.copy2(best_model_path, dest_path)
                                print(f"\n✓ New best model saved: {dest_path} (Epoch {epoch})")
                            except Exception as e:
                                print(f"Warning: Could not copy best model: {e}")
                
                time.sleep(2)  # Check every 2 seconds
            except Exception as e:
                if self.monitoring:  # Only print if still monitoring
                    pass  # Silent fail during monitoring
                time.sleep(2)
    
    def _get_current_epoch(self, results_csv):
        """Get the current epoch from results.csv file."""
        try:
            if results_csv.exists():
                with open(results_csv, 'r') as f:
                    lines = f.readlines()
                    if len(lines) > 1:  # Has header + at least one data row
                        # Last line should be the latest epoch
                        last_line = lines[-1].strip()
                        if last_line:
                            parts = last_line.split(',')
                            if len(parts) > 0:
                                return int(float(parts[0]))  # First column is epoch
        except Exception:
            pass
        return 0
    
    def stop_monitoring(self):
        """Stop monitoring."""
        self.monitoring = False
        if hasattr(self, 'monitor_thread'):
            self.monitor_thread.join(timeout=1)


def train_bone_fracture_model(
    model_size="n",  # Options: n, s, m, l, x
    data_yaml="data.yaml",
    epochs=100,
    imgsz=640,
    batch=16,
    patience=50,
    device=None,
    project="results",
    name=None,
    **kwargs
):
    """
    Train YOLO11 model for bone fracture detection.
    
    Args:
        model_size: Model size ('n', 's', 'm', 'l', 'x'). Default 'n' for nano.
        data_yaml: Path to data.yaml configuration file
        epochs: Number of training epochs
        imgsz: Image size for training (640 recommended for fracture detection)
        batch: Batch size (adjust based on GPU memory)
        patience: Early stopping patience
        device: Device to use (None for auto, 'cuda', 'cpu', or device ID)
        project: Project directory name
        name: Experiment name (if None, uses timestamp)
        **kwargs: Additional training arguments
    
    Returns:
        Path to training results directory
    """
    
    # Create timestamped experiment name if not provided
    if name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        name = f"train_{timestamp}"
    
    # Initialize YOLO11 model
    model_name = f"yolo11{model_size}.pt"
    print(f"Loading YOLO11 model: {model_name}")
    model = YOLO(model_name)
    
    # Initialize best model tracker
    tracker = BestModelTracker(models_dir="models")
    
    # Configure training parameters optimized for bone fracture detection
    training_args = {
        "data": data_yaml,
        "epochs": epochs,
        "imgsz": imgsz,
        "batch": batch,
        "patience": patience,
        "project": project,
        "name": name,
        "device": device,
        
        # Optimizations for medical imaging / bone fracture detection
        "optimizer": "SGD",  # SGD often works better for medical imaging
        "lr0": 0.01,  # Initial learning rate
        "lrf": 0.01,  # Final learning rate (lr0 * lrf)
        "momentum": 0.937,  # SGD momentum
        "weight_decay": 0.0005,  # Weight decay
        
        # Data augmentation - conservative for medical images
        "hsv_h": 0.015,  # Hue augmentation (lower for medical images)
        "hsv_s": 0.7,  # Saturation augmentation
        "hsv_v": 0.4,  # Value augmentation
        "degrees": 10.0,  # Rotation degrees (conservative for X-rays)
        "translate": 0.1,  # Translation
        "scale": 0.5,  # Scale augmentation
        "flipud": 0.0,  # Vertical flip probability (0 for X-rays)
        "fliplr": 0.5,  # Horizontal flip probability
        "mosaic": 1.0,  # Mosaic augmentation probability
        "mixup": 0.1,  # Mixup augmentation probability (lower for medical)
        "copy_paste": 0.1,  # Copy-paste augmentation probability
        
        # Loss function parameters
        "box": 7.5,  # Box loss gain
        "cls": 0.5,  # Class loss gain
        "dfl": 1.5,  # DFL loss gain
        
        # Validation settings
        "val": True,  # Validate during training
        "plots": True,  # Generate plots
        "save": True,  # Save checkpoints
        "save_period": 10,  # Save checkpoint every N epochs
        
        # Additional arguments
        **kwargs
    }
    
    print(f"\n{'='*60}")
    print(f"Training Configuration:")
    print(f"{'='*60}")
    print(f"Model: {model_name}")
    print(f"Data: {data_yaml}")
    print(f"Epochs: {epochs}")
    print(f"Image Size: {imgsz}")
    print(f"Batch Size: {batch}")
    print(f"Patience: {patience}")
    print(f"Results Directory: {project}/{name}")
    print(f"{'='*60}\n")
    
    # Get the results directory path before training
    results_dir = Path(project) / name
    
    # Start monitoring for best model updates
    tracker.start_monitoring(results_dir)
    
    # Train the model
    print("Starting training...")
    try:
        results = model.train(**training_args)
    finally:
        # Stop monitoring after training completes
        tracker.stop_monitoring()
    
    # Ensure results_dir exists (in case name was auto-generated)
    if not results_dir.exists():
        # Find the actual results directory (Ultralytics may have created it differently)
        project_path = Path(project)
        if project_path.exists():
            # Get the most recent directory
            dirs = sorted(project_path.glob("train_*"), key=os.path.getmtime, reverse=True)
            if dirs:
                results_dir = dirs[0]
    
    # Copy final last.pt to models directory
    last_model_path = results_dir / "weights" / "last.pt"
    if last_model_path.exists():
        dest_last = Path("models") / "last.pt"
        shutil.copy2(last_model_path, dest_last)
        print(f"\n✓ Final model saved: {dest_last}")
    
    # Also copy final best.pt to models directory
    best_model_path = results_dir / "weights" / "best.pt"
    if best_model_path.exists():
        dest_best = Path("models") / "best.pt"
        shutil.copy2(best_model_path, dest_best)
        print(f"✓ Best model saved: {dest_best}")
    
    # Get final best metrics from results
    best_map50_95 = 0.0
    best_epoch = 0
    results_csv = results_dir / "results.csv"
    if results_csv.exists():
        try:
            with open(results_csv, 'r') as f:
                lines = f.readlines()
                if len(lines) > 1:
                    # Find best mAP50-95 from all epochs
                    header = lines[0].strip().split(',')
                    try:
                        map_idx = header.index('metrics/mAP50-95(B)')
                        epoch_idx = header.index('epoch')
                        best_val = -1.0
                        for line in lines[1:]:
                            parts = line.strip().split(',')
                            if len(parts) > max(map_idx, epoch_idx):
                                try:
                                    map_val = float(parts[map_idx])
                                    epoch_val = int(float(parts[epoch_idx]))
                                    if map_val > best_val:
                                        best_val = map_val
                                        best_epoch = epoch_val
                                        best_map50_95 = map_val
                                except (ValueError, IndexError):
                                    continue
                    except (ValueError, IndexError):
                        pass
        except Exception:
            pass
    
    print(f"\n{'='*60}")
    print(f"Training completed!")
    print(f"Results saved to: {results_dir}")
    if best_map50_95 > 0:
        print(f"Best mAP50-95: {best_map50_95:.4f} (Epoch {best_epoch})")
    print(f"{'='*60}\n")
    
    return results_dir


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train YOLO11 model for bone fracture detection")
    parser.add_argument("--model", type=str, default="n", choices=["n", "s", "m", "l", "x"],
                        help="Model size: n (nano), s (small), m (medium), l (large), x (xlarge)")
    parser.add_argument("--data", type=str, default="data.yaml",
                        help="Path to data.yaml file")
    parser.add_argument("--epochs", type=int, default=100,
                        help="Number of training epochs")
    parser.add_argument("--imgsz", type=int, default=640,
                        help="Image size for training")
    parser.add_argument("--batch", type=int, default=16,
                        help="Batch size")
    parser.add_argument("--patience", type=int, default=50,
                        help="Early stopping patience")
    parser.add_argument("--device", type=str, default=None,
                        help="Device to use (cuda, cpu, or device ID)")
    parser.add_argument("--project", type=str, default="results",
                        help="Project directory name")
    parser.add_argument("--name", type=str, default=None,
                        help="Experiment name (default: timestamp)")
    
    args = parser.parse_args()
    
    # Train the model
    train_bone_fracture_model(
        model_size=args.model,
        data_yaml=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        patience=args.patience,
        device=args.device,
        project=args.project,
        name=args.name
    )
