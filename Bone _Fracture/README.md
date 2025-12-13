# Bone Fracture Detection using YOLO11

A complete deep learning pipeline for detecting bone fractures in medical X-ray images using YOLO11 (You Only Look Once version 11). This project includes training scripts, inference tools, and optimized configurations specifically designed for medical imaging applications.

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Training](#training)
- [Inference/Prediction](#inferenceprediction)
- [Results](#results)
- [Configuration](#configuration)
- [Troubleshooting](#troubleshooting)
- [License](#license)

## 🎯 Overview

This project implements a YOLO11-based object detection system for identifying bone fractures in X-ray images. The system can detect 7 different types of bone fractures and conditions across various body parts including elbows, fingers, forearms, humerus, shoulders, and wrists.

**Key Capabilities:**
- Automated training pipeline with best model tracking
- Batch inference on validation/test images
- Reproducible results with fixed random seeds
- Optimized hyperparameters for medical imaging
- Automatic GPU/CPU detection
- Comprehensive logging and statistics

## ✨ Features

### Training Features
- **Automatic Best Model Tracking**: Monitors validation metrics and saves best models automatically
- **Timestamped Experiments**: Each training run creates a unique timestamped directory
- **Early Stopping**: Prevents overfitting with configurable patience
- **Medical Imaging Optimizations**: Conservative data augmentation suitable for X-ray images
- **Multiple Model Sizes**: Support for nano (n), small (s), medium (m), large (l), and xlarge (x) models
- **Comprehensive Logging**: Saves training curves, metrics, and checkpoints

### Inference Features
- **Batch Processing**: Processes all images in a directory efficiently
- **Selective Output**: Only saves images with detections to reduce storage
- **Annotated Results**: Saves images with bounding boxes, labels, and confidence scores
- **Reproducible Results**: Fixed seeds ensure consistent outputs across runs
- **Device Auto-Detection**: Automatically uses GPU if available

## 📊 Dataset

The project uses the Bone Fracture Detection dataset from Roboflow:

- **Source**: [Roboflow Universe - Bone Fracture Detection](https://universe.roboflow.com/veda/bone-fracture-detection-daoon/dataset/4)
- **License**: CC BY 4.0
- **Classes**: 7 fracture types/conditions
- **Format**: YOLO format with train/validation/test splits

### Detection Classes

1. `elbow positive` - Elbow fractures/conditions
2. `fingers positive` - Finger fractures/conditions
3. `forearm fracture` - Forearm bone fractures
4. `humerus fracture` - Humerus bone fractures
5. `humerus` - Humerus bone (normal or abnormal)
6. `shoulder fracture` - Shoulder fractures
7. `wrist positive` - Wrist fractures/conditions

### Dataset Structure

```
dataset/
├── train/
│   └── images/          # Training images
│   └── labels/         # Training annotations (YOLO format)
├── valid/
│   └── images/         # Validation images
│   └── labels/         # Validation annotations
└── test/
    └── images/         # Test images
    └── labels/         # Test annotations
```

## 🏗️ Model Architecture

This project uses **YOLO11** (Ultralytics YOLO version 11), the latest iteration of the YOLO object detection architecture. YOLO11 offers:

- **Improved Accuracy**: Better detection performance than previous versions
- **Multiple Sizes**: From nano (fastest) to xlarge (most accurate)
- **Efficient Inference**: Optimized for both speed and accuracy
- **Medical Imaging Ready**: Works well with grayscale X-ray images

### Model Size Options

| Size | Parameters | Speed | Accuracy | Use Case |
|------|-----------|-------|----------|----------|
| **n** (nano) | ~3M | Fastest | Good | Real-time inference, limited resources |
| **s** (small) | ~12M | Fast | Better | Balanced performance |
| **m** (medium) | ~26M | Moderate | High | Production deployment |
| **l** (large) | ~44M | Slower | Very High | High accuracy requirements |
| **x** (xlarge) | ~68M | Slowest | Highest | Maximum accuracy |

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- CUDA-capable GPU (recommended) or CPU
- 8GB+ RAM (16GB+ recommended)
- 10GB+ free disk space

### Step 1: Clone Repository

```bash
git clone <repository-url>
cd "Bone _Fracture"
```

### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 3: Verify Installation

```bash
python -c "from ultralytics import YOLO; print('YOLO11 installed successfully')"
```

### Step 4: Download Dataset

Ensure your dataset is organized according to the structure specified in `data.yaml`. Update the paths in `data.yaml` to match your dataset location.

## 📁 Project Structure

```
Bone _Fracture/
├── train.py                    # Training script
├── pred.py                      # Inference/prediction script
├── data.yaml                    # Dataset configuration
├── requirements.txt             # Python dependencies
├── README.md                    # This file
│
├── models/                      # Trained models directory
│   ├── best.pt                 # Best model from latest training
│   ├── last.pt                 # Last checkpoint from latest training
│   ├── best_epoch100.pt       # Best model at epoch 100
│   └── BoneYolotest1.pt       # Custom trained model
│
├── results/                     # Training results
│   ├── train_YYYYMMDD_HHMMSS/  # Timestamped training runs
│   │   ├── weights/
│   │   │   ├── best.pt        # Best model weights
│   │   │   └── last.pt        # Last epoch weights
│   │   ├── results.csv        # Training metrics
│   │   ├── confusion_matrix.png
│   │   ├── F1_curve.png
│   │   └── ...
│   └── model_output/           # Inference output images
│
├── train/                       # Training images (if local)
│   └── images/
├── valid/                       # Validation images
│   └── images/
└── test/                        # Test images (if local)
    └── images/
```

## 🎓 Training

### Basic Training

Train a model with default settings:

```bash
python train.py
```

This will:
- Use YOLO11 nano model (smallest, fastest)
- Train for 100 epochs
- Use batch size of 16
- Save results to `results/train_YYYYMMDD_HHMMSS/`
- Automatically save best models to `models/` directory

### Advanced Training Options

```bash
python train.py \
    --model s \                    # Model size: n, s, m, l, x
    --epochs 200 \                 # Number of epochs
    --batch 32 \                   # Batch size (adjust for GPU memory)
    --imgsz 640 \                   # Image size (640 recommended)
    --patience 50 \                 # Early stopping patience
    --device cuda \                 # Device: cuda, cpu, or 0,1,2...
    --data data.yaml \              # Dataset config file
    --project results \             # Project directory
    --name my_experiment            # Custom experiment name
```

### Training Parameters Explained

#### Model Size
```bash
--model n    # Nano: Fastest, ~3M parameters
--model s    # Small: Balanced, ~12M parameters
--model m    # Medium: High accuracy, ~26M parameters
--model l    # Large: Very high accuracy, ~44M parameters
--model x    # XLarge: Maximum accuracy, ~68M parameters
```

#### Batch Size Guidelines
- **GPU Memory 8GB**: Use `--batch 8` or `--batch 16`
- **GPU Memory 16GB**: Use `--batch 16` or `--batch 32`
- **GPU Memory 24GB+**: Use `--batch 32` or `--batch 64`

### Training Configuration

The training script uses optimized hyperparameters for medical imaging:

**Optimizer**: SGD (often better for medical images than Adam)
- Initial learning rate: 0.01
- Final learning rate: 0.0001
- Momentum: 0.937
- Weight decay: 0.0005

**Data Augmentation** (Conservative for X-rays):
- Hue augmentation: 0.015 (very low - preserves X-ray characteristics)
- Saturation: 0.7
- Value: 0.4
- Rotation: ±10° (conservative)
- Vertical flip: 0.0 (disabled - X-rays shouldn't be flipped vertically)
- Horizontal flip: 0.5 (enabled)
- Mosaic: 1.0
- Mixup: 0.1 (low - conservative for medical)

**Loss Function Weights**:
- Box loss: 7.5
- Classification loss: 0.5
- DFL loss: 1.5

### Monitoring Training

During training, you'll see:
- Real-time loss values (train/validation)
- mAP metrics (mAP50, mAP50-95)
- Best model updates
- Training progress bar

Training results are saved in `results/train_YYYYMMDD_HHMMSS/`:
- `weights/best.pt` - Best model based on validation mAP
- `weights/last.pt` - Final epoch model
- `results.csv` - Complete training metrics
- `*.png` - Training curves, confusion matrix, PR curves

### Best Model Tracking

The `BestModelTracker` class automatically:
- Monitors validation metrics during training
- Saves best models to `models/best_epoch{N}.pt` when validation improves
- Copies final `best.pt` and `last.pt` to `models/` directory
- Provides epoch numbers in saved filenames

## 🔍 Inference/Prediction

### Basic Inference

Run inference on validation images:

```bash
python pred.py
```

This will:
- Load model from `models/BoneYolotest1.pt`
- Process all images in `valid/images/`
- Save detected images to `results/model_output/`
- Print summary statistics

### Configuration

Edit `pred.py` to customize:

```python
# Paths (lines 21-23)
VAL_IMAGE_DIR = "valid\images"           # Input directory
MODEL_PATH    = "models\BoneYolotest1.pt" # Model to use
OUTPUT_DIR    = "results\model_output"    # Output directory

# Inference Parameters (line 46)
results = model(val_images, 
               conf=0.15,      # Confidence threshold (0.0-1.0)
               iou=0.45,       # IoU threshold for NMS
               device=device,  # Auto-detected
               verbose=True)   # Detailed logs
```

### Confidence Threshold Tuning

Adjust `conf` parameter based on your needs:

```python
# More sensitive (more detections, may include false positives)
conf=0.1   # Lower threshold
conf=0.05  # Very sensitive

# Less sensitive (fewer detections, higher precision)
conf=0.25  # Standard YOLO default
conf=0.3   # Higher threshold
```

### Output Format

Detected images are saved with:
- **Bounding boxes** around fractures
- **Class labels** (e.g., "forearm fracture")
- **Confidence scores** for each detection
- **Original filename** preserved

### Inference Output Example

```
Using device: cuda
Processing 48 images for inference
Saved: image_001.jpg (2 detection(s))
Saved: image_015.jpg (1 detection(s))
...

============================================================
Summary:
  Total images processed: 48
  Images with detections: 15
  Total detections: 23
  Output directory: results\model_output
============================================================
```

## 📈 Results

### Training Metrics

Key metrics tracked during training:
- **mAP50**: Mean Average Precision at IoU=0.50
- **mAP50-95**: Mean Average Precision averaged over IoU 0.50-0.95
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)
- **F1 Score**: Harmonic mean of precision and recall

### Model Performance

Expected performance (varies by model size and dataset):
- **mAP50**: 0.70-0.90+ (depending on model size)
- **mAP50-95**: 0.50-0.70+ (depending on model size)
- **Inference Speed**: 10-100+ FPS (depending on model size and hardware)

### Viewing Results

Training results can be viewed:
1. **CSV File**: `results/train_*/results.csv` - All metrics per epoch
2. **Plots**: `results/train_*/` - PNG files with training curves
3. **TensorBoard** (if enabled): `tensorboard --logdir results/`

## ⚙️ Configuration

### Dataset Configuration (`data.yaml`)

```yaml
train: ../train/images      # Training images path
val: ../valid/images        # Validation images path
test: ../test/images        # Test images path

nc: 7                       # Number of classes
names:                      # Class names
  - 'elbow positive'
  - 'fingers positive'
  - 'forearm fracture'
  - 'humerus fracture'
  - 'humerus'
  - 'shoulder fracture'
  - 'wrist positive'
```

### Customizing Training

Modify `train.py` function `train_bone_fracture_model()` to adjust:
- Learning rates
- Augmentation parameters
- Loss function weights
- Optimizer settings

## 🔧 Troubleshooting

### Training Issues

**Out of Memory (OOM) Error**
```bash
# Reduce batch size
python train.py --batch 8

# Use smaller model
python train.py --model n

# Reduce image size
python train.py --imgsz 416
```

**Training Too Slow**
```bash
# Use GPU
python train.py --device cuda

# Use smaller model
python train.py --model n

# Increase batch size (if memory allows)
python train.py --batch 32
```

**No Improvement in Validation Metrics**
- Check dataset quality and annotations
- Try different model size (larger model)
- Adjust learning rate
- Increase training epochs
- Check for class imbalance

### Inference Issues

**No Detections Found**
```python
# Lower confidence threshold in pred.py
results = model(val_images, conf=0.1, ...)  # Instead of 0.15
```

**Too Many False Positives**
```python
# Increase confidence threshold
results = model(val_images, conf=0.25, ...)  # Instead of 0.15
```

**GPU Not Detected**
```bash
# Check CUDA installation
python -c "import torch; print(torch.cuda.is_available())"

# Verify PyTorch CUDA version matches system CUDA
python -c "import torch; print(torch.version.cuda)"
```

**Different Results from Kaggle/Other Environments**
- Ensure same model file is used
- Check confidence threshold matches
- Verify Ultralytics version: `pip show ultralytics`
- Check random seeds are set (already handled in pred.py)

### Common Errors

**FileNotFoundError: data.yaml**
- Ensure `data.yaml` exists in project root
- Update paths in `data.yaml` to match your dataset structure

**ModuleNotFoundError: ultralytics**
```bash
pip install ultralytics>=8.0.0
```

**CUDA Out of Memory**
- Reduce batch size
- Use smaller model
- Reduce image size
- Close other GPU applications

## 📝 Reproducibility

Both training and inference scripts use fixed random seeds for reproducibility:

- **Python random**: seed=42
- **NumPy**: seed=42
- **PyTorch**: seed=42 (CPU and CUDA)

This ensures consistent results across different runs and environments.

## 📄 License

This project uses the Bone Fracture Detection dataset from Roboflow, which is licensed under **CC BY 4.0**. See `data.yaml` for dataset details.

**Dataset Source**: [Roboflow Universe - Bone Fracture Detection](https://universe.roboflow.com/veda/bone-fracture-detection-daoon/dataset/4)

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📚 References

- [Ultralytics YOLO11 Documentation](https://docs.ultralytics.com/)
- [YOLO11 Paper](https://arxiv.org/abs/2405.14458)
- [Roboflow Dataset](https://universe.roboflow.com/veda/bone-fracture-detection-daoon/dataset/4)

## 📧 Support

For issues, questions, or contributions, please open an issue on the repository.

---

**Note**: This project is for research and educational purposes. Medical diagnosis should always be performed by qualified healthcare professionals.
