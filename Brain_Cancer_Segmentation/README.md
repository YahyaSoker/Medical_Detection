# Brain Cancer Segmentation System

This system provides both training and prediction capabilities for brain cancer segmentation using deep learning models.

## Files

- `brain_cancer_training.py` - Training script for the segmentation model
- `brain_cancer_prediction.py` - Prediction system for trained models
- `brain cancer.json` - COCO format annotations for brain cancer images
- `train/` - Directory containing segmented brain cancer images
- `full/` - Directory containing all brain cancer images
- `requirements.txt` - Python dependencies

## Installation

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Training a Model

To train a brain cancer segmentation model:

```python
python brain_cancer_training.py
```

This will:
- Load the COCO format annotations from `brain cancer.json`
- Use images from the `train/` directory
- Train a DeepLabV3 model for brain cancer segmentation
- Save the best model to `models/best_model.pth`
- Generate training history plots

### Making Predictions

#### Single Image Prediction

```python
from brain_cancer_prediction import predict_single

# Predict on a single image
pred_mask, prob_map, stats = predict_single(
    model_path='models/best_model.pth',
    image_path='train/103_jpg.rf.545b7cac5ee5582ca96be24af6e900fe.jpg',
    output_dir='results'
)
```

#### Batch Prediction

```python
from brain_cancer_prediction import predict_folder

# Predict on all images in a folder
results = predict_folder(
    model_path='models/best_model.pth',
    folder_path='train',
    output_dir='results',
    limit=10  # Optional: limit number of images
)
```

#### Quick Demo

```python
from brain_cancer_prediction import demo

# Quick prediction with default settings
demo('train/103_jpg.rf.545b7cac5ee5582ca96be24af6e900fe.jpg')
```

### Command Line Usage

You can also run the prediction system directly:

```bash
python brain_cancer_prediction.py
```

This will automatically process images from the `train/` directory and save results to the `results/` directory.

## Features

### Training Features
- DeepLabV3-ResNet50 architecture
- Combined Cross-Entropy and Dice Loss
- IoU (Intersection over Union) metrics
- Automatic model checkpointing
- Training history visualization
- GPU support with automatic device detection

### Prediction Features
- Multiple visualization methods:
  - Original image
  - Binary cancer mask
  - Probability maps (Jet and Hot colormaps)
  - Contour detection
  - Overlay visualizations (red and green)
  - High confidence regions
  - Edge detection
  - Region analysis with area statistics
- Comprehensive statistics:
  - Total pixels
  - Cancer pixels
  - Cancer percentage
  - Average confidence score
- Batch processing capabilities
- Automatic result saving

## Model Architecture

The system uses DeepLabV3 with ResNet50 backbone, modified for binary segmentation:
- Input: 640x640 RGB images
- Output: 640x640 binary masks (background + cancer)
- Classes: 2 (Background: 0, Cancer: 1)

## Data Format

The system expects:
- COCO format JSON annotations
- Images in common formats (JPG, PNG, etc.)
- 640x640 pixel images (automatically resized if different)

## Output

### Training Output
- `models/best_model.pth` - Best model checkpoint
- `models/checkpoint_epoch_X.pth` - Epoch checkpoints
- `models/training_history_YYYYMMDD_HHMMSS.png` - Training plots

### Prediction Output
- `results/[image_name]_cancer_result_YYYYMMDD_HHMMSS.png` - Comprehensive visualization
- `results/[image_name]_cancer_mask_YYYYMMDD_HHMMSS.png` - Binary mask
- Console output with detailed statistics

## Requirements

- Python 3.7+
- PyTorch 1.9+
- CUDA support (optional, for GPU acceleration)
- 8GB+ RAM recommended
- 2GB+ disk space for models and results

## Notes

- The system automatically detects and uses GPU if available
- Images are automatically resized to 640x640 pixels
- All results include timestamps for easy organization
- The prediction system provides 10 different visualization methods for comprehensive analysis
