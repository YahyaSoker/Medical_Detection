# Brain Cancer Prediction System

A comprehensive deep learning system for brain tumor detection and classification using CNN-based models with advanced training techniques.

## Features

- **Multi-class Classification**: Detects No Tumor, Benign, and Malignant brain tumors
- **Transfer Learning**: Uses pre-trained ResNet50 for better performance
- **Data Augmentation**: Advanced augmentation techniques for improved generalization
- **Class Balancing**: Handles imbalanced datasets with weighted loss
- **Comprehensive Evaluation**: Detailed metrics and visualizations
- **Real-time Prediction**: Single image prediction capabilities
- **Model Visualization**: Training curves, confusion matrix, and ROC curves

## Dataset Structure

The system expects the following directory structure:
```
brain cancer/
├── train/
│   ├── _annotations.coco.json
│   └── *.jpg (training images)
├── valid/
│   ├── _annotations.coco.json
│   └── *.jpg (validation images)
├── test/
│   ├── _annotations.coco.json
│   └── *.jpg (test images)
└── README.txt
```

## Installation

1. **Install Python dependencies:**
```bash
pip install -r requirements.txt
```

2. **Verify PyTorch installation:**
```bash
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
```

## Usage

### 1. Training the Model

Train a new model on your dataset:

```bash
python train_model.py
```

This will:
- Load and preprocess the dataset
- Train a ResNet50-based model
- Generate training visualizations
- Save the trained model as `brain_cancer_model.pth`
- Create evaluation plots (confusion matrix, ROC curves)

### 2. Single Image Prediction

Predict the class of a single brain image:

```bash
python predict_single.py path/to/your/image.jpg
```

Options:
- `--model model_path`: Specify custom model path (default: `brain_cancer_model.pth`)
- `--no-display`: Skip image visualization

Example:
```bash
python predict_single.py test/brain_scan.jpg --model brain_cancer_model.pth
```

### 3. Programmatic Usage

```python
from brain_cancer_prediction_system import BrainCancerPredictor

# Initialize predictor
predictor = BrainCancerPredictor(model_type='resnet', num_classes=3)

# Load trained model
predictor.load_model('brain_cancer_model.pth')

# Predict single image
result = predictor.predict_single_image('path/to/image.jpg')
print(f"Predicted: {result['class_name']} (confidence: {result['confidence']:.2f})")
```

## Model Architecture

### Custom CNN
- 4 convolutional layers with batch normalization
- Global average pooling
- Dropout for regularization
- Fully connected layers for classification

### Transfer Learning (ResNet50)
- Pre-trained ResNet50 backbone
- Custom classifier head
- Fine-tuning capabilities

## Training Features

- **Early Stopping**: Prevents overfitting
- **Learning Rate Scheduling**: Adaptive learning rate reduction
- **Class Weighting**: Handles imbalanced datasets
- **Data Augmentation**: Random flips, rotations, color jittering
- **Progress Tracking**: Real-time training metrics

## Evaluation Metrics

- **Accuracy**: Overall classification accuracy
- **Precision/Recall**: Per-class performance
- **F1-Score**: Harmonic mean of precision and recall
- **Confusion Matrix**: Detailed classification breakdown
- **ROC Curves**: Multi-class ROC analysis
- **AUC Scores**: Area under the curve metrics

## Output Files

After training, the system generates:

- `brain_cancer_model.pth`: Trained model weights
- `training_history.png`: Training/validation curves
- `confusion_matrix.png`: Classification confusion matrix
- `roc_curves.png`: ROC curves for each class
- `prediction_result.png`: Single image prediction visualization

## Class Definitions

- **No Tumor (Class 0)**: No detectable brain tumor
- **Benign (Class 1)**: Non-cancerous brain tumor
- **Malignant (Class 2)**: Cancerous brain tumor

## Hardware Requirements

- **GPU**: Recommended for faster training (CUDA-compatible)
- **RAM**: Minimum 8GB (16GB+ recommended)
- **Storage**: 2GB+ for model and data

## Performance Tips

1. **Use GPU**: Training is significantly faster with CUDA
2. **Batch Size**: Adjust based on available memory
3. **Epochs**: Monitor validation loss to prevent overfitting
4. **Data Quality**: Ensure high-quality, properly labeled images

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**:
   - Reduce batch size in `train_model.py`
   - Use CPU training: Set `device='cpu'`

2. **Data Loading Errors**:
   - Verify COCO annotation files exist
   - Check image file formats (JPG/PNG)

3. **Model Loading Errors**:
   - Ensure model file exists and is compatible
   - Check model type matches predictor initialization

### Performance Optimization

- Use mixed precision training for faster training
- Implement gradient accumulation for larger effective batch sizes
- Use data parallel training for multiple GPUs

## License

This project uses the CC BY 4.0 license as specified in the dataset.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## Citation

If you use this system in your research, please cite:

```bibtex
@software{brain_cancer_prediction,
  title={Brain Cancer Prediction System},
  author={Your Name},
  year={2024},
  url={https://github.com/your-repo/brain-cancer-prediction}
}
```

## Support

For issues and questions:
- Check the troubleshooting section
- Review the code documentation
- Open an issue on GitHub

---

**Note**: This system is for research and educational purposes. Always consult medical professionals for actual medical diagnosis.

