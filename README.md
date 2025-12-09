# Medical Detection & AI Applications Worktree

This worktree combines multiple medical detection and AI applications, integrating deep learning models, computer vision techniques, and large language models (LLMs) to create comprehensive medical imaging analysis systems. The project encompasses various medical domains including cancer detection, tissue segmentation, and diagnostic assistance.

## 🎯 Project Overview

This repository consolidates several medical AI applications that leverage state-of-the-art deep learning architectures for medical image analysis. The worktree demonstrates the integration of:

- **Medical Detection Systems**: Object detection and classification models for identifying medical conditions
- **Medical AI Applications**: Advanced AI systems combining computer vision with natural language processing for diagnostic assistance
- **Segmentation Models**: Pixel-level segmentation for precise tissue and anomaly identification
- **LLM Integration**: Large language models for intelligent medical report generation and consultation

## 📁 Project Structure

```
DAI/
├── Brain_Cancer/                          # Brain tumor classification system
│   ├── train_model.py                     # Training script with ResNet50
│   ├── predict_single.py                  # Single image prediction
│   └── best_brain_cancer_model.pth        # Trained classification model
│
├── Brain_Cancer_Segmentation/              # Brain cancer pixel-level segmentation
│   ├── brain_cancer_prediction.py         # DeepLabV3 segmentation predictions
│   ├── models/                            # Trained segmentation models
│   └── results/                           # Segmentation visualization results
│
├── Breast_Cancer_Detection/                # YOLO-based mammography detection
│   ├── main.py                            # YOLO prediction pipeline
│   ├── Yolo_Detection_Mamografi_*.pt      # YOLO detection model
│   └── pred/                              # Detection results and reports
│
├── breast_cancer_GPT/                      # Breast cancer detection with LLM integration
│   ├── main.py                            # YOLO + LLM prediction system
│   ├── app.py                             # Streamlit web interface
│   ├── llm_local.py                       # Local LLM integration
│   └── pred/                              # AI-generated reports and predictions
│
├── BREAST_FULL_MRI_efficientnet_b3/       # MRI-based breast cancer classification
│   ├── train.py                           # EfficientNet-B3 training
│   ├── predict.py                         # MRI prediction system
│   ├── model.py                           # Model architecture definitions
│   └── KNOWLEDGE_DISTILLATION.md          # Knowledge distillation guide
│
├── Tooth_Decay/                            # Dental decay and restoration segmentation
│   ├── predict.py                         # DeepLabV3 tooth segmentation
│   ├── models/                            # Trained dental models
│   └── dataset/                           # Dental image datasets
│
└── Bone_Death_Tissue_Segmentation/         # Bone necrosis tissue segmentation
    ├── simple_test.py                     # Tissue segmentation testing
    └── DIA*.pth                           # Trained segmentation models
```

## 🔬 Applications & Technologies

### 1. Brain Cancer Detection & Segmentation

**Brain_Cancer/** - Multi-class classification system
- **Technology**: ResNet50 transfer learning
- **Task**: Classify brain images into No Tumor, Benign, or Malignant
- **Features**: 
  - Advanced data augmentation
  - Class balancing with weighted loss
  - Comprehensive evaluation metrics (ROC curves, confusion matrices)
  - Real-time single image prediction

**Brain_Cancer_Segmentation/** - Pixel-level cancer segmentation
- **Technology**: DeepLabV3-ResNet50
- **Task**: Precise segmentation of brain cancer regions
- **Features**:
  - Binary segmentation (background vs. cancer)
  - Multiple visualization methods (masks, overlays, probability maps)
  - Comprehensive statistics and region analysis
  - Batch processing capabilities

### 2. Breast Cancer Detection Systems

**Breast_Cancer_Detection/** - YOLO object detection
- **Technology**: YOLO (You Only Look Once) object detection
- **Task**: Detect and localize breast abnormalities in mammography images
- **Features**:
  - Automatic batch processing
  - Bounding box visualization
  - JSON results export
  - Summary report generation

**breast_cancer_GPT/** - AI-powered diagnostic assistant
- **Technology**: YOLO detection + Large Language Models (LLMs)
- **Task**: Combine detection with intelligent report generation
- **Features**:
  - YOLO-based detection pipeline
  - LLM integration for medical report generation
  - Streamlit web interface
  - Doctor's assistant chat functionality
  - Context-aware medical consultations

**BREAST_FULL_MRI_efficientnet_b3/** - MRI classification system
- **Technology**: EfficientNet-B3 with knowledge distillation support
- **Task**: Classify breast MRI images as Benign or Malignant
- **Features**:
  - EfficientNet architecture for optimal accuracy/speed trade-off
  - Knowledge distillation (teacher-student learning)
  - Comprehensive training pipeline
  - Model evaluation and metrics

### 3. Dental & Orthopedic Applications

**Tooth_Decay/** - Dental restoration segmentation
- **Technology**: DeepLabV3-ResNet50
- **Task**: Multi-class segmentation of dental conditions
- **Classes**: Background, Dolgu (Filling), Kanal (Root Canal), Çürük (Decay), Protez (Prosthesis)
- **Features**:
  - Multi-class semantic segmentation
  - Color-coded visualization
  - Detailed statistics per category
  - Composite result visualization

**Bone_Death_Tissue_Segmentation/** - Bone necrosis detection
- **Technology**: Deep learning segmentation models
- **Task**: Identify and segment necrotic bone tissue
- **Features**:
  - Tissue segmentation
  - Interactive testing scripts
  - Model checkpoint management

## 🛠️ Technologies & Frameworks

### Deep Learning Frameworks
- **PyTorch**: Primary deep learning framework
- **Torchvision**: Pre-trained models and transforms
- **Ultralytics YOLO**: Object detection models

### Computer Vision
- **OpenCV**: Image processing
- **PIL/Pillow**: Image manipulation
- **Matplotlib**: Visualization and plotting

### Natural Language Processing
- **Local LLM Integration**: Medical report generation
- **Streamlit**: Web interface for AI assistant

### Model Architectures
- **ResNet50**: Transfer learning for classification
- **DeepLabV3**: Semantic segmentation
- **EfficientNet**: Efficient classification models
- **YOLO**: Real-time object detection

## 🚀 Getting Started

### Prerequisites

- Python 3.7+
- CUDA-compatible GPU (recommended for training)
- 8GB+ RAM (16GB+ recommended)
- 10GB+ disk space for models and datasets

### Installation

Each subdirectory contains its own `requirements.txt`. Install dependencies for specific applications:

```bash
# Example: Install dependencies for Brain Cancer Detection
cd Brain_Cancer
pip install -r requirements.txt

# Example: Install dependencies for Breast Cancer Detection
cd Breast_Cancer_Detection
pip install -r requirements.txt
```

### Common Dependencies

Most projects require:
```bash
pip install torch torchvision
pip install opencv-python pillow matplotlib
pip install numpy tqdm
pip install ultralytics  # For YOLO projects
pip install streamlit    # For web interfaces
```

## 📊 Key Features Across Applications

### 1. **Detection & Classification**
   - Multi-class medical condition classification
   - Object detection with bounding boxes
   - Confidence scoring and uncertainty quantification

### 2. **Segmentation**
   - Pixel-level precise segmentation
   - Multi-class tissue identification
   - Overlay visualizations and probability maps

### 3. **AI Integration**
   - LLM-powered medical report generation
   - Interactive diagnostic assistance
   - Context-aware medical consultations

### 4. **Comprehensive Analysis**
   - Statistical summaries
   - Visual result generation
   - Export capabilities (JSON, images, reports)

## 🎓 Usage Examples

### Brain Cancer Classification
```bash
cd Brain_Cancer
python predict_single.py path/to/brain_scan.jpg
```

### Breast Cancer Detection with AI Assistant
```bash
cd breast_cancer_GPT
streamlit run app.py
# Or run the command-line version
python main.py
```

### Tooth Decay Segmentation
```bash
cd Tooth_Decay
python predict.py
# Processes images from target/data/ and saves to target/pred/
```

### MRI Breast Cancer Classification
```bash
cd BREAST_FULL_MRI_efficientnet_b3
python train.py    # Train model
python predict.py  # Make predictions
```

## 📈 Model Performance & Evaluation

Each application includes comprehensive evaluation metrics:

- **Classification Metrics**: Accuracy, Precision, Recall, F1-Score, ROC-AUC
- **Segmentation Metrics**: IoU (Intersection over Union), Dice Score
- **Detection Metrics**: mAP (mean Average Precision), Confidence scores
- **Visualization**: Confusion matrices, training curves, result overlays

## 🔄 Workflow Integration

This worktree demonstrates a complete medical AI workflow:

1. **Data Preparation**: COCO format annotations, dataset organization
2. **Model Training**: Transfer learning, fine-tuning, knowledge distillation
3. **Inference**: Single image and batch processing
4. **Visualization**: Comprehensive result visualization
5. **AI Enhancement**: LLM integration for intelligent reporting
6. **Deployment**: Web interfaces and command-line tools

## 🏥 Medical Domains Covered

- **Oncology**: Brain cancer, breast cancer detection and segmentation
- **Dentistry**: Tooth decay and restoration identification
- **Orthopedics**: Bone necrosis tissue segmentation
- **Radiology**: Mammography and MRI analysis

## ⚠️ Important Notes

### Medical Disclaimer
**This software is for research and educational purposes only. It is NOT intended for clinical diagnosis or treatment decisions. Always consult qualified medical professionals for actual medical diagnosis and treatment.**

### Data Privacy
- Ensure compliance with HIPAA, GDPR, and local medical data regulations
- Medical images should be properly anonymized
- Follow institutional data handling policies

### Model Limitations
- Models are trained on specific datasets and may not generalize to all populations
- Performance depends on image quality and acquisition parameters
- Regular model validation and updates are recommended

## 📝 Contributing

When contributing to this worktree:

1. Maintain clear separation between different medical applications
2. Include comprehensive documentation for each module
3. Follow medical data privacy guidelines
4. Add appropriate disclaimers and citations
5. Test thoroughly before integration

## 📚 Documentation

Each subdirectory contains detailed README files with:
- Installation instructions
- Usage examples
- Model architecture details
- Configuration options
- Troubleshooting guides

## 🔗 Related Resources

- PyTorch Documentation: https://pytorch.org/docs/
- Ultralytics YOLO: https://docs.ultralytics.com/
- Medical Imaging Datasets: Various public and private sources
- Streamlit: https://docs.streamlit.io/

## 📄 License

Please refer to individual project directories for specific licensing information. Most projects follow research and educational use licenses.

## 🙏 Acknowledgments

This worktree combines various medical AI applications and leverages:
- Pre-trained models from PyTorch and Torchvision
- YOLO detection models from Ultralytics
- Public medical imaging datasets
- Open-source deep learning frameworks

---

**Last Updated**: 2025  
**Maintainer**: Yahya 
**Status**: Active Development

