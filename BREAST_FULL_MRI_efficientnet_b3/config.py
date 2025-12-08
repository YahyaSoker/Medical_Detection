"""
Configuration file for Breast Cancer MRI Classification
"""
import os

# Get the directory where this config file is located
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Data paths (relative to config file location)
DATA_DIR = os.path.join(BASE_DIR, "breast_mri_dataset")
TRAIN_DIR = os.path.join(DATA_DIR, "train")
VAL_DIR = os.path.join(DATA_DIR, "val")
TEST_DIR = os.path.join(DATA_DIR, "test")

# Model configuration
MODEL_NAME = "efficientnet_b3"  # Options: efficientnet_b3, efficientnet_b4, resnet50, densenet121
NUM_CLASSES = 2
IMAGE_SIZE = 224  # Input image size (224x224)
BATCH_SIZE = 32
NUM_EPOCHS = 20
LEARNING_RATE = 0.001
WEIGHT_DECAY = 1e-4

# Knowledge Distillation configuration
USE_DISTILLATION = True
TEACHER_MODEL = "efficientnet_b4"  # Teacher model (larger, pre-trained)
STUDENT_MODEL = "efficientnet_b0"  # Student model (smaller, to be trained)
TEACHER_MODEL_PATH = None  # Set to None to use ImageNet pretrained weights, or path to trained teacher
DISTILLATION_TEMPERATURE = 4.0  # Temperature for softmax in distillation
DISTILLATION_ALPHA = 0.7  # Weight for distillation loss (1-alpha for hard loss)

# Training configuration
EARLY_STOPPING_PATIENCE = 5
EARLY_STOPPING_MIN_DELTA = 0.001
USE_GPU = True  # Will auto-detect if CUDA is available

# Model save paths (relative to config file location)
MODEL_SAVE_DIR = os.path.join(BASE_DIR, "models")
BEST_MODEL_PATH = os.path.join(MODEL_SAVE_DIR, "best_model.pth")
FINAL_MODEL_PATH = os.path.join(MODEL_SAVE_DIR, "final_model.pth")
TEACHER_MODEL_SAVE_PATH = os.path.join(MODEL_SAVE_DIR, "teacher_model.pth")
STUDENT_MODEL_SAVE_PATH = os.path.join(MODEL_SAVE_DIR, "student_model.pth")

# Results save paths (relative to config file location)
RESULTS_DIR = os.path.join(BASE_DIR, "results")
TRAINING_HISTORY_PATH = os.path.join(RESULTS_DIR, "training_history.csv")
CONFUSION_MATRIX_PATH = os.path.join(RESULTS_DIR, "confusion_matrix.png")
METRICS_REPORT_PATH = os.path.join(RESULTS_DIR, "metrics_report.txt")

# Data augmentation parameters
ROTATION_DEGREES = 15
HORIZONTAL_FLIP_PROB = 0.5
BRIGHTNESS_RANGE = (0.8, 1.2)
CONTRAST_RANGE = (0.8, 1.2)

# Class names
CLASS_NAMES = ["Benign", "Malignant"]

# Create directories if they don't exist
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)


