# Knowledge Distillation Guide

This project supports Knowledge Distillation, a technique where a smaller "student" model learns from a larger "teacher" model.

## How It Works

1. **Teacher Model**: A larger, pre-trained model (e.g., EfficientNet-B4) that provides soft labels
2. **Student Model**: A smaller model (e.g., EfficientNet-B0) that learns from the teacher
3. **Distillation Loss**: Combines:
   - **Hard Loss**: Cross-entropy with true labels
   - **Soft Loss**: KL divergence between teacher and student predictions (with temperature scaling)

## Benefits

- **Model Compression**: Smaller, faster models for deployment
- **Better Performance**: Student often outperforms training from scratch
- **Knowledge Transfer**: Student learns from teacher's learned representations

## Usage

### Step 1: Train Teacher Model (Optional)

If you want to train your own teacher model first:

```python
# In config.py, set:
USE_DISTILLATION = False
MODEL_NAME = "efficientnet_b4"  # or your preferred teacher model

# Train the teacher
python train.py
```

### Step 2: Enable Knowledge Distillation

```python
# In config.py, set:
USE_DISTILLATION = True
TEACHER_MODEL = "efficientnet_b4"  # Larger model
STUDENT_MODEL = "efficientnet_b0"  # Smaller model
TEACHER_MODEL_PATH = "models/best_model.pth"  # Path to trained teacher (optional)
DISTILLATION_TEMPERATURE = 4.0  # Temperature for softmax (higher = softer)
DISTILLATION_ALPHA = 0.7  # Weight for soft loss (0.7 = 70% soft, 30% hard)
```

### Step 3: Train Student Model

```bash
python train.py
```

The student model will learn from both:
- True labels (hard loss)
- Teacher's predictions (soft loss)

## Configuration Parameters

- **DISTILLATION_TEMPERATURE**: Controls softness of probabilities
  - Higher temperature = softer probabilities (more information)
  - Typical range: 3.0 - 10.0
  - Default: 4.0

- **DISTILLATION_ALPHA**: Weight for combining losses
  - Alpha = 0.7 means 70% soft loss, 30% hard loss
  - Typical range: 0.5 - 0.9
  - Default: 0.7

## Example Configurations

### EfficientNet-B4 → EfficientNet-B0
```python
TEACHER_MODEL = "efficientnet_b4"
STUDENT_MODEL = "efficientnet_b0"
DISTILLATION_TEMPERATURE = 4.0
DISTILLATION_ALPHA = 0.7
```

### EfficientNet-B3 → EfficientNet-B1
```python
TEACHER_MODEL = "efficientnet_b3"
STUDENT_MODEL = "efficientnet_b1"
DISTILLATION_TEMPERATURE = 5.0
DISTILLATION_ALPHA = 0.75
```

## Model Size Comparison

- EfficientNet-B0: ~5.3M parameters, fastest
- EfficientNet-B1: ~7.8M parameters
- EfficientNet-B2: ~9.1M parameters
- EfficientNet-B3: ~12M parameters
- EfficientNet-B4: ~19M parameters, slowest but most accurate

## Notes

- Teacher model can use ImageNet pretrained weights (if TEACHER_MODEL_PATH is None)
- Teacher is frozen during training (eval mode)
- Student model is trainable
- Both models run on the same device (GPU recommended)

