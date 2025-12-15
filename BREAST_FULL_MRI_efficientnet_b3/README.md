# Breast MRI Classification (Benign vs Malignant) — EfficientNet + Knowledge Distillation

PyTorch image-classification project for **breast MRI** images with:

- **Transfer learning** (EfficientNet / ResNet / DenseNet from `torchvision`)
- Optional **knowledge distillation** (teacher → student) to compress models for deployment
- Simple CLI to **train**, **evaluate**, and **predict**

This repository is intentionally script-based (single-folder Python files) so you can trace the full pipeline end-to-end.

---

## Quickstart

### 1) Install dependencies

```bash
pip install -r requirements.txt
```

### 2) Dataset layout (required)

The loader expects an ImageFolder-style structure:

```
breast_mri_dataset/
  train/
    Benign/
    Malignant/
  val/
    Benign/
    Malignant/
  test/
    Benign/
    Malignant/
```

- **Labels** are hard-coded as: `Benign -> 0`, `Malignant -> 1` (see `data_loader.py`).
- Supported image extensions: `.jpg/.jpeg/.png` (and more for prediction).

### 3) Run via the unified CLI (`main.py`)

```bash
# Train
python main.py train

# Evaluate (writes confusion matrix + metrics report into results/)
python main.py evaluate --model models/best_model.pth

# Predict a single image
python main.py predict --image path/to/image.jpg --model models/best_model.pth --verbose

# Predict a folder of images
python main.py predict --folder path/to/images --model models/best_model.pth
```

You can also run scripts directly (`python train.py`, `python evaluate.py`, `python predict.py`).

---

## Repository structure

```
.
├─ main.py                    # CLI entrypoint: train/evaluate/predict subcommands
├─ config.py                  # All paths + hyperparameters + distillation settings
├─ data_loader.py             # Dataset + transforms + DataLoaders
├─ model.py                   # Model factory + device selection + KD loss + checkpoint loader
├─ train.py                   # Training loop (+ distillation option) + checkpointing
├─ evaluate.py                # Test-set evaluation + confusion matrix + metrics report
├─ predict.py                 # Single-image / folder inference
├─ KNOWLEDGE_DISTILLATION.md  # Practical KD notes + recommended settings
├─ models/                    # Saved checkpoints (.pth)
└─ results/                   # Confusion matrix + metrics report + training history
```

---

## End-to-end flowcharts

### High-level CLI flow

```mermaid
flowchart TD
  A[User runs main.py] --> B{command}
  B -->|train| C[train.train()]
  B -->|evaluate| D[evaluate.evaluate(model_path)]
  B -->|predict| E[predict.main(args)]
```

### Training (standard vs knowledge distillation)

```mermaid
flowchart TD
  A[train.py: train()] --> B[get_device()]
  B --> C[get_dataloaders(): train/val/test]
  C --> D{USE_DISTILLATION?}

  D -->|No| E[get_model(MODEL_NAME, pretrained=True)]
  E --> F[criterion = CrossEntropyLoss]

  D -->|Yes| G[Teacher: get_model(TEACHER_MODEL, pretrained=True)]
  G --> H{TEACHER_MODEL_PATH set?}
  H -->|Yes| I[Load teacher checkpoint weights]
  H -->|No| J[Use ImageNet weights]
  I --> K[Student: get_model(STUDENT_MODEL, pretrained=True)]
  J --> K
  K --> L[distill_criterion = DistillationLoss(T, alpha)]
  L --> M[criterion = CrossEntropyLoss (validation)]

  F --> N[Epoch loop]
  M --> N
  N --> O[train_epoch(): forward + loss + backward + optimizer.step]
  N --> P[validate(): forward only]
  P --> Q[Early stopping + save best_model.pth]
  Q --> R[Save final_model.pth + training_history.csv]
```

### Evaluation / prediction checkpoint loading (important)

```mermaid
flowchart TD
  A[models/*.pth] --> B{Checkpoint format}
  B -->|Legacy: state_dict only| C[Use config default architecture]
  B -->|New: metadata checkpoint (state_dict + model_name)| D[Use stored model_name]
  C --> E[build model + load_state_dict]
  D --> E
  E --> F[Inference/eval loop]
```

---

## How each file works (code walkthrough)

### `config.py` — configuration & paths

Centralizes:

- **Dataset paths**: `TRAIN_DIR`, `VAL_DIR`, `TEST_DIR` under `breast_mri_dataset/`
- **Training hyperparameters**: `BATCH_SIZE`, `NUM_EPOCHS`, `LEARNING_RATE`, `WEIGHT_DECAY`
- **Model selection**:
  - Standard training uses `MODEL_NAME`
  - Distillation uses `TEACHER_MODEL` and `STUDENT_MODEL`
- **Distillation knobs**:
  - `DISTILLATION_TEMPERATURE` (softening)
  - `DISTILLATION_ALPHA` (soft vs hard loss weight)
- **Outputs**: `models/` and `results/` paths

### `data_loader.py` — dataset, transforms, and loaders

- **`BreastMRIDataset`**:
  - scans `<split>/Benign` and `<split>/Malignant`
  - returns `(image_tensor, label_int)`
  - uses PIL to open images and converts to RGB
- **Transforms**:
  - training: resize + rotation + horizontal flip + color jitter + ImageNet normalize
  - val/test: resize + normalize only
- **`get_dataloaders()`**:
  - prints whether split folders exist and how many samples were found
  - raises if train/val is empty (prevents silent training on zero data)

### `model.py` — model factory, KD loss, and checkpoint loading

- **`get_model(model_name, pretrained)`**:
  - wraps `torchvision.models.*`
  - replaces the classifier head for 2 classes (`config.NUM_CLASSES`)
- **`DistillationLoss`**:
  - hard loss: standard cross entropy
  - soft loss: KL divergence between teacher and student distributions
  - combined: \( \alpha \cdot L_{soft} + (1-\alpha)\cdot L_{hard} \)
- **`load_model_from_checkpoint()`** (used by `evaluate.py` and `predict.py`):
  - loads both legacy `state_dict` checkpoints and new metadata checkpoints
  - if the checkpoint contains `model_name`, it rebuilds that exact architecture automatically

### `train.py` — training loop and checkpointing

Key pieces:

- **Distillation mode** (`config.USE_DISTILLATION=True`):
  - teacher runs in `eval()` and is never updated
  - student is trained using combined distillation loss
- **Standard mode**:
  - single model trained with cross entropy
- **ReduceLROnPlateau** scheduler on validation loss
- **Early stopping** by `EARLY_STOPPING_PATIENCE` and `EARLY_STOPPING_MIN_DELTA`
- Saves:
  - `models/best_model.pth` (best validation loss)
  - `models/final_model.pth` (last model after training ends)
  - `results/training_history.csv`

### `evaluate.py` — test-set evaluation

- Loads model checkpoint (auto-detects architecture when available)
- Evaluates on `breast_mri_dataset/test`
- Computes:
  - accuracy, per-class precision/recall/F1, macro averages, ROC-AUC (when possible)
- Writes:
  - `results/confusion_matrix.png`
  - `results/metrics_report.txt`

### `predict.py` — inference

- Loads a checkpoint (auto-detects architecture when available)
- `--image`: prints predicted class + confidence (optionally full probabilities)
- `--folder`: prints a table with one row per image

### `main.py` — CLI orchestrator

`main.py` is a thin wrapper so you can do everything via:

```bash
python main.py train|evaluate|predict ...
```

---

## Knowledge distillation notes

See `KNOWLEDGE_DISTILLATION.md` for practical guidance and suggested settings.

Recommended workflow:

1) Train a strong teacher (optional, if you want your own teacher weights).
2) Enable distillation and train the student.
3) Deploy the student checkpoint for faster inference.

---

## Outputs you should expect

- **`models/best_model.pth`**: best validation checkpoint (now saved with architecture metadata)
- **`models/final_model.pth`**: final checkpoint
- **`results/training_history.csv`**: epoch-by-epoch loss/accuracy
- **`results/confusion_matrix.png`**, **`results/metrics_report.txt`**

---

## Troubleshooting

### “size mismatch / missing keys” when loading a checkpoint

This happens when the checkpoint architecture doesn’t match the model you’re constructing.

- New checkpoints saved by `train.py` include `model_name`, and `evaluate.py`/`predict.py` will rebuild the right model automatically.
- If you have an older checkpoint (saved as raw `state_dict`), make sure `config.MODEL_NAME` matches what was trained.

### Windows DataLoader issues

If you see worker-related hangs/crashes on Windows, edit `data_loader.py` and set `num_workers=0` in the `DataLoader(...)` constructors.

---

## License / disclaimer

This code is for research/education. It is **not** a medical device and must not be used for clinical decisions without appropriate validation and regulatory approval.
