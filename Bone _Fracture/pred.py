from ultralytics import YOLO
import os
import random
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
from pathlib import Path

# === Set seeds for reproducibility ===
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(RANDOM_SEED)
    torch.cuda.manual_seed_all(RANDOM_SEED)

# === Paths ===
VAL_IMAGE_DIR = "valid/images"
MODEL_PATH    = "models\BoneYolotest1.pt"  # or wherever your model is
OUTPUT_DIR    = "results/model_output"

# === Create output directory ===
os.makedirs(OUTPUT_DIR, exist_ok=True)

# === Device detection ===
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# === Load trained model ===
model = YOLO(MODEL_PATH) 

# === Collect ALL validation images ===
val_images = [os.path.join(VAL_IMAGE_DIR, f)
              for f in os.listdir(VAL_IMAGE_DIR)
              if f.lower().endswith((".png", ".jpg", ".jpeg"))]

print(f"Processing {len(val_images)} images for inference")

# === Run inference with explicit parameters ===
# Try lower confidence threshold if only few detections
# Standard YOLO defaults: conf=0.25, iou=0.45
# Lowering to conf=0.15 to match Kaggle results better
results = model(val_images, conf=0.15, iou=0.45, device=device, verbose=True)

# === Save only detected images ===
detected_count = 0
total_detections = 0

for i, result in enumerate(results):
    num_detections = len(result.boxes) if result.boxes is not None else 0
    total_detections += num_detections
    
    # Only save images with detections
    if num_detections > 0:
        detected_count += 1
        # Use YOLO's built-in plot method to get annotated image
        annotated_img = result.plot()  # Returns numpy array with annotations
        
        # Save image
        output_path = os.path.join(OUTPUT_DIR, os.path.basename(result.path))
        Image.fromarray(annotated_img).save(output_path)
        print(f"Saved: {os.path.basename(result.path)} ({num_detections} detection(s))")

print(f"\n{'='*60}")
print(f"Summary:")
print(f"  Total images processed: {len(val_images)}")
print(f"  Images with detections: {detected_count}")
print(f"  Total detections: {total_detections}")
print(f"  Output directory: {OUTPUT_DIR}")
print(f"{'='*60}")


# === Optional: Plotting detected images (if you want to visualize) ===
# Uncomment below to show plots of detected images
"""
detected_results = [r for r in results if r.boxes is not None and len(r.boxes) > 0]
if detected_results:
    n_images = len(detected_results)
    n_cols = 4
    n_rows = (n_images + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4*n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    axes = axes.flatten()

    for ax, result in zip(axes[:n_images], detected_results):
        img = result.orig_img  # NumPy image
        boxes = result.boxes

        ax.imshow(img, cmap="gray" if len(img.shape) == 2 or img.shape[2] == 1 else None)

        if boxes is not None and boxes.xyxy is not None:
            for box in boxes.xyxy.cpu().numpy():
                x1, y1, x2, y2 = box[:4]
                rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                         linewidth=2, edgecolor="lime", facecolor="none")
                ax.add_patch(rect)

        ax.set_title(os.path.basename(result.path), fontsize=10)
        ax.axis("off")
    
    # Hide unused subplots
    for ax in axes[n_images:]:
        ax.axis("off")

    plt.tight_layout()
    plt.show()
"""