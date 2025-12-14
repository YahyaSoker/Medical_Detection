import os
import matplotlib.pyplot as plt
import cv2

# Paths
image_dir = "BoneFractureYolo8/train/images"
label_dir = "BoneFractureYolo8/train/labels"

# Get first 10 image filenames
files = sorted(os.listdir(image_dir))[:10]

# Plot images
plt.figure(figsize=(15, 10))
for i, file in enumerate(files):
    img_path = os.path.join(image_dir, file)
    label_path = os.path.join(label_dir, file.replace(".jpg", ".txt").replace(".png", ".txt"))
    
    # Load image
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w, _ = img.shape
    
    # Draw YOLO bounding boxes if label exists
    if os.path.exists(label_path):
        with open(label_path, "r") as f:
            for line in f.readlines():
                cls, x_center, y_center, bw, bh = map(float, line.strip().split())
                
                # Convert YOLO to pixel coordinates
                x1 = int((x_center - bw/2) * w)
                y1 = int((y_center - bh/2) * h)
                x2 = int((x_center + bw/2) * w)
                y2 = int((y_center + bh/2) * h)
                
                # Draw rectangle & label
                cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(img, str(int(cls)), (x1, y1-5), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
    
    # Show subplot
    plt.subplot(2, 5, i+1)
    plt.imshow(img)
    plt.title(file, fontsize=8)
    plt.axis("off")

plt.tight_layout()
plt.show()