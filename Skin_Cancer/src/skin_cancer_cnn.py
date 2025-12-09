"""
CNN-based Skin Cancer Detection with GradCAM Heatmap Visualization
This is the recommended approach for skin cancer detection as it provides
interpretable heatmaps showing which parts of the image the model focuses on.
"""

import os
import sys
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Get project root directory (parent of src/)
PROJECT_ROOT = Path(__file__).parent.parent

class SkinCancerDataset(Dataset):
    def __init__(self, df, image_dir, transform=None, is_training=True):
        self.df = df.reset_index(drop=True)
        self.image_dir = image_dir
        self.transform = transform
        self.is_training = is_training
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_id = row['image_id']
        label = row['label_encoded']
        
        # Find image file
        image_path = None
        for part in ['HAM10000_images_part_1', 'HAM10000_images_part_2']:
            potential_path = self.image_dir / part / f"{image_id}.jpg"
            if potential_path.exists():
                image_path = str(potential_path)
                break
        
        if image_path is None:
            # Return a dummy image if file not found
            image = np.zeros((224, 224, 3), dtype=np.uint8)
        else:
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        if self.transform:
            if isinstance(self.transform, A.Compose):
                image = self.transform(image=image)['image']
            else:
                image = self.transform(image)
        
        return image, label

class SkinCancerCNN(nn.Module):
    def __init__(self, num_classes=7, pretrained=True):
        super(SkinCancerCNN, self).__init__()
        
        if pretrained:
            self.backbone = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
        else:
            self.backbone = efficientnet_b0(weights=None)
        
        # Replace classifier
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.backbone.classifier[1].in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes)
        )
        
    def forward(self, x):
        return self.backbone(x)
    
    def get_features(self, x):
        """Extract features before classifier for GradCAM"""
        x = self.backbone.features(x)
        x = self.backbone.avgpool(x)
        x = torch.flatten(x, 1)
        return x

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Register hooks
        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_backward_hook(self.save_gradient)
    
    def save_activation(self, module, input, output):
        self.activations = output
    
    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]
    
    def generate_cam(self, input_tensor, class_idx=None):
        # Forward pass
        output = self.model(input_tensor)
        
        if class_idx is None:
            class_idx = output.argmax(dim=1)
        
        # Backward pass
        self.model.zero_grad()
        output[0, class_idx].backward(retain_graph=True)
        
        # Generate CAM
        gradients = self.gradients[0]  # [C, H, W]
        activations = self.activations[0]  # [C, H, W]
        
        # Global average pooling of gradients
        weights = gradients.mean(dim=(1, 2))  # [C]
        
        # Weighted combination of activation maps
        cam = torch.zeros(activations.shape[1:], dtype=torch.float32)
        for i, w in enumerate(weights):
            cam += w * activations[i, :, :]
        
        # Apply ReLU
        cam = torch.relu(cam)
        
        # Normalize
        cam = cam - cam.min()
        cam = cam / cam.max()
        
        return cam.detach().cpu().numpy()

class SkinCancerDetector:
    def __init__(self, data_path=None, image_dir=None, device=None):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.data_path = Path(data_path) if data_path else PROJECT_ROOT / 'data'
        self.image_dir = Path(image_dir) if image_dir else PROJECT_ROOT / 'data'
        self.classes = ['nv', 'mel', 'bkl', 'bcc', 'akiec', 'vasc', 'df']
        self.class_names = {
            'nv': 'Melanocytic nevus',
            'mel': 'Melanoma',
            'bkl': 'Benign keratosis',
            'bcc': 'Basal cell carcinoma',
            'akiec': 'Actinic keratosis',
            'vasc': 'Vascular lesion',
            'df': 'Dermatofibroma'
        }
        self.label_encoder = LabelEncoder()
        self.model = None
        self.gradcam = None
        
    def prepare_data(self, test_size=0.2, val_size=0.1):
        """Prepare and split the dataset"""
        print("Loading and preparing data...")
        
        # Load metadata
        df = pd.read_csv(self.data_path / "HAM10000_metadata.csv")
        
        # Encode labels
        df['label_encoded'] = self.label_encoder.fit_transform(df['dx'])
        
        # Split data
        train_df, temp_df = train_test_split(
            df, test_size=test_size + val_size, random_state=42, stratify=df['dx']
        )
        val_df, test_df = train_test_split(
            temp_df, test_size=test_size/(test_size + val_size), random_state=42, stratify=temp_df['dx']
        )
        
        print(f"Training samples: {len(train_df)}")
        print(f"Validation samples: {len(val_df)}")
        print(f"Test samples: {len(test_df)}")
        
        return train_df, val_df, test_df
    
    def get_transforms(self):
        """Get data augmentation transforms"""
        train_transform = A.Compose([
            A.Resize(224, 224),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.RandomBrightnessContrast(p=0.3),
            A.HueSaturationValue(p=0.3),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
        
        val_transform = A.Compose([
            A.Resize(224, 224),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
        
        return train_transform, val_transform
    
    def create_dataloaders(self, train_df, val_df, test_df, batch_size=32):
        """Create data loaders"""
        train_transform, val_transform = self.get_transforms()
        
        train_dataset = SkinCancerDataset(train_df, self.data_path, train_transform, True)
        val_dataset = SkinCancerDataset(val_df, self.data_path, val_transform, False)
        test_dataset = SkinCancerDataset(test_df, self.data_path, val_transform, False)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
        
        return train_loader, val_loader, test_loader
    
    def train_model(self, train_loader, val_loader, epochs=20, learning_rate=0.001):
        """Train the CNN model"""
        print("Initializing model...")
        self.model = SkinCancerCNN(num_classes=len(self.classes)).to(self.device)
        
        # Setup GradCAM
        target_layer = self.model.backbone.features[-1]
        self.gradcam = GradCAM(self.model, target_layer)
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5)
        
        best_val_acc = 0
        train_losses = []
        val_losses = []
        val_accuracies = []
        
        print("Starting training...")
        for epoch in range(epochs):
            # Training phase
            self.model.train()
            train_loss = 0
            train_correct = 0
            train_total = 0
            
            for images, labels in tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}'):
                images, labels = images.to(self.device), labels.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                train_total += labels.size(0)
                train_correct += (predicted == labels).sum().item()
            
            # Validation phase
            self.model.eval()
            val_loss = 0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for images, labels in val_loader:
                    images, labels = images.to(self.device), labels.to(self.device)
                    outputs = self.model(images)
                    loss = criterion(outputs, labels)
                    
                    val_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    val_total += labels.size(0)
                    val_correct += (predicted == labels).sum().item()
            
            # Calculate metrics
            train_loss /= len(train_loader)
            val_loss /= len(val_loader)
            train_acc = 100 * train_correct / train_total
            val_acc = 100 * val_correct / val_total
            
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            val_accuracies.append(val_acc)
            
            scheduler.step(val_loss)
            
            print(f'Epoch {epoch+1}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, '
                  f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                model_path = PROJECT_ROOT / 'models' / 'best_skin_cancer_model.pth'
                model_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(self.model.state_dict(), str(model_path))
        
        # Plot training history
        self.plot_training_history(train_losses, val_losses, val_accuracies)
        
        return train_losses, val_losses, val_accuracies
    
    def plot_training_history(self, train_losses, val_losses, val_accuracies):
        """Plot training history"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Loss plot
        ax1.plot(train_losses, label='Training Loss')
        ax1.plot(val_losses, label='Validation Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Accuracy plot
        ax2.plot(val_accuracies, label='Validation Accuracy')
        ax2.set_title('Validation Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        output_path = PROJECT_ROOT / 'outputs' / 'training_history.png'
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(str(output_path), dpi=300, bbox_inches='tight')
        plt.show()
    
    def evaluate_model(self, test_loader):
        """Evaluate the model on test set"""
        if self.model is None:
            print("Model not trained yet!")
            return
        
        self.model.eval()
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for images, labels in tqdm(test_loader, desc='Evaluating'):
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                _, predicted = torch.max(outputs, 1)
                
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # Classification report
        print("\nClassification Report:")
        print(classification_report(all_labels, all_predictions, target_names=self.classes))
        
        # Confusion matrix
        cm = confusion_matrix(all_labels, all_predictions)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=self.classes, yticklabels=self.classes)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.tight_layout()
        output_path = PROJECT_ROOT / 'outputs' / 'confusion_matrix.png'
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(str(output_path), dpi=300, bbox_inches='tight')
        plt.show()
        
        return all_predictions, all_labels
    
    def predict_with_heatmap(self, image_path, save_path=None):
        """Make prediction with GradCAM heatmap visualization"""
        if self.model is None:
            print("Model not trained yet!")
            return
        
        # Load and preprocess image
        image = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        original_image = image_rgb.copy()
        
        # Preprocess for model
        transform = A.Compose([
            A.Resize(224, 224),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
        
        transformed = transform(image=image_rgb)
        input_tensor = transformed['image'].unsqueeze(0).to(self.device)
        
        # Make prediction
        self.model.eval()
        with torch.no_grad():
            output = self.model(input_tensor)
            probabilities = torch.softmax(output, dim=1)
            predicted_class = torch.argmax(output, dim=1).item()
            confidence = probabilities[0, predicted_class].item()
        
        # Generate GradCAM
        cam = self.gradcam.generate_cam(input_tensor, predicted_class)
        
        # Resize CAM to original image size
        cam_resized = cv2.resize(cam, (original_image.shape[1], original_image.shape[0]))
        
        # Create visualization
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # Original image
        axes[0].imshow(original_image)
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        # Heatmap
        im = axes[1].imshow(cam_resized, cmap='jet', alpha=0.8)
        axes[1].set_title('GradCAM Heatmap')
        axes[1].axis('off')
        plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
        
        # Overlay
        axes[2].imshow(original_image)
        axes[2].imshow(cam_resized, cmap='jet', alpha=0.4)
        axes[2].set_title('Overlay')
        axes[2].axis('off')
        
        # Add prediction text
        predicted_class_name = self.classes[predicted_class]
        class_description = self.class_names[predicted_class_name]
        
        plt.suptitle(f'Prediction: {class_description}\nConfidence: {confidence:.2%}', 
                    fontsize=14, y=0.95)
        
        plt.tight_layout()
        if save_path:
            output_path = PROJECT_ROOT / 'outputs' / Path(save_path).name
            output_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(str(output_path), dpi=300, bbox_inches='tight')
        plt.show()
        
        return {
            'predicted_class': predicted_class_name,
            'class_description': class_description,
            'confidence': confidence,
            'heatmap': cam_resized
        }

def main():
    """Main function to demonstrate CNN approach with GradCAM"""
    print("=== CNN-based Skin Cancer Detection with GradCAM ===")
    
    # Initialize detector
    detector = SkinCancerDetector()
    
    # Prepare data
    train_df, val_df, test_df = detector.prepare_data()
    
    # Create data loaders
    train_loader, val_loader, test_loader = detector.create_dataloaders(train_df, val_df, test_df)
    
    # Train model (uncomment to train)
    # print("Training model...")
    # detector.train_model(train_loader, val_loader, epochs=20)
    
    # Load pre-trained model if available
    model_path = PROJECT_ROOT / 'models' / 'best_skin_cancer_model.pth'
    if model_path.exists():
        print("Loading pre-trained model...")
        detector.model = SkinCancerCNN(num_classes=len(detector.classes)).to(detector.device)
        detector.model.load_state_dict(torch.load(str(model_path)))
        target_layer = detector.model.backbone.features[-1]
        detector.gradcam = GradCAM(detector.model, target_layer)
    
    # Example prediction with heatmap
    sample_image = PROJECT_ROOT / 'data' / 'HAM10000_images_part_1' / 'ISIC_0024306.jpg'
    if sample_image.exists():
        print("Making prediction with heatmap...")
        result = detector.predict_with_heatmap(str(sample_image), "cnn_prediction_heatmap.png")
        print(f"Prediction: {result['class_description']}")
        print(f"Confidence: {result['confidence']:.2%}")
    
    print("CNN setup complete. This approach provides interpretable heatmaps!")

if __name__ == "__main__":
    main()
