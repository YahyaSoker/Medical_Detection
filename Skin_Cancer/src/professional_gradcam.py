"""
Professional GradCAM Heatmap Visualization for Medical AI
Enhanced visualization with medical-grade styling and professional appearance.
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Rectangle
import seaborn as sns
import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.preprocessing import LabelEncoder
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
import warnings
warnings.filterwarnings('ignore')

# Set professional styling
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

class ProfessionalGradCAM:
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

def create_medical_model():
    """Create a medical-grade model for demonstration"""
    model = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
    # Modify classifier for 7 classes
    model.classifier = nn.Sequential(
        nn.Dropout(0.2),
        nn.Linear(model.classifier[1].in_features, 7)
    )
    return model

def preprocess_image_medical(image_path, size=(224, 224)):
    """Preprocess image with medical-grade preprocessing"""
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image: {image_path}")
    
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    original_image = image_rgb.copy()
    
    # Resize for preprocessing
    image_resized = cv2.resize(image_rgb, size)
    
    # Medical-grade normalization
    transform = A.Compose([
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])
    
    transformed = transform(image=image_resized)
    input_tensor = transformed['image'].unsqueeze(0)
    
    return input_tensor, original_image, image_resized

def create_professional_heatmap(original_image, heatmap, prediction_info, save_path=None):
    """Create a professional medical-grade heatmap visualization"""
    
    # Medical color scheme
    medical_colors = {
        'background': '#f8f9fa',
        'text_primary': '#2c3e50',
        'text_secondary': '#7f8c8d',
        'accent': '#3498db',
        'success': '#27ae60',
        'warning': '#f39c12',
        'danger': '#e74c3c',
        'border': '#bdc3c7'
    }
    
    # Create figure with medical styling
    fig = plt.figure(figsize=(20, 12))
    fig.patch.set_facecolor(medical_colors['background'])
    
    # Create grid layout
    gs = fig.add_gridspec(3, 4, height_ratios=[1, 2, 1], width_ratios=[1, 1, 1, 1],
                         hspace=0.3, wspace=0.2)
    
    # Title
    title_ax = fig.add_subplot(gs[0, :])
    title_ax.text(0.5, 0.5, 'AI-Powered Skin Lesion Analysis with GradCAM Visualization', 
                 ha='center', va='center', fontsize=24, fontweight='bold', 
                 color=medical_colors['text_primary'])
    title_ax.text(0.5, 0.2, 'Medical-Grade Interpretable AI for Dermatological Diagnosis', 
                 ha='center', va='center', fontsize=14, 
                 color=medical_colors['text_secondary'])
    title_ax.set_xlim(0, 1)
    title_ax.set_ylim(0, 1)
    title_ax.axis('off')
    
    # Original image
    ax1 = fig.add_subplot(gs[1, 0])
    ax1.imshow(original_image)
    ax1.set_title('Original Dermoscopic Image', fontsize=16, fontweight='bold', 
                 color=medical_colors['text_primary'], pad=20)
    ax1.axis('off')
    
    # Add border
    for spine in ax1.spines.values():
        spine.set_edgecolor(medical_colors['border'])
        spine.set_linewidth(2)
    
    # Heatmap
    ax2 = fig.add_subplot(gs[1, 1])
    im = ax2.imshow(heatmap, cmap='jet', alpha=0.9)
    ax2.set_title('GradCAM Attention Map', fontsize=16, fontweight='bold', 
                 color=medical_colors['text_primary'], pad=20)
    ax2.axis('off')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax2, fraction=0.046, pad=0.04, shrink=0.8)
    cbar.set_label('Attention Intensity', fontsize=12, fontweight='bold')
    cbar.ax.tick_params(labelsize=10)
    
    # Add border
    for spine in ax2.spines.values():
        spine.set_edgecolor(medical_colors['border'])
        spine.set_linewidth(2)
    
    # Overlay
    ax3 = fig.add_subplot(gs[1, 2])
    ax3.imshow(original_image)
    overlay = ax3.imshow(heatmap, cmap='jet', alpha=0.5)
    ax3.set_title('Attention Overlay', fontsize=16, fontweight='bold', 
                 color=medical_colors['text_primary'], pad=20)
    ax3.axis('off')
    
    # Add border
    for spine in ax3.spines.values():
        spine.set_edgecolor(medical_colors['border'])
        spine.set_linewidth(2)
    
    # Enhanced overlay with contours
    ax4 = fig.add_subplot(gs[1, 3])
    ax4.imshow(original_image)
    
    # Create contour lines for better visualization
    contours = ax4.contour(heatmap, levels=8, colors='white', alpha=0.8, linewidths=1.5)
    ax4.imshow(heatmap, cmap='jet', alpha=0.4)
    ax4.set_title('Enhanced Overlay with Contours', fontsize=16, fontweight='bold', 
                 color=medical_colors['text_primary'], pad=20)
    ax4.axis('off')
    
    # Add border
    for spine in ax4.spines.values():
        spine.set_edgecolor(medical_colors['border'])
        spine.set_linewidth(2)
    
    # Prediction information panel
    info_ax = fig.add_subplot(gs[2, :2])
    info_ax.set_facecolor('#ecf0f1')
    info_ax.set_xlim(0, 1)
    info_ax.set_ylim(0, 1)
    info_ax.axis('off')
    
    # Prediction details
    class_name = prediction_info['class_name']
    confidence = prediction_info['confidence']
    is_malignant = prediction_info['is_malignant']
    
    # Confidence color
    conf_color = medical_colors['danger'] if is_malignant else medical_colors['success']
    
    info_ax.text(0.05, 0.8, 'AI DIAGNOSTIC RESULT', fontsize=18, fontweight='bold', 
                color=medical_colors['text_primary'])
    
    info_ax.text(0.05, 0.6, f'Predicted Lesion Type: {class_name}', fontsize=14, 
                color=medical_colors['text_primary'])
    
    info_ax.text(0.05, 0.5, f'Confidence Level: {confidence:.1%}', fontsize=14, 
                color=conf_color, fontweight='bold')
    
    info_ax.text(0.05, 0.4, f'Malignancy Risk: {"HIGH" if is_malignant else "LOW"}', 
                fontsize=14, color=conf_color, fontweight='bold')
    
    # Add confidence bar
    bar_width = confidence
    bar = Rectangle((0.05, 0.25), bar_width * 0.9, 0.1, 
                   facecolor=conf_color, alpha=0.7)
    info_ax.add_patch(bar)
    info_ax.text(0.5, 0.2, f'{confidence:.1%} Confidence', ha='center', va='center', 
                fontsize=12, fontweight='bold', color='white')
    
    # Medical disclaimer
    disclaimer_ax = fig.add_subplot(gs[2, 2:])
    disclaimer_ax.set_facecolor('#f8f9fa')
    disclaimer_ax.set_xlim(0, 1)
    disclaimer_ax.set_ylim(0, 1)
    disclaimer_ax.axis('off')
    
    disclaimer_ax.text(0.05, 0.8, 'MEDICAL DISCLAIMER', fontsize=14, fontweight='bold', 
                      color=medical_colors['text_primary'])
    
    disclaimer_text = """This AI analysis is for research and educational purposes only.
It should not be used as a substitute for professional medical diagnosis.
Always consult a qualified dermatologist for medical decisions.
The heatmap shows areas of high attention but does not guarantee clinical accuracy."""
    
    disclaimer_ax.text(0.05, 0.6, disclaimer_text, fontsize=10, 
                      color=medical_colors['text_secondary'], va='top')
    
    # Add logo/watermark area
    logo_ax = fig.add_subplot(gs[2, 3])
    logo_ax.text(0.5, 0.5, 'AI MED', ha='center', va='center', 
                fontsize=16, fontweight='bold', color=medical_colors['accent'],
                bbox=dict(boxstyle="round,pad=0.3", facecolor='white', edgecolor=medical_colors['accent']))
    logo_ax.axis('off')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', 
                   facecolor=medical_colors['background'])
    
    plt.show()

def create_attention_analysis(original_image, heatmap, prediction_info, save_path=None):
    """Create detailed attention analysis visualization"""
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.patch.set_facecolor('#f8f9fa')
    
    # Medical color scheme
    colors = ['#e74c3c', '#f39c12', '#f1c40f', '#2ecc71', '#3498db', '#9b59b6']
    
    # Original image
    axes[0, 0].imshow(original_image)
    axes[0, 0].set_title('Original Image', fontsize=14, fontweight='bold')
    axes[0, 0].axis('off')
    
    # Heatmap with different colormaps
    colormaps = ['jet', 'hot', 'viridis']
    titles = ['Jet (Medical Standard)', 'Hot (High Contrast)', 'Viridis (Accessible)']
    
    for i, (cmap, title) in enumerate(zip(colormaps, titles)):
        im = axes[0, i+1].imshow(heatmap, cmap=cmap, alpha=0.9)
        axes[0, i+1].set_title(title, fontsize=14, fontweight='bold')
        axes[0, i+1].axis('off')
        plt.colorbar(im, ax=axes[0, i+1], fraction=0.046, pad=0.04)
    
    # Attention regions analysis
    # Threshold the heatmap to find high attention regions
    threshold = 0.7
    high_attention = heatmap > threshold
    
    axes[1, 0].imshow(original_image)
    axes[1, 0].imshow(high_attention, cmap='Reds', alpha=0.6)
    axes[1, 0].set_title(f'High Attention Regions (>{threshold:.0%})', fontsize=14, fontweight='bold')
    axes[1, 0].axis('off')
    
    # Attention intensity histogram
    axes[1, 1].hist(heatmap.flatten(), bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    axes[1, 1].axvline(threshold, color='red', linestyle='--', linewidth=2, label=f'Threshold ({threshold:.0%})')
    axes[1, 1].set_title('Attention Intensity Distribution', fontsize=14, fontweight='bold')
    axes[1, 1].set_xlabel('Attention Value')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    # Prediction confidence visualization
    confidence = prediction_info['confidence']
    is_malignant = prediction_info['is_malignant']
    
    # Create confidence gauge
    theta = np.linspace(0, np.pi, 100)
    r = np.ones_like(theta)
    
    axes[1, 2].plot(theta, r, 'k-', linewidth=2)
    axes[1, 2].fill_between(theta, 0, r, alpha=0.3, color='lightgray')
    
    # Confidence arc
    conf_theta = np.linspace(0, np.pi * confidence, 50)
    conf_r = np.ones_like(conf_theta)
    conf_color = '#e74c3c' if is_malignant else '#27ae60'
    
    axes[1, 2].plot(conf_theta, conf_r, color=conf_color, linewidth=4)
    axes[1, 2].fill_between(conf_theta, 0, conf_r, alpha=0.6, color=conf_color)
    
    axes[1, 2].set_title(f'Confidence: {confidence:.1%}', fontsize=14, fontweight='bold')
    axes[1, 2].set_xlim(0, np.pi)
    axes[1, 2].set_ylim(0, 1.2)
    axes[1, 2].axis('off')
    
    # Add confidence text
    axes[1, 2].text(np.pi/2, 0.5, f'{confidence:.1%}', ha='center', va='center', 
                   fontsize=20, fontweight='bold', color=conf_color)
    
    plt.suptitle('Detailed Attention Analysis for Medical AI', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()

def create_medical_report(original_image, heatmap, prediction_info, save_path=None):
    """Create a comprehensive medical report visualization"""
    
    fig = plt.figure(figsize=(16, 20))
    fig.patch.set_facecolor('#ffffff')
    
    # Create grid
    gs = fig.add_gridspec(4, 3, height_ratios=[0.5, 2, 1, 1], width_ratios=[1, 1, 1],
                         hspace=0.3, wspace=0.2)
    
    # Header
    header_ax = fig.add_subplot(gs[0, :])
    header_ax.text(0.5, 0.7, 'DERMATOLOGICAL AI ANALYSIS REPORT', 
                  ha='center', va='center', fontsize=20, fontweight='bold', 
                  color='#2c3e50')
    header_ax.text(0.5, 0.3, f'Generated: {pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")}', 
                  ha='center', va='center', fontsize=12, color='#7f8c8d')
    header_ax.set_xlim(0, 1)
    header_ax.set_ylim(0, 1)
    header_ax.axis('off')
    
    # Main visualization
    main_ax = fig.add_subplot(gs[1, :])
    
    # Create subplot for main image
    sub_gs = main_ax.get_subplotspec().subgridspec(1, 3, wspace=0.1)
    
    # Original image
    ax1 = fig.add_subplot(sub_gs[0])
    ax1.imshow(original_image)
    ax1.set_title('Dermoscopic Image', fontsize=14, fontweight='bold')
    ax1.axis('off')
    
    # Heatmap
    ax2 = fig.add_subplot(sub_gs[1])
    im = ax2.imshow(heatmap, cmap='jet', alpha=0.9)
    ax2.set_title('AI Attention Map', fontsize=14, fontweight='bold')
    ax2.axis('off')
    plt.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
    
    # Overlay
    ax3 = fig.add_subplot(sub_gs[2])
    ax3.imshow(original_image)
    ax3.imshow(heatmap, cmap='jet', alpha=0.5)
    ax3.set_title('Attention Overlay', fontsize=14, fontweight='bold')
    ax3.axis('off')
    
    # Diagnosis section
    diag_ax = fig.add_subplot(gs[2, :])
    diag_ax.set_facecolor('#ecf0f1')
    diag_ax.set_xlim(0, 1)
    diag_ax.set_ylim(0, 1)
    diag_ax.axis('off')
    
    class_name = prediction_info['class_name']
    confidence = prediction_info['confidence']
    is_malignant = prediction_info['is_malignant']
    
    diag_ax.text(0.05, 0.8, 'AI DIAGNOSIS', fontsize=16, fontweight='bold', color='#2c3e50')
    
    # Diagnosis box
    diag_box = Rectangle((0.05, 0.4), 0.9, 0.35, linewidth=2, 
                        edgecolor='#3498db', facecolor='white', alpha=0.8)
    diag_ax.add_patch(diag_box)
    
    diag_ax.text(0.1, 0.65, f'Predicted Lesion Type: {class_name}', 
                fontsize=14, fontweight='bold', color='#2c3e50')
    diag_ax.text(0.1, 0.55, f'Confidence Level: {confidence:.1%}', 
                fontsize=12, color='#7f8c8d')
    diag_ax.text(0.1, 0.45, f'Malignancy Risk: {"HIGH" if is_malignant else "LOW"}', 
                fontsize=12, color='#e74c3c' if is_malignant else '#27ae60', fontweight='bold')
    
    # Footer with disclaimers
    footer_ax = fig.add_subplot(gs[3, :])
    footer_ax.set_facecolor('#f8f9fa')
    footer_ax.set_xlim(0, 1)
    footer_ax.set_ylim(0, 1)
    footer_ax.axis('off')
    
    disclaimer_text = """IMPORTANT MEDICAL DISCLAIMER:
This AI analysis is for research and educational purposes only. It should not be used as a substitute 
for professional medical diagnosis, treatment, or advice. Always consult a qualified dermatologist 
or healthcare professional for medical decisions. The attention map shows areas of high AI focus but 
does not guarantee clinical accuracy or replace human medical judgment."""
    
    footer_ax.text(0.05, 0.7, disclaimer_text, fontsize=10, color='#7f8c8d', 
                  va='top', ha='left')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()

def main():
    """Main function for professional GradCAM visualization"""
    print("=== Professional Medical GradCAM Visualization ===")
    
    # Medical class information
    class_info = {
        'nv': {'name': 'Melanocytic Nevus', 'malignant': False, 'description': 'Benign mole'},
        'mel': {'name': 'Melanoma', 'malignant': True, 'description': 'Malignant skin cancer'},
        'bkl': {'name': 'Benign Keratosis', 'malignant': False, 'description': 'Benign skin growth'},
        'bcc': {'name': 'Basal Cell Carcinoma', 'malignant': True, 'description': 'Most common skin cancer'},
        'akiec': {'name': 'Actinic Keratosis', 'malignant': False, 'description': 'Precancerous lesion'},
        'vasc': {'name': 'Vascular Lesion', 'malignant': False, 'description': 'Blood vessel abnormality'},
        'df': {'name': 'Dermatofibroma', 'malignant': False, 'description': 'Benign skin tumor'}
    }
    
    # Create model
    print("Initializing medical AI model...")
    model = create_medical_model()
    model.eval()
    
    # Setup GradCAM
    target_layer = model.features[-1]
    gradcam = ProfessionalGradCAM(model, target_layer)
    
    # Find sample images
    sample_images = []
    for part in ['HAM10000_images_part_1', 'HAM10000_images_part_2']:
        if os.path.exists(part):
            images = [f for f in os.listdir(part) if f.endswith('.jpg')][:2]
            sample_images.extend([os.path.join(part, img) for img in images])
        if len(sample_images) >= 2:
            break
    
    if not sample_images:
        print("No sample images found!")
        return
    
    # Process images
    for i, image_path in enumerate(sample_images):
        print(f"\nProcessing image {i+1}: {os.path.basename(image_path)}")
        
        try:
            # Preprocess
            input_tensor, original_image, processed_image = preprocess_image_medical(image_path)
            
            # Make prediction
            with torch.no_grad():
                output = model(input_tensor)
                probabilities = torch.softmax(output, dim=1)
                predicted_class_idx = torch.argmax(output, dim=1).item()
                confidence = probabilities[0, predicted_class_idx].item()
            
            # Get class information
            class_names = list(class_info.keys())
            predicted_class = class_names[predicted_class_idx]
            class_data = class_info[predicted_class]
            
            prediction_info = {
                'class_name': class_data['name'],
                'confidence': confidence,
                'is_malignant': class_data['malignant'],
                'description': class_data['description']
            }
            
            # Generate heatmap
            heatmap = gradcam.generate_cam(input_tensor, predicted_class_idx)
            heatmap_resized = cv2.resize(heatmap, (original_image.shape[1], original_image.shape[0]))
            
            # Create professional visualizations
            print("Creating professional medical visualization...")
            create_professional_heatmap(
                original_image, heatmap_resized, prediction_info,
                f"professional_medical_analysis_{i+1}.png"
            )
            
            print("Creating detailed attention analysis...")
            create_attention_analysis(
                original_image, heatmap_resized, prediction_info,
                f"attention_analysis_{i+1}.png"
            )
            
            print("Creating medical report...")
            create_medical_report(
                original_image, heatmap_resized, prediction_info,
                f"medical_report_{i+1}.png"
            )
            
            print(f"✓ Prediction: {class_data['name']} ({confidence:.1%} confidence)")
            print(f"✓ Malignancy Risk: {'HIGH' if class_data['malignant'] else 'LOW'}")
            
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
    
    print("\n=== Professional Medical Analysis Complete ===")
    print("Generated files:")
    print("- professional_medical_analysis_*.png - Main medical visualization")
    print("- attention_analysis_*.png - Detailed attention analysis")
    print("- medical_report_*.png - Comprehensive medical report")

if __name__ == "__main__":
    main()

