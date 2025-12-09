"""
Comprehensive Skin Cancer Detection Visualization Tool
This tool provides various visualization methods for skin cancer detection results.
"""

import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
import torch
import torch.nn.functional as F
from PIL import Image
import warnings
warnings.filterwarnings('ignore')

class SkinCancerVisualizer:
    def __init__(self, data_path="."):
        self.data_path = data_path
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
        self.colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD', '#98D8C8']
        
    def load_dataset_info(self):
        """Load and analyze dataset information"""
        df = pd.read_csv(os.path.join(self.data_path, "HAM10000_metadata.csv"))
        
        # Add class descriptions
        df['class_description'] = df['dx'].map(self.class_names)
        
        # Categorize as malignant/benign
        malignant_classes = ['mel', 'bcc']  # Melanoma and Basal cell carcinoma
        df['is_malignant'] = df['dx'].isin(malignant_classes)
        
        return df
    
    def plot_dataset_distribution(self, save_path=None):
        """Plot dataset distribution and characteristics"""
        df = self.load_dataset_info()
        
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        
        # Class distribution
        class_counts = df['dx'].value_counts()
        axes[0, 0].pie(class_counts.values, labels=class_counts.index, autopct='%1.1f%%', 
                      colors=self.colors[:len(class_counts)])
        axes[0, 0].set_title('Distribution of Skin Lesion Types')
        
        # Malignant vs Benign
        malignant_counts = df['is_malignant'].value_counts()
        axes[0, 1].pie(malignant_counts.values, labels=['Benign', 'Malignant'], 
                      autopct='%1.1f%%', colors=['#90EE90', '#FF6B6B'])
        axes[0, 1].set_title('Malignant vs Benign Distribution')
        
        # Age distribution
        axes[0, 2].hist(df['age'].dropna(), bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        axes[0, 2].set_title('Age Distribution')
        axes[0, 2].set_xlabel('Age')
        axes[0, 2].set_ylabel('Frequency')
        
        # Gender distribution
        gender_counts = df['sex'].value_counts()
        axes[1, 0].bar(gender_counts.index, gender_counts.values, color=['#FFB6C1', '#87CEEB'])
        axes[1, 0].set_title('Gender Distribution')
        axes[1, 0].set_ylabel('Count')
        
        # Localization distribution
        loc_counts = df['localization'].value_counts().head(10)
        axes[1, 1].barh(range(len(loc_counts)), loc_counts.values, color='lightcoral')
        axes[1, 1].set_yticks(range(len(loc_counts)))
        axes[1, 1].set_yticklabels(loc_counts.index)
        axes[1, 1].set_title('Top 10 Body Localizations')
        axes[1, 1].set_xlabel('Count')
        
        # Age vs Malignancy
        df_clean = df.dropna(subset=['age'])
        for i, (malignant, group) in enumerate(df_clean.groupby('is_malignant')):
            axes[1, 2].hist(group['age'], alpha=0.6, label=['Benign', 'Malignant'][i], 
                           bins=20, color=['#90EE90', '#FF6B6B'][i])
        axes[1, 2].set_title('Age Distribution by Malignancy')
        axes[1, 2].set_xlabel('Age')
        axes[1, 2].set_ylabel('Frequency')
        axes[1, 2].legend()
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        return df
    
    def plot_sample_images(self, samples_per_class=3, save_path=None):
        """Plot sample images for each class"""
        df = self.load_dataset_info()
        
        fig, axes = plt.subplots(len(self.classes), samples_per_class, figsize=(15, 20))
        if len(self.classes) == 1:
            axes = axes.reshape(1, -1)
        
        for i, class_name in enumerate(self.classes):
            class_df = df[df['dx'] == class_name].head(samples_per_class)
            
            for j, (_, row) in enumerate(class_df.iterrows()):
                image_id = row['image_id']
                
                # Find image file
                image_path = None
                for part in ['HAM10000_images_part_1', 'HAM10000_images_part_2']:
                    potential_path = os.path.join(self.data_path, part, f"{image_id}.jpg")
                    if os.path.exists(potential_path):
                        image_path = potential_path
                        break
                
                if image_path:
                    image = cv2.imread(image_path)
                    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    
                    axes[i, j].imshow(image_rgb)
                    axes[i, j].set_title(f'{self.class_names[class_name]}\nAge: {row["age"]}, Sex: {row["sex"]}')
                    axes[i, j].axis('off')
                else:
                    axes[i, j].text(0.5, 0.5, 'Image not found', ha='center', va='center')
                    axes[i, j].axis('off')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_model_comparison(self, results_dict, save_path=None):
        """Plot comparison between different models"""
        models = list(results_dict.keys())
        metrics = ['accuracy', 'precision', 'recall', 'f1_score']
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.flatten()
        
        for i, metric in enumerate(metrics):
            values = [results_dict[model].get(metric, 0) for model in models]
            bars = axes[i].bar(models, values, color=self.colors[:len(models)])
            axes[i].set_title(f'{metric.replace("_", " ").title()}')
            axes[i].set_ylabel('Score')
            axes[i].set_ylim(0, 1)
            
            # Add value labels on bars
            for bar, value in zip(bars, values):
                axes[i].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                           f'{value:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_roc_curves(self, y_true, y_scores, save_path=None):
        """Plot ROC curves for multi-class classification"""
        # Binarize the output
        y_true_bin = label_binarize(y_true, classes=range(len(self.classes)))
        
        # Compute ROC curve and ROC area for each class
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        
        for i in range(len(self.classes)):
            fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_scores[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
        
        # Plot all ROC curves
        plt.figure(figsize=(12, 8))
        
        for i, color in zip(range(len(self.classes)), self.colors):
            plt.plot(fpr[i], tpr[i], color=color, lw=2,
                    label=f'{self.class_names[self.classes[i]]} (AUC = {roc_auc[i]:.2f})')
        
        plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Random')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves for Skin Cancer Classification')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        return roc_auc
    
    def create_interactive_dashboard(self, df):
        """Create an interactive dashboard using Plotly"""
        # Age distribution by class
        fig1 = px.box(df, x='dx', y='age', color='dx',
                     title='Age Distribution by Skin Lesion Type',
                     labels={'dx': 'Lesion Type', 'age': 'Age'})
        
        # Gender distribution by class
        gender_class = df.groupby(['dx', 'sex']).size().reset_index(name='count')
        fig2 = px.bar(gender_class, x='dx', y='count', color='sex',
                     title='Gender Distribution by Skin Lesion Type',
                     labels={'dx': 'Lesion Type', 'count': 'Count'})
        
        # Localization heatmap
        loc_class = df.groupby(['dx', 'localization']).size().unstack(fill_value=0)
        fig3 = px.imshow(loc_class, aspect="auto", title='Lesion Type vs Body Localization Heatmap')
        
        # Malignancy analysis
        malignant_analysis = df.groupby(['dx', 'is_malignant']).size().reset_index(name='count')
        fig4 = px.bar(malignant_analysis, x='dx', y='count', color='is_malignant',
                     title='Malignant vs Benign by Lesion Type',
                     labels={'dx': 'Lesion Type', 'count': 'Count'})
        
        return [fig1, fig2, fig3, fig4]
    
    def plot_attention_maps(self, image_path, attention_maps, model_name="Model", save_path=None):
        """Plot attention maps for different models"""
        # Load original image
        image = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        n_maps = len(attention_maps)
        fig, axes = plt.subplots(2, n_maps + 1, figsize=(5 * (n_maps + 1), 10))
        
        # Original image
        axes[0, 0].imshow(image_rgb)
        axes[0, 0].set_title('Original Image')
        axes[0, 0].axis('off')
        
        axes[1, 0].imshow(image_rgb)
        axes[1, 0].set_title('Original Image')
        axes[1, 0].axis('off')
        
        # Attention maps
        for i, (map_name, attention_map) in enumerate(attention_maps.items()):
            # Resize attention map to match image
            attention_resized = cv2.resize(attention_map, (image_rgb.shape[1], image_rgb.shape[0]))
            
            # Heatmap
            im = axes[0, i + 1].imshow(attention_resized, cmap='jet', alpha=0.8)
            axes[0, i + 1].set_title(f'{map_name} Heatmap')
            axes[0, i + 1].axis('off')
            plt.colorbar(im, ax=axes[0, i + 1], fraction=0.046, pad=0.04)
            
            # Overlay
            axes[1, i + 1].imshow(image_rgb)
            axes[1, i + 1].imshow(attention_resized, cmap='jet', alpha=0.4)
            axes[1, i + 1].set_title(f'{map_name} Overlay')
            axes[1, i + 1].axis('off')
        
        plt.suptitle(f'Attention Maps Comparison - {model_name}', fontsize=16)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def generate_report(self, model_results, save_path="skin_cancer_analysis_report.html"):
        """Generate a comprehensive HTML report"""
        df = self.load_dataset_info()
        
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Skin Cancer Detection Analysis Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
                .section {{ margin: 30px 0; }}
                .metric {{ display: inline-block; margin: 10px; padding: 15px; background-color: #e8f4f8; border-radius: 5px; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Skin Cancer Detection Analysis Report</h1>
                <p>Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            </div>
            
            <div class="section">
                <h2>Dataset Overview</h2>
                <p><strong>Total Images:</strong> {len(df):,}</p>
                <p><strong>Classes:</strong> {len(self.classes)}</p>
                <p><strong>Malignant Cases:</strong> {df['is_malignant'].sum():,} ({df['is_malignant'].mean()*100:.1f}%)</p>
            </div>
            
            <div class="section">
                <h2>Class Distribution</h2>
                <table>
                    <tr><th>Class</th><th>Description</th><th>Count</th><th>Percentage</th><th>Malignant</th></tr>
        """
        
        class_counts = df['dx'].value_counts()
        malignant_classes = ['mel', 'bcc']
        
        for class_name in self.classes:
            count = class_counts.get(class_name, 0)
            percentage = (count / len(df)) * 100
            is_malignant = "Yes" if class_name in malignant_classes else "No"
            
            html_content += f"""
                    <tr>
                        <td>{class_name}</td>
                        <td>{self.class_names[class_name]}</td>
                        <td>{count:,}</td>
                        <td>{percentage:.1f}%</td>
                        <td>{is_malignant}</td>
                    </tr>
            """
        
        html_content += """
                </table>
            </div>
            
            <div class="section">
                <h2>Model Performance</h2>
        """
        
        for model_name, results in model_results.items():
            html_content += f"""
                <h3>{model_name}</h3>
                <div class="metric">Accuracy: {results.get('accuracy', 0):.3f}</div>
                <div class="metric">Precision: {results.get('precision', 0):.3f}</div>
                <div class="metric">Recall: {results.get('recall', 0):.3f}</div>
                <div class="metric">F1-Score: {results.get('f1_score', 0):.3f}</div>
            """
        
        html_content += """
            </div>
            
            <div class="section">
                <h2>Recommendations</h2>
                <ul>
                    <li><strong>CNN with GradCAM:</strong> Best for interpretable predictions with heatmap visualization</li>
                    <li><strong>Vision Transformers:</strong> Excellent for complex pattern recognition</li>
                    <li><strong>Ensemble Methods:</strong> Combine multiple models for improved accuracy</li>
                    <li><strong>Data Augmentation:</strong> Essential for medical imaging to handle class imbalance</li>
                </ul>
            </div>
        </body>
        </html>
        """
        
        with open(save_path, 'w') as f:
            f.write(html_content)
        
        print(f"Report saved to: {save_path}")

def main():
    """Main function to demonstrate visualization capabilities"""
    print("=== Skin Cancer Detection Visualization Tool ===")
    
    visualizer = SkinCancerVisualizer()
    
    # Load and analyze dataset
    print("Analyzing dataset...")
    df = visualizer.load_dataset_info()
    
    # Create various visualizations
    print("Creating dataset distribution plot...")
    visualizer.plot_dataset_distribution("dataset_distribution.png")
    
    print("Creating sample images plot...")
    visualizer.plot_sample_images(samples_per_class=2, save_path="sample_images.png")
    
    # Example model comparison (you can replace with actual results)
    example_results = {
        'CNN + GradCAM': {'accuracy': 0.85, 'precision': 0.83, 'recall': 0.82, 'f1_score': 0.82},
        'YOLO': {'accuracy': 0.72, 'precision': 0.70, 'recall': 0.68, 'f1_score': 0.69},
        'Vision Transformer': {'accuracy': 0.88, 'precision': 0.86, 'recall': 0.85, 'f1_score': 0.85}
    }
    
    print("Creating model comparison plot...")
    visualizer.plot_model_comparison(example_results, "model_comparison.png")
    
    # Generate comprehensive report
    print("Generating comprehensive report...")
    visualizer.generate_report(example_results)
    
    print("Visualization complete! Check the generated files.")

if __name__ == "__main__":
    main()

