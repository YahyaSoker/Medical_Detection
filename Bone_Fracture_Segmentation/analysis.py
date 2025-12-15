"""
Data Analysis Script for Bone Fracture Detection Dataset
Analyzes dataset distribution and generates visualizations
"""

import os
import yaml
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

# Paths
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "BoneFractureYolo8"
DATA_YAML = DATA_DIR / "data.yaml"
OUTPUT_DIR = BASE_DIR / "output" / "analysis"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def load_data_config():
    """Load data.yaml configuration"""
    with open(DATA_YAML, 'r') as f:
        config = yaml.safe_load(f)
    return config

def count_images_in_split(split_name):
    """Count images in a split directory"""
    split_path = DATA_DIR / split_name / "images"
    if split_path.exists():
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        count = sum(1 for f in split_path.rglob('*') if f.suffix.lower() in image_extensions)
        return count
    return 0

def count_annotations_per_class(split_name, class_names):
    """Count annotations per class in a split"""
    labels_path = DATA_DIR / split_name / "labels"
    class_counts = Counter()
    total_annotations = 0
    
    if labels_path.exists():
        for label_file in labels_path.rglob('*.txt'):
            try:
                with open(label_file, 'r') as f:
                    lines = f.readlines()
                    for line in lines:
                        line = line.strip()
                        if line:
                            parts = line.split()
                            if len(parts) >= 5:
                                class_id = int(parts[0])
                                if 0 <= class_id < len(class_names):
                                    class_counts[class_names[class_id]] += 1
                                    total_annotations += 1
            except Exception as e:
                print(f"Error reading {label_file}: {e}")
    
    return class_counts, total_annotations

def create_pie_chart_class_distribution(class_counts, output_path):
    """Create pie chart for class distribution"""
    if not class_counts:
        print("No annotations found for pie chart")
        return
    
    classes = list(class_counts.keys())
    counts = list(class_counts.values())
    
    plt.figure(figsize=(10, 8))
    colors = sns.color_palette("husl", len(classes))
    plt.pie(counts, labels=classes, autopct='%1.1f%%', startangle=90, colors=colors)
    plt.title('Class Distribution Across All Splits', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path / 'class_distribution_pie.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved pie chart to {output_path / 'class_distribution_pie.png'}")

def create_bar_chart_images_per_split(split_counts, output_path):
    """Create bar chart for images per split"""
    splits = list(split_counts.keys())
    counts = list(split_counts.values())
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(splits, counts, color=['#3498db', '#2ecc71', '#e74c3c'])
    plt.xlabel('Dataset Split', fontsize=12, fontweight='bold')
    plt.ylabel('Number of Images', fontsize=12, fontweight='bold')
    plt.title('Number of Images per Dataset Split', fontsize=16, fontweight='bold')
    plt.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}',
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_path / 'images_per_split.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved bar chart to {output_path / 'images_per_split.png'}")

def create_bar_chart_annotations_per_class(class_counts, output_path):
    """Create bar chart for annotations per class"""
    if not class_counts:
        print("No annotations found for bar chart")
        return
    
    classes = list(class_counts.keys())
    counts = list(class_counts.values())
    
    plt.figure(figsize=(14, 8))
    bars = plt.barh(classes, counts, color=sns.color_palette("husl", len(classes)))
    plt.xlabel('Number of Annotations', fontsize=12, fontweight='bold')
    plt.ylabel('Class', fontsize=12, fontweight='bold')
    plt.title('Number of Annotations per Class', fontsize=16, fontweight='bold')
    plt.grid(axis='x', alpha=0.3)
    
    # Add value labels on bars
    for i, bar in enumerate(bars):
        width = bar.get_width()
        plt.text(width, bar.get_y() + bar.get_height()/2.,
                f'{int(width)}',
                ha='left', va='center', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_path / 'annotations_per_class.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved bar chart to {output_path / 'annotations_per_class.png'}")

def create_statistics_table(split_counts, class_counts, total_annotations, class_names, output_path):
    """Create statistics summary table"""
    # Dataset split statistics
    split_stats = {
        'Split': ['Train', 'Valid', 'Test', 'Total'],
        'Images': [
            split_counts.get('train', 0),
            split_counts.get('valid', 0),
            split_counts.get('test', 0),
            sum(split_counts.values())
        ]
    }
    split_df = pd.DataFrame(split_stats)
    
    # Class statistics
    class_stats = {
        'Class': class_names,
        'Annotations': [class_counts.get(name, 0) for name in class_names]
    }
    class_df = pd.DataFrame(class_stats)
    class_df['Percentage'] = (class_df['Annotations'] / total_annotations * 100).round(2)
    
    # Save to CSV
    split_df.to_csv(output_path / 'split_statistics.csv', index=False)
    class_df.to_csv(output_path / 'class_statistics.csv', index=False)
    
    # Create visual table
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Split statistics table
    ax1.axis('tight')
    ax1.axis('off')
    table1 = ax1.table(cellText=split_df.values, colLabels=split_df.columns,
                      cellLoc='center', loc='center')
    table1.auto_set_font_size(False)
    table1.set_fontsize(11)
    table1.scale(1.2, 1.5)
    ax1.set_title('Dataset Split Statistics', fontsize=14, fontweight='bold', pad=20)
    
    # Class statistics table
    ax2.axis('tight')
    ax2.axis('off')
    table2 = ax2.table(cellText=class_df.values, colLabels=class_df.columns,
                      cellLoc='center', loc='center')
    table2.auto_set_font_size(False)
    table2.set_fontsize(10)
    table2.scale(1.2, 1.5)
    ax2.set_title('Class Statistics', fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig(output_path / 'statistics_tables.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved statistics tables to {output_path / 'statistics_tables.png'}")
    print(f"Saved CSV files to {output_path}")

def main():
    """Main analysis function"""
    print("=" * 60)
    print("Bone Fracture Detection Dataset Analysis")
    print("=" * 60)
    
    # Load configuration
    config = load_data_config()
    class_names = config['names']
    num_classes = config['nc']
    
    print(f"\nDataset Configuration:")
    print(f"  Number of classes: {num_classes}")
    print(f"  Classes: {class_names}")
    
    # Count images per split
    print("\nCounting images per split...")
    split_counts = {}
    for split in ['train', 'valid', 'test']:
        count = count_images_in_split(split)
        split_counts[split] = count
        print(f"  {split}: {count} images")
    
    # Count annotations per class across all splits
    print("\nCounting annotations per class...")
    all_class_counts = Counter()
    total_annotations = 0
    
    for split in ['train', 'valid', 'test']:
        class_counts, split_total = count_annotations_per_class(split, class_names)
        all_class_counts.update(class_counts)
        total_annotations += split_total
        print(f"  {split}: {split_total} annotations")
    
    print(f"\nTotal annotations: {total_annotations}")
    print(f"\nAnnotations per class:")
    for class_name in class_names:
        count = all_class_counts.get(class_name, 0)
        percentage = (count / total_annotations * 100) if total_annotations > 0 else 0
        print(f"  {class_name}: {count} ({percentage:.2f}%)")
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    create_pie_chart_class_distribution(all_class_counts, OUTPUT_DIR)
    create_bar_chart_images_per_split(split_counts, OUTPUT_DIR)
    create_bar_chart_annotations_per_class(all_class_counts, OUTPUT_DIR)
    create_statistics_table(split_counts, all_class_counts, total_annotations, class_names, OUTPUT_DIR)
    
    print("\n" + "=" * 60)
    print("Analysis complete! All outputs saved to:", OUTPUT_DIR)
    print("=" * 60)

if __name__ == "__main__":
    main()

