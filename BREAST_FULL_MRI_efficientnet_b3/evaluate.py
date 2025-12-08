"""
Evaluation script for Breast Cancer MRI Classification
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report,
    precision_score, recall_score, f1_score, roc_auc_score
)
from tqdm import tqdm
import config
from model import get_model, get_device
from data_loader import get_dataloaders


def evaluate(model_path=None):
    """Evaluate the model on test set"""
    print("=" * 60)
    print("Breast Cancer MRI Classification - Evaluation")
    print("=" * 60)
    
    # Get device
    device = get_device()
    
    # Load model
    if model_path is None:
        model_path = config.BEST_MODEL_PATH
    
    print(f"\nLoading model from: {model_path}")
    model = get_model(config.MODEL_NAME, pretrained=False)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    
    # Get test loader
    print("\nLoading test dataset...")
    _, _, test_loader = get_dataloaders()
    print(f"Test samples: {len(test_loader.dataset)}")
    
    # Evaluate
    print("\nEvaluating on test set...")
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        pbar = tqdm(test_loader, desc="Evaluating")
        for images, labels in pbar:
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    # Convert to numpy arrays
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    
    # Calculate metrics
    print("\n" + "-" * 60)
    print("EVALUATION METRICS")
    print("-" * 60)
    
    # Overall accuracy
    overall_accuracy = accuracy_score(all_labels, all_preds)
    print(f"\nOverall Accuracy: {overall_accuracy * 100:.2f}%")
    
    # Per-class accuracy
    cm = confusion_matrix(all_labels, all_preds)
    per_class_accuracy = cm.diagonal() / cm.sum(axis=1)
    
    print("\nPer-Class Accuracy:")
    for i, class_name in enumerate(config.CLASS_NAMES):
        print(f"  {class_name}: {per_class_accuracy[i] * 100:.2f}%")
    
    # Additional metrics
    precision = precision_score(all_labels, all_preds, average=None)
    recall = recall_score(all_labels, all_preds, average=None)
    f1 = f1_score(all_labels, all_preds, average=None)
    
    print("\nPer-Class Metrics:")
    print(f"{'Class':<15} {'Precision':<12} {'Recall':<12} {'F1-Score':<12}")
    print("-" * 50)
    for i, class_name in enumerate(config.CLASS_NAMES):
        print(f"{class_name:<15} {precision[i]:<12.4f} {recall[i]:<12.4f} {f1[i]:<12.4f}")
    
    # Macro averages
    macro_precision = precision_score(all_labels, all_preds, average='macro')
    macro_recall = recall_score(all_labels, all_preds, average='macro')
    macro_f1 = f1_score(all_labels, all_preds, average='macro')
    
    print("\nMacro Averages:")
    print(f"  Precision: {macro_precision:.4f}")
    print(f"  Recall: {macro_recall:.4f}")
    print(f"  F1-Score: {macro_f1:.4f}")
    
    # ROC-AUC
    try:
        roc_auc = roc_auc_score(all_labels, all_probs[:, 1])
        print(f"  ROC-AUC: {roc_auc:.4f}")
    except Exception as e:
        print(f"  ROC-AUC: Could not calculate ({e})")
    
    # Confusion Matrix
    print("\nConfusion Matrix:")
    print(cm)
    
    # Save accuracy table
    accuracy_table = {
        'Metric': ['Overall Accuracy'] + [f'{name} Accuracy' for name in config.CLASS_NAMES],
        'Value (%)': [overall_accuracy * 100] + [acc * 100 for acc in per_class_accuracy]
    }
    
    # Create and save confusion matrix visualization
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=config.CLASS_NAMES,
        yticklabels=config.CLASS_NAMES,
        cbar_kws={'label': 'Count'}
    )
    plt.title('Confusion Matrix', fontsize=16, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    plt.savefig(config.CONFUSION_MATRIX_PATH, dpi=300, bbox_inches='tight')
    print(f"\nConfusion matrix saved to {config.CONFUSION_MATRIX_PATH}")
    
    # Save detailed metrics report
    with open(config.METRICS_REPORT_PATH, 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("BREAST CANCER MRI CLASSIFICATION - EVALUATION REPORT\n")
        f.write("=" * 60 + "\n\n")
        
        f.write("ACCURACY TABLE\n")
        f.write("-" * 60 + "\n")
        f.write(f"{'Metric':<30} {'Value (%)':<15}\n")
        f.write("-" * 60 + "\n")
        f.write(f"{'Overall Accuracy':<30} {overall_accuracy * 100:<15.2f}\n")
        for i, class_name in enumerate(config.CLASS_NAMES):
            f.write(f"{class_name + ' Accuracy':<30} {per_class_accuracy[i] * 100:<15.2f}\n")
        
        f.write("\n\nPER-CLASS METRICS\n")
        f.write("-" * 60 + "\n")
        f.write(f"{'Class':<15} {'Precision':<12} {'Recall':<12} {'F1-Score':<12}\n")
        f.write("-" * 50 + "\n")
        for i, class_name in enumerate(config.CLASS_NAMES):
            f.write(f"{class_name:<15} {precision[i]:<12.4f} {recall[i]:<12.4f} {f1[i]:<12.4f}\n")
        
        f.write("\n\nMACRO AVERAGES\n")
        f.write("-" * 60 + "\n")
        f.write(f"Precision: {macro_precision:.4f}\n")
        f.write(f"Recall: {macro_recall:.4f}\n")
        f.write(f"F1-Score: {macro_f1:.4f}\n")
        if 'roc_auc' in locals():
            f.write(f"ROC-AUC: {roc_auc:.4f}\n")
        
        f.write("\n\nCONFUSION MATRIX\n")
        f.write("-" * 60 + "\n")
        f.write(f"{'':<15} {'Predicted ' + config.CLASS_NAMES[0]:<20} {'Predicted ' + config.CLASS_NAMES[1]:<20}\n")
        f.write(f"{'True ' + config.CLASS_NAMES[0]:<15} {cm[0][0]:<20} {cm[0][1]:<20}\n")
        f.write(f"{'True ' + config.CLASS_NAMES[1]:<15} {cm[1][0]:<20} {cm[1][1]:<20}\n")
        
        f.write("\n\nCLASSIFICATION REPORT\n")
        f.write("-" * 60 + "\n")
        f.write(classification_report(all_labels, all_preds, target_names=config.CLASS_NAMES))
    
    print(f"Detailed metrics report saved to {config.METRICS_REPORT_PATH}")
    
    print("\n" + "=" * 60)
    print("Evaluation completed!")
    print("=" * 60)
    
    return {
        'accuracy': overall_accuracy,
        'per_class_accuracy': per_class_accuracy,
        'confusion_matrix': cm,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }


if __name__ == "__main__":
    evaluate()


