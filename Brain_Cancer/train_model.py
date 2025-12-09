"""
Simple training script for the Brain Cancer Prediction System
"""

import os
import sys
from brain_cancer_prediction_system import BrainCancerPredictor

def main():
    """Main training function"""
    print("Brain Cancer Prediction System - Training")
    print("=" * 50)
    
    # Check if data directories exist
    required_dirs = ['train', 'valid', 'test']
    for dir_name in required_dirs:
        if not os.path.exists(dir_name):
            print(f"Error: {dir_name} directory not found!")
            print("Please ensure you have the following structure:")
            print("  train/ (with _annotations.coco.json)")
            print("  valid/ (with _annotations.coco.json)")
            print("  test/ (with _annotations.coco.json)")
            return
    
    try:
        # Initialize predictor with ResNet (better performance)
        print("Initializing ResNet-based model...")
        predictor = BrainCancerPredictor(model_type='resnet', num_classes=3)
        
        # Create data loaders
        print("Loading datasets...")
        train_loader, val_loader, test_loader, class_weights = predictor.create_data_loaders(
            train_dir='train',
            val_dir='valid',
            test_dir='test',
            batch_size=16  # Reduced batch size for memory efficiency
        )
        
        # Train the model
        print("Starting training...")
        print("This may take a while depending on your hardware...")
        
        history = predictor.train(
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=20,  # Reduced epochs for faster training
            learning_rate=0.001,
            class_weights=class_weights
        )
        
        # Plot training history
        print("Generating training plots...")
        predictor.plot_training_history()
        
        # Evaluate on test set
        print("Evaluating on test set...")
        predictions, targets, probabilities = predictor.evaluate(test_loader)
        
        # Calculate metrics
        import numpy as np
        accuracy = np.mean(np.array(predictions) == np.array(targets))
        print(f"\nTest Accuracy: {accuracy:.4f}")
        
        # Classification report
        from sklearn.metrics import classification_report
        class_names = ['No Tumor', 'Benign', 'Malignant']
        print("\nClassification Report:")
        print(classification_report(targets, predictions, target_names=class_names))
        
        # Generate visualizations
        print("Generating evaluation plots...")
        predictor.plot_confusion_matrix(predictions, targets, class_names)
        predictor.plot_roc_curves(targets, probabilities, class_names)
        
        # Save the model
        predictor.save_model('brain_cancer_model.pth')
        
        print("\n" + "="*50)
        print("Training completed successfully!")
        print("Files generated:")
        print("  - brain_cancer_model.pth (trained model)")
        print("  - training_history.png (training curves)")
        print("  - confusion_matrix.png (confusion matrix)")
        print("  - roc_curves.png (ROC curves)")
        print("="*50)
        
    except Exception as e:
        print(f"Error during training: {str(e)}")
        print("Please check your data and try again.")
        return

if __name__ == "__main__":
    main()
