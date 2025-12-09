"""
Brain Cancer Prediction System - Random Test Images Demo
Shows predictions for 10 random images from the test directory
"""

import os
import sys
import random
import argparse
import json
from pathlib import Path
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import numpy as np
import cv2
from brain_cancer_prediction_system import BrainCancerPredictor

def load_tumor_annotations(annotations_file, image_id):
    """Load tumor annotations for a specific image"""
    try:
        with open(annotations_file, 'r') as f:
            coco_data = json.load(f)
        
        # Find annotations for this image
        annotations = []
        for ann in coco_data['annotations']:
            if ann['image_id'] == image_id:
                annotations.append(ann)
        
        return annotations
    except Exception as e:
        print(f"Error loading annotations: {e}")
        return []

def create_tumor_overlay(image_path, annotations, prediction_result):
    """Create an overlay showing tumor segmentation with highlighted cancer areas"""
    # Load the original image
    image = Image.open(image_path).convert('RGB')
    overlay = image.copy()
    draw = ImageDraw.Draw(overlay)
    
    # Colors for different tumor types
    colors = {
        'No Tumor': (0, 255, 0),      # Green
        'Benign': (255, 165, 0),       # Orange  
        'Malignant': (255, 0, 0)      # Red
    }
    
    # Get prediction color
    pred_class = prediction_result['class_name']
    color = colors.get(pred_class, (255, 0, 0))
    
    # Draw tumor annotations with highlighted cancer areas
    for ann in annotations:
        if 'segmentation' in ann and ann['segmentation']:
            # COCO segmentation format
            for seg in ann['segmentation']:
                if len(seg) >= 6:  # At least 3 points
                    # Convert to polygon points
                    points = [(seg[i], seg[i+1]) for i in range(0, len(seg), 2)]
                    # Draw polygon with bright, semi-transparent fill
                    draw.polygon(points, fill=color + (180,), outline=color, width=5)
                    
                    # Add a thicker border to make it more visible
                    draw.polygon(points, outline=color, width=8)
    
    # Create composite image with stronger overlay
    composite = Image.blend(image, overlay, 0.5)
    
    return composite, overlay

def create_segmentation_mask(image_path, annotations, prediction_result):
    """Create a segmentation mask highlighting cancer areas"""
    # Load the original image
    image = Image.open(image_path).convert('RGB')
    width, height = image.size
    
    # Create a mask image
    mask = Image.new('RGBA', (width, height), (0, 0, 0, 0))
    mask_draw = ImageDraw.Draw(mask)
    
    # Colors for different tumor types
    colors = {
        'No Tumor': (0, 255, 0, 150),      # Green with transparency
        'Benign': (255, 165, 0, 150),       # Orange with transparency
        'Malignant': (255, 0, 0, 150)      # Red with transparency
    }
    
    # Get prediction color
    pred_class = prediction_result['class_name']
    color = colors.get(pred_class, (255, 0, 0, 150))
    
    # Draw tumor annotations on mask
    for ann in annotations:
        if 'segmentation' in ann and ann['segmentation']:
            for seg in ann['segmentation']:
                if len(seg) >= 6:  # At least 3 points
                    # Convert to polygon points
                    points = [(seg[i], seg[i+1]) for i in range(0, len(seg), 2)]
                    # Draw filled polygon on mask
                    mask_draw.polygon(points, fill=color, outline=color[:3] + (255,), width=3)
    
    # Create composite with original image and mask
    composite = Image.alpha_composite(image.convert('RGBA'), mask)
    
    return composite, mask

def detect_tumor_regions(image_path, prediction_result):
    """Detect tumor regions using image processing techniques"""
    import cv2
    import numpy as np
    
    # Load image
    image = cv2.imread(str(image_path))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Apply threshold to create binary image
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter contours by area (remove very small ones)
    min_area = (image.shape[0] * image.shape[1]) * 0.01  # 1% of image area
    tumor_contours = [c for c in contours if cv2.contourArea(c) > min_area]
    
    # If no significant contours found, try edge detection
    if not tumor_contours:
        edges = cv2.Canny(blurred, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        tumor_contours = [c for c in contours if cv2.contourArea(c) > min_area]
    
    return tumor_contours

def create_tumor_detection_visualization(image_path, prediction_result):
    """Create visualization with actual tumor detection"""
    import cv2
    import numpy as np
    
    # Load image
    image = cv2.imread(str(image_path))
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Colors for different tumor types
    colors = {
        'No Tumor': (0, 255, 0),      # Green
        'Benign': (255, 165, 0),       # Orange  
        'Malignant': (255, 0, 0)      # Red
    }
    
    # Get prediction color
    pred_class = prediction_result['class_name']
    color = colors.get(pred_class, (255, 0, 0))
    
    # If prediction is not "No Tumor", try to detect tumor regions
    if pred_class != 'No Tumor':
        # Detect tumor regions
        tumor_contours = detect_tumor_regions(image_path, prediction_result)
        
        # Create overlay
        overlay = image_rgb.copy()
        
        if tumor_contours:
            # Draw detected tumor regions
            for contour in tumor_contours:
                # Draw filled contour
                cv2.fillPoly(overlay, [contour], color)
                
                # Draw contour outline
                cv2.drawContours(overlay, [contour], -1, color, 3)
                
                # Get bounding box
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(overlay, (x, y), (x + w, y + h), color, 2)
                
                # Add label
                label = f"{pred_class} ({prediction_result['confidence']:.2f})"
                cv2.putText(overlay, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        else:
            # If no contours detected, use image analysis to find potential tumor areas
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Use adaptive threshold to find regions of interest
            adaptive_thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
            
            # Find contours in adaptive threshold
            contours, _ = cv2.findContours(adaptive_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Draw the largest contour as potential tumor
            if contours:
                largest_contour = max(contours, key=cv2.contourArea)
                if cv2.contourArea(largest_contour) > 100:  # Minimum area threshold
                    cv2.fillPoly(overlay, [largest_contour], color + (100,))
                    cv2.drawContours(overlay, [largest_contour], -1, color, 3)
                    
                    # Add bounding box
                    x, y, w, h = cv2.boundingRect(largest_contour)
                    cv2.rectangle(overlay, (x, y), (x + w, y + h), color, 2)
                    
                    # Add label
                    label = f"{pred_class} ({prediction_result['confidence']:.2f})"
                    cv2.putText(overlay, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    
    # Create composite
    composite = cv2.addWeighted(image_rgb, 0.7, overlay, 0.3, 0)
    
    return Image.fromarray(composite), Image.fromarray(overlay)

def predict_random_images(model_path, test_dir='final/target', num_images=10, show_visualization=True, tumor_details=False, output_dir='final/pred'):
    """
    Predict classes for random images from test directory
    
    Args:
        model_path (str): Path to the trained model
        test_dir (str): Directory containing test images
        num_images (int): Number of random images to predict
        show_visualization (bool): Whether to show visualization
        tumor_details (bool): Whether to create detailed tumor visualizations
        output_dir (str): Directory to save prediction results
    """
    # Check if model exists
    if not os.path.exists(model_path):
        print(f"ERROR: Model file '{model_path}' not found!")
        print("Please train the model first using train_model.py")
        return
    
    # Check if test directory exists
    test_path = Path(test_dir)
    if not test_path.exists():
        print(f"ERROR: Test directory '{test_dir}' not found!")
        return
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_path.absolute()}")
    
    # Find all test images
    test_images = list(test_path.glob('*.jpg'))
    if not test_images:
        print(f"ERROR: No JPG images found in '{test_dir}' directory!")
        return
    
    # Select random images
    if len(test_images) < num_images:
        print(f"WARNING: Only {len(test_images)} images available, showing all of them")
        selected_images = test_images
    else:
        selected_images = random.sample(test_images, num_images)
    
    try:
        # Load the trained model
        print("Loading trained model...")
        predictor = BrainCancerPredictor(model_type='resnet', num_classes=3)
        predictor.load_model(model_path)
        print("Model loaded successfully!")
        
        # Predict for each image
        results = []
        class_names = ['No Tumor', 'Benign', 'Malignant']
        
        print(f"\nAnalyzing {len(selected_images)} random test images...")
        print("=" * 80)
        
        for i, image_path in enumerate(selected_images, 1):
            print(f"\nImage {i}/{len(selected_images)}: {image_path.name}")
            
            try:
                # Make prediction
                result = predictor.predict_single_image(str(image_path))
                
                # Load tumor annotations for segmentation overlay
                annotations_file = test_path / '_annotations.coco.json'
                image_id = None
                annotations = []
                
                if annotations_file.exists():
                    # Try to find image ID from filename
                    try:
                        with open(annotations_file, 'r') as f:
                            coco_data = json.load(f)
                        
                        # Find image ID by filename
                        for img_info in coco_data['images']:
                            if img_info['file_name'] == image_path.name:
                                image_id = img_info['id']
                                break
                        
                        if image_id is not None:
                            annotations = load_tumor_annotations(annotations_file, image_id)
                    except Exception as e:
                        print(f"   Warning: Could not load annotations: {e}")
                
                results.append({
                    'image_path': image_path,
                    'result': result,
                    'annotations': annotations,
                    'image_id': image_id
                })
                
                # Display results
                print(f"   Prediction: {result['class_name']}")
                print(f"   Confidence: {result['confidence']:.3f} ({result['confidence']*100:.1f}%)")
                print(f"   Tumor Annotations: {len(annotations)} found")
                
                # Show all probabilities
                print("   Class Probabilities:")
                for class_name, prob in zip(class_names, result['all_probabilities']):
                    bar_length = int(prob * 15)  # Scale to 15 chars
                    bar = "#" * bar_length + "-" * (15 - bar_length)
                    print(f"      {class_name:12}: {prob:.3f} |{bar}|")
                
            except Exception as e:
                print(f"   Error analyzing image: {str(e)}")
                continue
        
        # Create visualization if requested
        if show_visualization and results:
            # Save main visualization to output directory
            main_viz_path = output_path / 'predictions_overview.png'
            create_prediction_visualization(results, save_path=str(main_viz_path))
            
            # Create detailed tumor visualizations if requested
            if tumor_details:
                print("\nCreating detailed tumor segmentation visualizations...")
                for i, result_data in enumerate(results):
                    if result_data.get('annotations'):
                        print(f"Creating detailed visualization for image {i+1}...")
                        save_path = output_path / f"tumor_analysis_{i+1}.png"
                        create_individual_tumor_visualization(
                            result_data['image_path'],
                            result_data['annotations'],
                            result_data['result'],
                            str(save_path)
                        )
        
        # Summary statistics
        print("\n" + "=" * 80)
        print("SUMMARY STATISTICS")
        print("=" * 80)
        
        # Count predictions
        prediction_counts = {}
        confidence_scores = []
        
        for result_data in results:
            class_name = result_data['result']['class_name']
            confidence = result_data['result']['confidence']
            
            prediction_counts[class_name] = prediction_counts.get(class_name, 0) + 1
            confidence_scores.append(confidence)
        
        print("Prediction Distribution:")
        for class_name, count in prediction_counts.items():
            percentage = (count / len(results)) * 100
            print(f"  {class_name:12}: {count:2d} images ({percentage:5.1f}%)")
        
        if confidence_scores:
            avg_confidence = np.mean(confidence_scores)
            min_confidence = np.min(confidence_scores)
            max_confidence = np.max(confidence_scores)
            print(f"\nConfidence Statistics:")
            print(f"  Average: {avg_confidence:.3f}")
            print(f"  Range:   {min_confidence:.3f} - {max_confidence:.3f}")
        
        print("=" * 80)
        
        # Save prediction results to JSON
        results_data = []
        for result_data in results:
            results_data.append({
                'image_name': result_data['image_path'].name,
                'prediction': result_data['result']['class_name'],
                'confidence': float(result_data['result']['confidence']),
                'probabilities': {
                    'No Tumor': float(result_data['result']['all_probabilities'][0]),
                    'Benign': float(result_data['result']['all_probabilities'][1]),
                    'Malignant': float(result_data['result']['all_probabilities'][2])
                },
                'tumor_annotations_count': len(result_data.get('annotations', [])),
                'image_id': result_data.get('image_id')
            })
        
        # Save results to JSON
        results_json_path = output_path / 'prediction_results.json'
        with open(results_json_path, 'w') as f:
            json.dump(results_data, f, indent=2)
        print(f"\nPrediction results saved to: {results_json_path}")
        
    except Exception as e:
        print(f"ERROR during prediction: {str(e)}")
        return

def create_prediction_visualization(results, save_path=None):
    """Create a comprehensive visualization of all predictions with tumor segmentation"""
    if not results:
        return
    
    num_images = len(results)
    cols = 3
    rows = (num_images + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(20, 6 * rows))
    if rows == 1:
        axes = axes.reshape(1, -1)
    
    class_names = ['No Tumor', 'Benign', 'Malignant']
    colors = ['lightgreen', 'orange', 'red']
    
    for i, result_data in enumerate(results):
        row = i // cols
        col = i % cols
        ax = axes[row, col]
        
        # Get prediction result
        result = result_data['result']
        predicted_class = result['predicted_class']
        confidence = result['confidence']
        annotations = result_data.get('annotations', [])
        
        # Load original image
        image = Image.open(result_data['image_path'])
        
        # Create tumor overlay if annotations exist, otherwise use bounding box
        if annotations:
            try:
                # Use the improved segmentation mask
                composite, mask = create_segmentation_mask(
                    result_data['image_path'], 
                    annotations, 
                    result
                )
                ax.imshow(composite)
            except Exception as e:
                print(f"   Warning: Could not create tumor overlay: {e}")
                ax.imshow(image)
        else:
            # Use actual tumor detection for predicted cancer areas
            try:
                composite, overlay = create_tumor_detection_visualization(
                    result_data['image_path'], 
                    result
                )
                ax.imshow(composite)
            except Exception as e:
                print(f"   Warning: Could not create tumor detection: {e}")
                ax.imshow(image)
        
        # Set title with prediction and tumor info
        tumor_count = len(annotations)
        title = f"{result['class_name']}\nConfidence: {confidence:.3f}\nTumors: {tumor_count}"
        ax.set_title(title, fontsize=10, fontweight='bold')
        ax.axis('off')
        
        # Add color border based on prediction
        for spine in ax.spines.values():
            spine.set_edgecolor(colors[predicted_class])
            spine.set_linewidth(3)
    
    # Hide empty subplots
    for i in range(num_images, rows * cols):
        row = i // cols
        col = i % cols
        axes[row, col].axis('off')
    
    plt.suptitle('Brain Cancer Prediction Results with Tumor Segmentation', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\nVisualization saved as '{save_path}'")
    else:
        plt.savefig('random_predictions_visualization.png', dpi=300, bbox_inches='tight')
        print(f"\nVisualization saved as 'random_predictions_visualization.png'")
    
    plt.show()

def create_individual_tumor_visualization(image_path, annotations, prediction_result, save_path=None):
    """Create a detailed tumor segmentation visualization for a single image"""
    
    # Create figure with original, detection, and composite
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Original image
    original = Image.open(image_path)
    axes[0,0].imshow(original)
    axes[0,0].set_title('Original Brain Scan', fontsize=12, fontweight='bold')
    axes[0,0].axis('off')
    
    if annotations:
        # Use annotation-based segmentation if available
        composite, overlay = create_tumor_overlay(image_path, annotations, prediction_result)
        seg_composite, mask = create_segmentation_mask(image_path, annotations, prediction_result)
        
        # Tumor overlay
        axes[0,1].imshow(overlay)
        axes[0,1].set_title('Annotation Overlay', fontsize=12, fontweight='bold')
        axes[0,1].axis('off')
        
        # Segmentation mask
        axes[1,0].imshow(mask)
        axes[1,0].set_title('Segmentation Mask', fontsize=12, fontweight='bold')
        axes[1,0].axis('off')
        
        # Composite (original + segmentation)
        axes[1,1].imshow(seg_composite)
        axes[1,1].set_title(f'Prediction: {prediction_result["class_name"]}\nConfidence: {prediction_result["confidence"]:.3f}', 
                         fontsize=12, fontweight='bold')
        axes[1,1].axis('off')
    else:
        # Use computer vision-based tumor detection
        try:
            composite, overlay = create_tumor_detection_visualization(image_path, prediction_result)
            
            # Detection overlay
            axes[0,1].imshow(overlay)
            axes[0,1].set_title('Tumor Detection Overlay', fontsize=12, fontweight='bold')
            axes[0,1].axis('off')
            
            # Detection result
            axes[1,0].imshow(composite)
            axes[1,0].set_title('Detection Result', fontsize=12, fontweight='bold')
            axes[1,0].axis('off')
            
            # Final composite
            axes[1,1].imshow(composite)
            axes[1,1].set_title(f'Prediction: {prediction_result["class_name"]}\nConfidence: {prediction_result["confidence"]:.3f}', 
                             fontsize=12, fontweight='bold')
            axes[1,1].axis('off')
            
        except Exception as e:
            print(f"Error in tumor detection: {e}")
            # Fallback to original image
            axes[0,1].imshow(original)
            axes[0,1].set_title('Detection Failed', fontsize=12, fontweight='bold')
            axes[0,1].axis('off')
            
            axes[1,0].imshow(original)
            axes[1,0].set_title('Detection Failed', fontsize=12, fontweight='bold')
            axes[1,0].axis('off')
            
            axes[1,1].imshow(original)
            axes[1,1].set_title(f'Prediction: {prediction_result["class_name"]}\nConfidence: {prediction_result["confidence"]:.3f}', 
                             fontsize=12, fontweight='bold')
            axes[1,1].axis('off')
    
    plt.suptitle(f'Tumor Analysis: {Path(image_path).name}', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Tumor visualization saved as '{save_path}'")
    
    plt.show()
    return fig

def predict_single_image(model_path, image_path, show_image=True):
    """
    Predict the class of a single brain image (legacy function)
    
    Args:
        model_path (str): Path to the trained model
        image_path (str): Path to the image to predict
        show_image (bool): Whether to display the image and prediction
    """
    # Check if files exist
    if not os.path.exists(model_path):
        print(f"Error: Model file '{model_path}' not found!")
        print("Please train the model first using train_model.py")
        return
    
    if not os.path.exists(image_path):
        print(f"Error: Image file '{image_path}' not found!")
        return
    
    try:
        # Load the trained model
        print("Loading trained model...")
        predictor = BrainCancerPredictor(model_type='resnet', num_classes=3)
        predictor.load_model(model_path)
        
        # Make prediction
        print(f"Analyzing image: {image_path}")
        result = predictor.predict_single_image(image_path)
        
        # Display results
        print("\n" + "="*50)
        print("PREDICTION RESULTS")
        print("="*50)
        print(f"Predicted Class: {result['class_name']}")
        print(f"Confidence: {result['confidence']:.4f} ({result['confidence']*100:.2f}%)")
        print("\nClass Probabilities:")
        
        class_names = ['No Tumor', 'Benign', 'Malignant']
        for i, (class_name, prob) in enumerate(zip(class_names, result['all_probabilities'])):
            print(f"  {class_name}: {prob:.4f} ({prob*100:.2f}%)")
        
        # Visualize if requested
        if show_image:
            # Load and display image
            image = Image.open(image_path)
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # Show image
            ax1.imshow(image)
            ax1.set_title(f'Brain Scan\nPredicted: {result["class_name"]}')
            ax1.axis('off')
            
            # Show probability bar chart
            classes = ['No Tumor', 'Benign', 'Malignant']
            probabilities = result['all_probabilities']
            colors = ['green' if i == result['predicted_class'] else 'lightblue' for i in range(len(classes))]
            
            bars = ax2.bar(classes, probabilities, color=colors)
            ax2.set_title('Class Probabilities')
            ax2.set_ylabel('Probability')
            ax2.set_ylim(0, 1)
            
            # Add probability values on bars
            for bar, prob in zip(bars, probabilities):
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{prob:.3f}', ha='center', va='bottom')
            
            plt.tight_layout()
            plt.savefig('prediction_result.png', dpi=300, bbox_inches='tight')
            plt.show()
            
            print(f"\nVisualization saved as 'prediction_result.png'")
        
        print("="*50)
        
    except Exception as e:
        print(f"Error during prediction: {str(e)}")
        return

def main():
    """Main function for command line usage"""
    parser = argparse.ArgumentParser(description='Brain Cancer Prediction System - Random Test Images Demo')
    parser.add_argument('--model', default='best_brain_cancer_model.pth', 
                       help='Path to the trained model (default: brain_cancer_model.pth)')
    parser.add_argument('--test-dir', default='final/target', 
                       help='Directory containing test images (default: final/target)')
    parser.add_argument('--output-dir', default='final/pred', 
                       help='Directory to save prediction results (default: final/pred)')
    parser.add_argument('--num-images', type=int, default=10, 
                       help='Number of random images to predict (default: 10)')
    parser.add_argument('--no-visualization', action='store_true', 
                       help='Do not show visualization')
    parser.add_argument('--single-image', 
                       help='Path to a single image to analyze (overrides random selection)')
    parser.add_argument('--tumor-details', action='store_true',
                       help='Create detailed tumor segmentation visualizations')
    
    args = parser.parse_args()
    
    if args.single_image:
        # Single image prediction
        predict_single_image(
            model_path=args.model,
            image_path=args.single_image,
            show_image=not args.no_visualization
        )
    else:
        # Random images prediction
        predict_random_images(
            model_path=args.model,
            test_dir=args.test_dir,
            num_images=args.num_images,
            show_visualization=not args.no_visualization,
            tumor_details=args.tumor_details,
            output_dir=args.output_dir
        )

if __name__ == "__main__":
    print("Brain Cancer Prediction System")
    print("=" * 50)
    print("This script will predict brain cancer from images in final/target/")
    print("Results will be saved to final/pred/")
    print("Use --help for more options")
    print()
    
    main()
