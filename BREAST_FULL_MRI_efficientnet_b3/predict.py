"""
Prediction script for Breast Cancer MRI Classification
"""
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import argparse
import os
import config
from model import get_model, get_device


def load_image(image_path):
    """Load and preprocess a single image"""
    transform = transforms.Compose([
        transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    try:
        image = Image.open(image_path).convert('RGB')
        image_tensor = transform(image).unsqueeze(0)  # Add batch dimension
        return image_tensor, image
    except Exception as e:
        raise ValueError(f"Error loading image {image_path}: {e}")


def predict_image(model, image_path, device, return_probabilities=False):
    """
    Predict class for a single image
    
    Args:
        model: Trained PyTorch model
        image_path: Path to the image file
        device: Device to run inference on
        return_probabilities: If True, return class probabilities
    
    Returns:
        prediction: Predicted class name
        confidence: Confidence score
        probabilities: (optional) Dictionary of class probabilities
    """
    model.eval()
    
    # Load and preprocess image
    image_tensor, original_image = load_image(image_path)
    image_tensor = image_tensor.to(device)
    
    # Make prediction
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = F.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)
    
    # Get class name
    predicted_class = config.CLASS_NAMES[predicted.item()]
    confidence_score = confidence.item()
    
    if return_probabilities:
        prob_dict = {
            config.CLASS_NAMES[i]: probabilities[0][i].item() 
            for i in range(len(config.CLASS_NAMES))
        }
        return predicted_class, confidence_score, prob_dict
    
    return predicted_class, confidence_score


def predict_batch(model, image_paths, device):
    """
    Predict classes for multiple images
    
    Args:
        model: Trained PyTorch model
        image_paths: List of paths to image files
        device: Device to run inference on
    
    Returns:
        results: List of tuples (image_path, predicted_class, confidence)
    """
    results = []
    for image_path in image_paths:
        try:
            pred_class, confidence = predict_image(model, image_path, device)
            results.append((image_path, pred_class, confidence))
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            results.append((image_path, "ERROR", 0.0))
    return results


def main(args=None):
    if args is None:
        parser = argparse.ArgumentParser(
            description='Predict breast cancer from MRI images'
        )
        parser.add_argument(
            '--image',
            type=str,
            help='Path to a single image file'
        )
        parser.add_argument(
            '--folder',
            type=str,
            help='Path to a folder containing images'
        )
        parser.add_argument(
            '--model',
            type=str,
            default=config.BEST_MODEL_PATH,
            help='Path to the trained model file'
        )
        parser.add_argument(
            '--verbose',
            action='store_true',
            help='Show detailed probabilities for each class'
        )
        
        args = parser.parse_args()
    
    if not args.image and not args.folder:
        error_msg = "Either --image or --folder must be provided"
        if 'parser' in locals():
            parser.error(error_msg)
        else:
            raise ValueError(error_msg)
    
    # Get device
    device = get_device()
    
    # Load model
    print(f"Loading model from: {args.model}")
    if not os.path.exists(args.model):
        raise FileNotFoundError(f"Model file not found: {args.model}")
    
    model = get_model(config.MODEL_NAME, pretrained=False)
    model.load_state_dict(torch.load(args.model, map_location=device))
    model = model.to(device)
    model.eval()
    print("Model loaded successfully!")
    
    # Process single image
    if args.image:
        if not os.path.exists(args.image):
            raise FileNotFoundError(f"Image file not found: {args.image}")
        
        print(f"\nPredicting: {args.image}")
        if args.verbose:
            pred_class, confidence, probabilities = predict_image(
                model, args.image, device, return_probabilities=True
            )
            print(f"\nPrediction: {pred_class}")
            print(f"Confidence: {confidence * 100:.2f}%")
            print("\nClass Probabilities:")
            for class_name, prob in probabilities.items():
                print(f"  {class_name}: {prob * 100:.2f}%")
        else:
            pred_class, confidence = predict_image(model, args.image, device)
            print(f"\nPrediction: {pred_class}")
            print(f"Confidence: {confidence * 100:.2f}%")
    
    # Process folder of images
    elif args.folder:
        if not os.path.exists(args.folder):
            raise FileNotFoundError(f"Folder not found: {args.folder}")
        
        # Get all image files
        image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
        image_paths = [
            os.path.join(args.folder, f)
            for f in os.listdir(args.folder)
            if f.lower().endswith(image_extensions)
        ]
        
        if not image_paths:
            print(f"No image files found in {args.folder}")
            return
        
        print(f"\nProcessing {len(image_paths)} images...")
        results = predict_batch(model, image_paths, device)
        
        # Print results
        print("\n" + "=" * 80)
        print(f"{'Image':<50} {'Prediction':<15} {'Confidence':<15}")
        print("=" * 80)
        for image_path, pred_class, confidence in results:
            image_name = os.path.basename(image_path)
            print(f"{image_name:<50} {pred_class:<15} {confidence * 100:<15.2f}%")


if __name__ == "__main__":
    main()

