"""
Main entry point for Breast Cancer MRI Classification
"""
import argparse
import sys
import config
from train import train
from evaluate import evaluate
from predict import main as predict_main


def main():
    parser = argparse.ArgumentParser(
        description='Breast Cancer MRI Classification System',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train the model
  python main.py train

  # Evaluate the model
  python main.py evaluate --model models/best_model.pth

  # Predict on a single image
  python main.py predict --image path/to/image.jpg

  # Predict on a folder of images
  python main.py predict --folder path/to/images/
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to execute')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train the model')
    
    # Evaluate command
    eval_parser = subparsers.add_parser('evaluate', help='Evaluate the model')
    eval_parser.add_argument(
        '--model',
        type=str,
        default=config.BEST_MODEL_PATH,
        help='Path to the trained model file'
    )
    
    # Predict command
    predict_parser = subparsers.add_parser('predict', help='Make predictions on new images')
    predict_parser.add_argument(
        '--image',
        type=str,
        help='Path to a single image file'
    )
    predict_parser.add_argument(
        '--folder',
        type=str,
        help='Path to a folder containing images'
    )
    predict_parser.add_argument(
        '--model',
        type=str,
        default=config.BEST_MODEL_PATH,
        help='Path to the trained model file'
    )
    predict_parser.add_argument(
        '--verbose',
        action='store_true',
        help='Show detailed probabilities for each class'
    )
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    # Execute the requested command
    if args.command == 'train':
        print("Starting training...")
        train()
    
    elif args.command == 'evaluate':
        print("Starting evaluation...")
        evaluate(model_path=args.model)
    
    elif args.command == 'predict':
        print("Starting prediction...")
        # Pass args directly to predict_main
        predict_main(args)


if __name__ == "__main__":
    main()

