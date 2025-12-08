"""
Data loading and preprocessing for Breast Cancer MRI Classification
"""
import os
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import config


class BreastMRIDataset(Dataset):
    """Custom Dataset class for loading breast MRI images"""
    
    def __init__(self, data_dir, transform=None):
        """
        Args:
            data_dir: Path to directory containing 'Benign' and 'Malignant' folders
            transform: Optional transform to be applied on images
        """
        self.data_dir = data_dir
        self.transform = transform
        self.images = []
        self.labels = []
        
        # Load images from Benign folder (label 0)
        benign_dir = os.path.join(data_dir, "Benign")
        if os.path.exists(benign_dir):
            for img_name in os.listdir(benign_dir):
                if img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                    self.images.append(os.path.join(benign_dir, img_name))
                    self.labels.append(0)  # Benign = 0
        
        # Load images from Malignant folder (label 1)
        malignant_dir = os.path.join(data_dir, "Malignant")
        if os.path.exists(malignant_dir):
            for img_name in os.listdir(malignant_dir):
                if img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                    self.images.append(os.path.join(malignant_dir, img_name))
                    self.labels.append(1)  # Malignant = 1
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]
        
        # Load image
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return a black image as fallback
            image = Image.new('RGB', (config.IMAGE_SIZE, config.IMAGE_SIZE))
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        return image, label


def get_transforms(is_training=False):
    """Get data transforms for training or validation/test"""
    if is_training:
        # Training transforms with augmentation
        return transforms.Compose([
            transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
            transforms.RandomRotation(config.ROTATION_DEGREES),
            transforms.RandomHorizontalFlip(p=config.HORIZONTAL_FLIP_PROB),
            transforms.ColorJitter(
                brightness=config.BRIGHTNESS_RANGE,
                contrast=config.CONTRAST_RANGE
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])  # ImageNet normalization
        ])
    else:
        # Validation/Test transforms (no augmentation)
        return transforms.Compose([
            transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])


def get_dataloaders():
    """Create DataLoaders for train, validation, and test sets"""
    # Check if directories exist
    print(f"Checking dataset paths...")
    print(f"  Train dir: {config.TRAIN_DIR} (exists: {os.path.exists(config.TRAIN_DIR)})")
    print(f"  Val dir: {config.VAL_DIR} (exists: {os.path.exists(config.VAL_DIR)})")
    print(f"  Test dir: {config.TEST_DIR} (exists: {os.path.exists(config.TEST_DIR)})")
    
    # Training dataset with augmentation
    train_dataset = BreastMRIDataset(
        config.TRAIN_DIR,
        transform=get_transforms(is_training=True)
    )
    print(f"  Training samples found: {len(train_dataset)}")
    
    # Validation dataset without augmentation
    val_dataset = BreastMRIDataset(
        config.VAL_DIR,
        transform=get_transforms(is_training=False)
    )
    print(f"  Validation samples found: {len(val_dataset)}")
    
    # Test dataset without augmentation
    test_dataset = BreastMRIDataset(
        config.TEST_DIR,
        transform=get_transforms(is_training=False)
    )
    print(f"  Test samples found: {len(test_dataset)}")
    
    # Validate datasets are not empty
    if len(train_dataset) == 0:
        raise ValueError(
            f"No training images found in {config.TRAIN_DIR}. "
            f"Please ensure the dataset is properly organized with 'Benign' and 'Malignant' subfolders."
        )
    if len(val_dataset) == 0:
        raise ValueError(
            f"No validation images found in {config.VAL_DIR}. "
            f"Please ensure the dataset is properly organized with 'Benign' and 'Malignant' subfolders."
        )
    if len(test_dataset) == 0:
        print(f"Warning: No test images found in {config.TEST_DIR}")
    
    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True if config.USE_GPU else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True if config.USE_GPU else False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True if config.USE_GPU else False
    )
    
    return train_loader, val_loader, test_loader


