"""
Model architecture for Breast Cancer MRI Classification using Transfer Learning
"""
import torch
import torch.nn as nn
import torchvision.models as models
import config


def get_model(model_name=None, pretrained=True):
    """
    Create a model with transfer learning
    
    Args:
        model_name: Name of the model architecture (defaults to config.MODEL_NAME)
        pretrained: Whether to use pretrained weights
    
    Returns:
        model: PyTorch model ready for training
    """
    if model_name is None:
        model_name = config.MODEL_NAME
    
    model_name = model_name.lower()
    
    # EfficientNet models
    if model_name == "efficientnet_b0":
        model = models.efficientnet_b0(pretrained=pretrained)
        num_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(num_features, config.NUM_CLASSES)
        )
    
    elif model_name == "efficientnet_b1":
        model = models.efficientnet_b1(pretrained=pretrained)
        num_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(num_features, config.NUM_CLASSES)
        )
    
    elif model_name == "efficientnet_b2":
        model = models.efficientnet_b2(pretrained=pretrained)
        num_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.3, inplace=True),
            nn.Linear(num_features, config.NUM_CLASSES)
        )
    
    elif model_name == "efficientnet_b3":
        model = models.efficientnet_b3(pretrained=pretrained)
        num_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.3, inplace=True),
            nn.Linear(num_features, config.NUM_CLASSES)
        )
    
    elif model_name == "efficientnet_b4":
        model = models.efficientnet_b4(pretrained=pretrained)
        num_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.3, inplace=True),
            nn.Linear(num_features, config.NUM_CLASSES)
        )
    
    # ResNet models
    elif model_name == "resnet50":
        model = models.resnet50(pretrained=pretrained)
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, config.NUM_CLASSES)
    
    elif model_name == "resnet101":
        model = models.resnet101(pretrained=pretrained)
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, config.NUM_CLASSES)
    
    # DenseNet models
    elif model_name == "densenet121":
        model = models.densenet121(pretrained=pretrained)
        num_features = model.classifier.in_features
        model.classifier = nn.Linear(num_features, config.NUM_CLASSES)
    
    elif model_name == "densenet169":
        model = models.densenet169(pretrained=pretrained)
        num_features = model.classifier.in_features
        model.classifier = nn.Linear(num_features, config.NUM_CLASSES)
    
    else:
        raise ValueError(f"Unknown model name: {model_name}. "
                        f"Supported: efficientnet_b0, efficientnet_b1, efficientnet_b2, "
                        f"efficientnet_b3, efficientnet_b4, resnet50, resnet101, "
                        f"densenet121, densenet169")
    
    return model


def get_device():
    """Get the device (GPU if available, else CPU)"""
    if config.USE_GPU and torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("Using CPU")
    return device


class DistillationLoss(nn.Module):
    """
    Knowledge Distillation Loss
    Combines hard loss (cross-entropy with true labels) and 
    soft loss (KL divergence with teacher's soft predictions)
    """
    def __init__(self, temperature=4.0, alpha=0.7):
        """
        Args:
            temperature: Temperature for softmax (higher = softer probabilities)
            alpha: Weight for distillation loss (1-alpha for hard loss)
        """
        super(DistillationLoss, self).__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.ce_loss = nn.CrossEntropyLoss()
        self.kl_loss = nn.KLDivLoss(reduction='batchmean')
    
    def forward(self, student_logits, teacher_logits, labels):
        """
        Args:
            student_logits: Student model raw outputs (logits)
            teacher_logits: Teacher model raw outputs (logits)
            labels: True labels
        Returns:
            Combined distillation loss
        """
        # Hard loss: Cross-entropy with true labels
        hard_loss = self.ce_loss(student_logits, labels)
        
        # Soft loss: KL divergence between teacher and student softmax
        # Apply temperature scaling
        student_soft = nn.functional.log_softmax(student_logits / self.temperature, dim=1)
        teacher_soft = nn.functional.softmax(teacher_logits / self.temperature, dim=1)
        soft_loss = self.kl_loss(student_soft, teacher_soft) * (self.temperature ** 2)
        
        # Combine losses
        total_loss = self.alpha * soft_loss + (1 - self.alpha) * hard_loss
        
        return total_loss, hard_loss, soft_loss

