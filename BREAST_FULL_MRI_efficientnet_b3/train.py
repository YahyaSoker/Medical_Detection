"""
Training script for Breast Cancer MRI Classification
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
import pandas as pd
from tqdm import tqdm
import os
import config
from model import get_model, get_device, DistillationLoss
from data_loader import get_dataloaders


def train_epoch(model, train_loader, criterion, optimizer, device, teacher_model=None, distill_criterion=None):
    """Train for one epoch"""
    model.train()
    if teacher_model is not None:
        teacher_model.eval()  # Teacher is always in eval mode
    
    running_loss = 0.0
    running_hard_loss = 0.0
    running_soft_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc="Training")
    for images, labels in pbar:
        images = images.to(device)
        labels = labels.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        student_outputs = model(images)
        
        # Knowledge distillation
        if teacher_model is not None and distill_criterion is not None:
            with torch.no_grad():
                teacher_outputs = teacher_model(images)
            loss, hard_loss, soft_loss = distill_criterion(student_outputs, teacher_outputs, labels)
            running_hard_loss += hard_loss.item()
            running_soft_loss += soft_loss.item()
        else:
            loss = criterion(student_outputs, labels)
            hard_loss = loss
            soft_loss = torch.tensor(0.0)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Statistics
        running_loss += loss.item()
        _, predicted = torch.max(student_outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        # Update progress bar
        if teacher_model is not None:
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'hard': f'{hard_loss.item():.4f}',
                'soft': f'{soft_loss.item():.4f}',
                'acc': f'{100 * correct / total:.2f}%'
            })
        else:
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100 * correct / total:.2f}%'
            })
    
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100 * correct / total
    
    if teacher_model is not None:
        epoch_hard_loss = running_hard_loss / len(train_loader)
        epoch_soft_loss = running_soft_loss / len(train_loader)
        return epoch_loss, epoch_acc, epoch_hard_loss, epoch_soft_loss
    
    return epoch_loss, epoch_acc


def validate(model, val_loader, criterion, device):
    """Validate the model"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        pbar = tqdm(val_loader, desc="Validation")
        for images, labels in pbar:
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100 * correct / total:.2f}%'
            })
    
    epoch_loss = running_loss / len(val_loader)
    epoch_acc = 100 * correct / total
    return epoch_loss, epoch_acc


def train():
    """Main training function"""
    print("=" * 60)
    print("Breast Cancer MRI Classification - Training")
    if config.USE_DISTILLATION:
        print("Mode: Knowledge Distillation")
        print(f"Teacher: {config.TEACHER_MODEL}, Student: {config.STUDENT_MODEL}")
    else:
        print(f"Mode: Standard Training ({config.MODEL_NAME})")
    print("=" * 60)
    
    # Get device
    device = get_device()
    
    # Get data loaders
    print("\nLoading datasets...")
    train_loader, val_loader, _ = get_dataloaders()
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")
    
    # Knowledge Distillation setup
    teacher_model = None
    distill_criterion = None
    
    if config.USE_DISTILLATION:
        # Create teacher model
        print(f"\nLoading teacher model: {config.TEACHER_MODEL}")
        teacher_model = get_model(config.TEACHER_MODEL, pretrained=True)
        
        # Load teacher weights if provided
        teacher_path = config.TEACHER_MODEL_PATH
        if teacher_path:
            # Handle relative paths
            if not os.path.isabs(teacher_path):
                teacher_path = os.path.join(config.BASE_DIR, teacher_path)
            
            if os.path.exists(teacher_path):
                print(f"Loading teacher weights from {teacher_path}")
                try:
                    checkpoint = torch.load(teacher_path, map_location=device)
                    teacher_model.load_state_dict(checkpoint, strict=True)
                    print("✓ Teacher weights loaded successfully")
                except RuntimeError as e:
                    if "size mismatch" in str(e) or "Missing key" in str(e):
                        print(f"\n⚠ ERROR: Model architecture mismatch!")
                        print(f"  The checkpoint file contains weights from a different model architecture.")
                        print(f"  Expected teacher model: {config.TEACHER_MODEL}")
                        print(f"  Checkpoint file: {teacher_path}")
                        print(f"\n  Solutions:")
                        print(f"  1. Train the teacher model first:")
                        print(f"     - Set USE_DISTILLATION = False")
                        print(f"     - Set MODEL_NAME = '{config.TEACHER_MODEL}'")
                        print(f"     - Run training, then enable distillation")
                        print(f"  2. Use ImageNet pretrained weights:")
                        print(f"     - Set TEACHER_MODEL_PATH = None")
                        print(f"  3. Use the correct checkpoint file for {config.TEACHER_MODEL}")
                        raise RuntimeError(
                            f"Cannot load teacher model: architecture mismatch. "
                            f"Checkpoint is for a different model than {config.TEACHER_MODEL}"
                        )
                    else:
                        raise
            else:
                print(f"Warning: Teacher model path not found: {teacher_path}")
                print("Using ImageNet pretrained weights for teacher")
        else:
            print("Using ImageNet pretrained weights for teacher")
        
        teacher_model = teacher_model.to(device)
        teacher_model.eval()  # Teacher is frozen during training
        
        # Create student model
        print(f"\nCreating student model: {config.STUDENT_MODEL}")
        model = get_model(config.STUDENT_MODEL, pretrained=True)
        model = model.to(device)
        
        # Create distillation loss
        distill_criterion = DistillationLoss(
            temperature=config.DISTILLATION_TEMPERATURE,
            alpha=config.DISTILLATION_ALPHA
        )
        criterion = nn.CrossEntropyLoss()  # Still needed for validation
    else:
        # Standard training
        print(f"\nCreating model: {config.MODEL_NAME}")
        model = get_model(config.MODEL_NAME, pretrained=True)
        model = model.to(device)
        criterion = nn.CrossEntropyLoss()
    
    # Optimizer
    optimizer = optim.Adam(
        model.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY
    )
    scheduler = ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3, verbose=True
    )
    
    # Training history
    history = {
        'epoch': [],
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    if config.USE_DISTILLATION:
        history['train_hard_loss'] = []
        history['train_soft_loss'] = []
    
    # Early stopping
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None
    
    print(f"\nStarting training for {config.NUM_EPOCHS} epochs...")
    print("-" * 60)
    
    for epoch in range(config.NUM_EPOCHS):
        print(f"\nEpoch {epoch + 1}/{config.NUM_EPOCHS}")
        
        # Train
        if config.USE_DISTILLATION:
            train_loss, train_acc, train_hard_loss, train_soft_loss = train_epoch(
                model, train_loader, criterion, optimizer, device, 
                teacher_model, distill_criterion
            )
        else:
            train_loss, train_acc = train_epoch(
                model, train_loader, criterion, optimizer, device
            )
        
        # Validate
        val_loss, val_acc = validate(
            model, val_loader, criterion, device
        )
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        # Save history
        history['epoch'].append(epoch + 1)
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        if config.USE_DISTILLATION:
            history['train_hard_loss'].append(train_hard_loss)
            history['train_soft_loss'].append(train_soft_loss)
        
        # Print epoch summary
        if config.USE_DISTILLATION:
            print(f"Train Loss: {train_loss:.4f} (Hard: {train_hard_loss:.4f}, Soft: {train_soft_loss:.4f}), Train Acc: {train_acc:.2f}%")
        else:
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        
        # Early stopping and model checkpointing
        if val_loss < best_val_loss - config.EARLY_STOPPING_MIN_DELTA:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = model.state_dict().copy()
            torch.save(best_model_state, config.BEST_MODEL_PATH)
            print(f"✓ New best model saved! (Val Loss: {val_loss:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= config.EARLY_STOPPING_PATIENCE:
                print(f"\nEarly stopping triggered after {epoch + 1} epochs")
                break
    
    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"\nBest model loaded (Val Loss: {best_val_loss:.4f})")
    
    # Save final model
    torch.save(model.state_dict(), config.FINAL_MODEL_PATH)
    print(f"Final model saved to {config.FINAL_MODEL_PATH}")
    
    # Save training history
    df_history = pd.DataFrame(history)
    df_history.to_csv(config.TRAINING_HISTORY_PATH, index=False)
    print(f"Training history saved to {config.TRAINING_HISTORY_PATH}")
    
    print("\n" + "=" * 60)
    print("Training completed!")
    print("=" * 60)
    
    return model, history


if __name__ == "__main__":
    train()

