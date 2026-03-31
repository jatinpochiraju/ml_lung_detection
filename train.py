"""
Chest X-Ray Classification Model
Classifies X-ray images into 4 categories:
  - COVID-19
  - Normal (Healthy)
  - Pneumonia
  - Tuberculosis (TB)

Uses transfer learning with ResNet18 or DenseNet121 backbone (PyTorch)
Usage:
  python train.py                        # Train ResNet18 (default)
  python train.py --model densenet121    # Train DenseNet121 for ensemble
"""

import os
import sys
import time
import copy
import json
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, WeightedRandomSampler
import torchvision
from torchvision import datasets, models, transforms
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm

# ============================================================
# CONFIGURATION
# ============================================================
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_DIR, "data")
OUTPUT_DIR = os.path.join(PROJECT_DIR, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Class names (matching folder names, mapped to clean labels)
CLASS_NAMES = ['COVID19', 'NORMAL', 'PNEUMONIA', 'TURBERCULOSIS']
DISPLAY_NAMES = ['COVID-19', 'Healthy', 'Pneumonia', 'Tuberculosis']

# Training hyperparameters
IMG_SIZE = 224
BATCH_SIZE = 32
NUM_EPOCHS = 20
LEARNING_RATE = 0.001
STEP_SIZE = 7       # LR scheduler step
GAMMA = 0.1         # LR scheduler gamma
NUM_WORKERS = 0     # 0 is safest on Windows (avoids worker crashes)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PIN_MEMORY = torch.cuda.is_available()  # Only pin memory if GPU is available
USE_AMP = torch.cuda.is_available()     # Mixed precision for faster GPU training & lower VRAM

# GPU optimizations
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True   # Auto-tune convolution algorithms
    torch.backends.cuda.matmul.allow_tf32 = True  # TF32 for Ampere GPUs
    torch.backends.cudnn.allow_tf32 = True
    gpu_name = torch.cuda.get_device_name(0)
    gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"GPU: {gpu_name} ({gpu_mem:.1f} GB VRAM)")
    print(f"Mixed Precision (AMP): {'Enabled' if USE_AMP else 'Disabled'}")

print(f"Using device: {DEVICE}")
print(f"Data directory: {DATA_DIR}")

# ============================================================
# DATA TRANSFORMS & LOADING
# ============================================================
def get_data_transforms():
    """Define data augmentation and normalization transforms."""
    # ImageNet normalization values (used by pretrained ResNet)
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize((IMG_SIZE + 32, IMG_SIZE + 32)),
            transforms.RandomCrop(IMG_SIZE),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ]),
        'val': transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ]),
        'test': transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ]),
    }
    return data_transforms


def load_datasets(data_transforms):
    """Load image datasets from directory structure."""
    image_datasets = {}
    dataloaders = {}
    dataset_sizes = {}
    
    for phase in ['train', 'val', 'test']:
        phase_dir = os.path.join(DATA_DIR, phase)
        if os.path.exists(phase_dir):
            image_datasets[phase] = datasets.ImageFolder(
                phase_dir, data_transforms[phase]
            )
            
            # For training, use weighted sampler to handle class imbalance
            if phase == 'train':
                targets = image_datasets[phase].targets
                class_counts = np.bincount(targets)
                class_weights = 1.0 / class_counts
                sample_weights = class_weights[targets]
                sampler = WeightedRandomSampler(
                    weights=sample_weights,
                    num_samples=len(sample_weights),
                    replacement=True
                )
                dataloaders[phase] = DataLoader(
                    image_datasets[phase],
                    batch_size=BATCH_SIZE,
                    sampler=sampler,
                    num_workers=NUM_WORKERS,
                    pin_memory=PIN_MEMORY
                )
            else:
                dataloaders[phase] = DataLoader(
                    image_datasets[phase],
                    batch_size=BATCH_SIZE,
                    shuffle=False,
                    num_workers=NUM_WORKERS,
                    pin_memory=PIN_MEMORY
                )
            
            dataset_sizes[phase] = len(image_datasets[phase])
            print(f"  {phase}: {dataset_sizes[phase]} images")
        else:
            print(f"  WARNING: {phase_dir} not found!")
    
    class_names = image_datasets['train'].classes
    print(f"\nClasses found: {class_names}")
    print(f"Class-to-index mapping: {image_datasets['train'].class_to_idx}")
    
    # Print class distribution
    train_targets = image_datasets['train'].targets
    class_counts = np.bincount(train_targets)
    print("\nTraining class distribution:")
    for i, (name, count) in enumerate(zip(class_names, class_counts)):
        print(f"  {name}: {count} images ({100*count/len(train_targets):.1f}%)")
    
    return image_datasets, dataloaders, dataset_sizes, class_names


# ============================================================
# MODEL DEFINITION
# ============================================================
def create_model(num_classes, architecture='resnet18'):
    """Create a model with transfer learning.
    Supported architectures: resnet18, densenet121
    """
    if architecture == 'densenet121':
        print("\nLoading pretrained DenseNet121...")
        model = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)

        # Freeze early layers
        for param in model.parameters():
            param.requires_grad = False

        # Unfreeze denseblock3, denseblock4, transition3, norm5, and classifier
        for name, param in model.named_parameters():
            if any(k in name for k in ['denseblock3', 'denseblock4', 'transition3', 'norm5', 'classifier']):
                param.requires_grad = True

        # Replace classifier
        num_ftrs = model.classifier.in_features
        model.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_ftrs, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
    else:
        # Default: ResNet18
        print("\nLoading pretrained ResNet18...")
        model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

        # Freeze early layers
        for param in model.parameters():
            param.requires_grad = False

        # Unfreeze layer3, layer4, and fc for fine-tuning
        for name, param in model.named_parameters():
            if 'layer3' in name or 'layer4' in name or 'fc' in name:
                param.requires_grad = True

        # Replace the final fully connected layer
        num_ftrs = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_ftrs, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    model = model.to(DEVICE)

    # Count trainable parameters
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Model: {architecture} | Parameters: {total:,} total, {trainable:,} trainable")

    return model


# ============================================================
# TRAINING LOOP
# ============================================================
def train_model(model, dataloaders, dataset_sizes, num_epochs=NUM_EPOCHS):
    """Train the model with validation monitoring."""
    
    # Loss function with label smoothing for better generalization
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    
    # Optimizer - only optimize trainable parameters
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LEARNING_RATE,
        weight_decay=1e-4
    )
    
    # Learning rate scheduler
    scheduler = lr_scheduler.StepLR(optimizer, step_size=STEP_SIZE, gamma=GAMMA)
    
    # Mixed precision scaler for GPU training (reduces VRAM, speeds up training)
    scaler = torch.amp.GradScaler('cuda', enabled=USE_AMP)
    
    # Training history
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': []
    }
    
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    best_epoch = 0
    start_time = time.time()
    
    print(f"\n{'='*60}")
    print(f"Starting Training - {num_epochs} epochs")
    print(f"{'='*60}\n")
    
    for epoch in range(num_epochs):
        epoch_start = time.time()
        print(f"Epoch {epoch+1}/{num_epochs}")
        print("-" * 40)
        
        for phase in ['train', 'val']:
            if phase not in dataloaders:
                continue
                
            if phase == 'train':
                model.train()
            else:
                model.eval()
            
            running_loss = 0.0
            running_corrects = 0
            
            pbar = tqdm(dataloaders[phase], desc=f"  {phase}", leave=False)
            for inputs, labels in pbar:
                inputs = inputs.to(DEVICE, non_blocking=True)
                labels = labels.to(DEVICE, non_blocking=True)
                
                optimizer.zero_grad(set_to_none=True)  # More efficient than zero_grad()
                
                # Mixed precision forward pass
                with torch.amp.autocast('cuda', enabled=USE_AMP):
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)
                
                if phase == 'train':
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                
                pbar.set_postfix({'loss': f'{loss.item():.4f}'})
            
            if phase == 'train':
                scheduler.step()
            
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.float() / dataset_sizes[phase]
            
            history[f'{phase}_loss'].append(epoch_loss)
            history[f'{phase}_acc'].append(epoch_acc.item())
            
            print(f"  {phase:5s} - Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.4f}")
            
            # Save best model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_epoch = epoch + 1
                best_model_wts = copy.deepcopy(model.state_dict())
                print(f"  >>> New best model! Val Acc: {best_acc:.4f}")
        
        epoch_time = time.time() - epoch_start
        print(f"  Time: {epoch_time:.1f}s\n")
    
    total_time = time.time() - start_time
    print(f"{'='*60}")
    print(f"Training complete in {total_time//60:.0f}m {total_time%60:.0f}s")
    print(f"Best val accuracy: {best_acc:.4f} at epoch {best_epoch}")
    print(f"{'='*60}\n")
    
    # Load best model weights
    model.load_state_dict(best_model_wts)
    return model, history


# ============================================================
# EVALUATION
# ============================================================
def evaluate_model(model, dataloader, dataset_size, class_names):
    """Evaluate model on test set with detailed metrics."""
    model.eval()
    
    all_preds = []
    all_labels = []
    running_corrects = 0
    
    print("Evaluating on test set...")
    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc="Testing"):
            inputs = inputs.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)
            
            with torch.amp.autocast('cuda', enabled=USE_AMP):
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
            
            running_corrects += torch.sum(preds == labels.data)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    test_acc = running_corrects.float() / dataset_size
    print(f"\nTest Accuracy: {test_acc:.4f} ({running_corrects}/{dataset_size})")
    
    # Classification Report
    display_names_mapped = [DISPLAY_NAMES[CLASS_NAMES.index(c)] if c in CLASS_NAMES else c for c in class_names]
    report = classification_report(all_labels, all_preds, target_names=display_names_mapped)
    print(f"\nClassification Report:\n{report}")
    
    # Save report
    with open(os.path.join(OUTPUT_DIR, "classification_report.txt"), 'w') as f:
        f.write(f"Test Accuracy: {test_acc:.4f}\n\n")
        f.write(report)
    
    # Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    
    return all_preds, all_labels, cm, display_names_mapped


# ============================================================
# PLOTTING
# ============================================================
def plot_training_history(history):
    """Plot training and validation loss/accuracy curves."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Loss
    ax1.plot(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=2)
    ax1.plot(epochs, history['val_loss'], 'r-', label='Val Loss', linewidth=2)
    ax1.set_title('Training & Validation Loss', fontsize=14)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Accuracy
    ax2.plot(epochs, history['train_acc'], 'b-', label='Train Acc', linewidth=2)
    ax2.plot(epochs, history['val_acc'], 'r-', label='Val Acc', linewidth=2)
    ax2.set_title('Training & Validation Accuracy', fontsize=14)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'training_history.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: training_history.png")


def plot_confusion_matrix(cm, class_names):
    """Plot confusion matrix heatmap."""
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_title('Confusion Matrix', fontsize=14)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'confusion_matrix.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: confusion_matrix.png")


def plot_sample_predictions(model, dataloader, class_names):
    """Show sample predictions on test images."""
    model.eval()
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    
    inputs, labels = next(iter(dataloader))
    inputs_device = inputs.to(DEVICE)
    
    with torch.no_grad():
        outputs = model(inputs_device)
        probs = torch.nn.functional.softmax(outputs, dim=1)
        _, preds = torch.max(outputs, 1)
    
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    for idx, ax in enumerate(axes.flat):
        if idx >= len(inputs):
            break
        img = inputs[idx].numpy().transpose((1, 2, 0))
        img = std * img + mean
        img = np.clip(img, 0, 1)
        
        pred_idx = preds[idx].item()
        true_idx = labels[idx].item()
        confidence = probs[idx][pred_idx].item()
        
        display_pred = DISPLAY_NAMES[CLASS_NAMES.index(class_names[pred_idx])] if class_names[pred_idx] in CLASS_NAMES else class_names[pred_idx]
        display_true = DISPLAY_NAMES[CLASS_NAMES.index(class_names[true_idx])] if class_names[true_idx] in CLASS_NAMES else class_names[true_idx]
        
        color = 'green' if pred_idx == true_idx else 'red'
        ax.imshow(img)
        ax.set_title(f"Pred: {display_pred} ({confidence:.1%})\nTrue: {display_true}",
                     color=color, fontsize=10)
        ax.axis('off')
    
    plt.suptitle('Sample Predictions on Test Set', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'sample_predictions.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: sample_predictions.png")


# ============================================================
# MAIN
# ============================================================
def main():
    parser = argparse.ArgumentParser(description='Train Chest X-Ray Classification Model')
    parser.add_argument('--model', type=str, default='resnet18',
                        choices=['resnet18', 'densenet121'],
                        help='Model architecture (default: resnet18)')
    parser.add_argument('--epochs', type=int, default=NUM_EPOCHS,
                        help=f'Number of training epochs (default: {NUM_EPOCHS})')
    args = parser.parse_args()

    architecture = args.model
    num_epochs = args.epochs

    # Output filename based on architecture
    if architecture == 'resnet18':
        model_filename = 'chest_xray_model.pth'
    else:
        model_filename = f'chest_xray_{architecture}.pth'

    print("=" * 60)
    print(f"  Chest X-Ray Classification Model ({architecture.upper()})")
    print("  COVID-19 | Healthy | Pneumonia | Tuberculosis")
    print("=" * 60)
    print()
    
    # 1. Load data
    print("Loading datasets...")
    data_transforms = get_data_transforms()
    image_datasets, dataloaders, dataset_sizes, class_names = load_datasets(data_transforms)
    
    # 2. Create model
    num_classes = len(class_names)
    model = create_model(num_classes, architecture=architecture)
    
    # 3. Train
    model, history = train_model(model, dataloaders, dataset_sizes, num_epochs)
    
    # 4. Save model
    model_path = os.path.join(OUTPUT_DIR, model_filename)
    torch.save({
        'model_state_dict': model.state_dict(),
        'class_names': class_names,
        'display_names': DISPLAY_NAMES,
        'num_classes': num_classes,
        'img_size': IMG_SIZE,
        'architecture': architecture,
    }, model_path)
    print(f"Model saved to: {model_path}")
    
    # 5. Evaluate on test set
    if 'test' in dataloaders:
        all_preds, all_labels, cm, display_names = evaluate_model(
            model, dataloaders['test'], dataset_sizes['test'], class_names
        )
        
        # 6. Generate plots
        plot_training_history(history)
        plot_confusion_matrix(cm, display_names)
        plot_sample_predictions(model, dataloaders['test'], class_names)
    else:
        plot_training_history(history)
        print("No test set found - skipping test evaluation.")
    
    # 7. Save training history
    history_path = os.path.join(OUTPUT_DIR, "training_history.json")
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"Training history saved to: {history_path}")
    
    print("\n" + "=" * 60)
    print("  TRAINING COMPLETE!")
    print(f"  All outputs saved to: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
