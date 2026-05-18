
import os
import time
import copy
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
import matplotlib.pyplot as plt
import numpy as np

# Agrandir la taille par défaut des figures
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 12

from torchvision.datasets import ImageFolder
from PIL import Image

# La classe Dataset doit être globale pour le multiprocessing Windows
class BloodDataset(Dataset):
    def __init__(self, file_list, labels, transform=None):
        self.file_list = file_list
        self.labels = labels
        self.transform = transform
    def __len__(self): return len(self.file_list)
    def __getitem__(self, idx):
        img_path = self.file_list[idx]
        label = self.labels[idx]
        try:
            img = Image.open(img_path).convert("RGB")
            if self.transform:
                img = self.transform(img)
            return img, label
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            return torch.zeros((3, 224, 224)), label

def train_model():
    # Configuration
    data_dir = os.path.join(os.path.dirname(__file__), "redimensionnement", "Original_images_240x306")
    
    anemic_dir = os.path.join(data_dir, "anemic")
    healthy_dir = os.path.join(data_dir, "healthy")
    
    print(f"[INFO] Répertoires cibles :")
    print(f" - Healthy: {healthy_dir}")
    print(f" - Anemic : {anemic_dir}")
    
    if not os.path.exists(anemic_dir) or not os.path.exists(healthy_dir):
        print("[ERREUR] Répertoires introuvables !")
        return

    # Hyperparamètres Optimisés
    BATCH_SIZE = 16
    NUM_EPOCHS = 30         
    LEARNING_RATE = 0.0005  
    PATIENCE = 8            
    WEIGHT_DECAY = 1e-4     # L2 Regularization
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Utilisé device : {DEVICE}")

    # Transformations : Data Augmentation pour Train, Normalisation pour Val
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.8, 1.0)), # Zoom léger
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    # Lister les fichiers
    all_files = []
    all_labels = [] # 0: Healthy, 1: Anemic
    
    for f in os.listdir(healthy_dir):
        if f.lower().endswith(('.png', '.jpg', '.jpeg')):
            all_files.append(os.path.join(healthy_dir, f))
            all_labels.append(0)

    for f in os.listdir(anemic_dir):
        if f.lower().endswith(('.png', '.jpg', '.jpeg')):
            all_files.append(os.path.join(anemic_dir, f))
            all_labels.append(1)
            
    print(f"[INFO] Total images: {len(all_files)} (Healthy: {all_labels.count(0)}, Anemic: {all_labels.count(1)})")
    
    if len(all_files) == 0:
        return

    # Split MANUEL (Indispensable pour séparer les transforms)
    # Shuffle
    temp = list(zip(all_files, all_labels))
    random.seed(42)
    random.shuffle(temp)
    all_files, all_labels = zip(*temp)
    all_files, all_labels = list(all_files), list(all_labels)

    split_idx = int(0.8 * len(all_files))
    
    train_files = all_files[:split_idx]
    train_labels = all_labels[:split_idx]
    
    val_files = all_files[split_idx:]
    val_labels = all_labels[split_idx:]
    
    # Création des Datasets distincts
    image_datasets = {
        'train': BloodDataset(train_files, train_labels, transform=data_transforms['train']),
        'val': BloodDataset(val_files, val_labels, transform=data_transforms['val'])
    }
    
    print(f"[INFO] Données Train : {len(image_datasets['train'])} | Données Val : {len(image_datasets['val'])}")

    dataloaders = {
        'train': DataLoader(image_datasets['train'], batch_size=BATCH_SIZE, shuffle=True, num_workers=2),
        'val': DataLoader(image_datasets['val'], batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    }

    # Modèle : ResNet18 Pretrained
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    
    # FREEZING : Geler le début du réseau
    for name, param in model.named_parameters():
        if "layer1" in name or "layer2" in name or "conv1" in name or "bn1" in name:
            param.requires_grad = False
        else:
            param.requires_grad = True 
            
    # Modifier la tête avec Dropout
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.5), # Regularization forte
        nn.Linear(num_ftrs, 2)
    )
    
    model = model.to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    # Optimizer avec Weight Decay (L2 Regularization)
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), 
                           lr=LEARNING_RATE, 
                           weight_decay=WEIGHT_DECAY)
   
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3)

    # Training Loop
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    best_loss = float('inf')
    epochs_no_improve = 0
    
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}

    print("[INFO] Démarrage de l'entraînement optimisé...")
    start_time = time.time()

    for epoch in range(NUM_EPOCHS):
        print(f'Epoch {epoch+1}/{NUM_EPOCHS}')
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels_batch in dataloaders[phase]:
                inputs = inputs.to(DEVICE)
                labels_batch = labels_batch.to(DEVICE)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels_batch)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels_batch.data)

            epoch_loss = running_loss / len(image_datasets[phase])
            epoch_acc = running_corrects.double() / len(image_datasets[phase])

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            history[f'{phase}_loss'].append(epoch_loss)
            history[f'{phase}_acc'].append(epoch_acc.item())

            if phase == 'val':
                scheduler.step(epoch_loss)

                if epoch_loss < best_loss:
                    best_loss = epoch_loss
                    best_model_wts = copy.deepcopy(model.state_dict())
                    epochs_no_improve = 0
                    torch.save(model.state_dict(), 'best_classifier.pth')
                    print("  -> Modèle sauvegardé (Best Loss)")
                else:
                    epochs_no_improve += 1
                
                if epoch_acc > best_acc:
                    best_acc = epoch_acc

        print()
        if epochs_no_improve >= PATIENCE:
            print(f"Early Stopping déclenché après {epoch+1} époques.")
            break

    time_elapsed = time.time() - start_time
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best Val Acc: {best_acc:.4f} | Best Val Loss: {best_loss:.4f}')

    model.load_state_dict(best_model_wts)
    plot_history(history)
    return model

def plot_history(history):
    acc = history['train_acc']
    val_acc = history['val_acc']
    loss = history['train_loss']
    val_loss = history['val_loss']
    epochs_range = range(1, len(acc) + 1)

    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Acc')
    plt.plot(epochs_range, val_acc, label='Validation Acc')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right')
    plt.title('Accuracy')
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend(loc='upper right')
    plt.title('Loss')
    plt.grid(True)
    
    plt.savefig('training_curves_optimized.png')
    print("[INFO] Courbes sauvegardées sous 'training_curves_optimized.png'")

if __name__ == '__main__':
    train_model()
