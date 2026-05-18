import os
import shutil
from pathlib import Path
import random

# Définir les chemins
source_dir = Path("redimensionnement/Original_images_240x306")
train_dir = Path("data/train")
val_dir = Path("data/val")

# Créer les dossiers s'ils n'existent pas
train_dir.mkdir(parents=True, exist_ok=True)
val_dir.mkdir(parents=True, exist_ok=True)

# Collecter toutes les images (anemic et healthy)
all_images = []

# Dossiers sources avec images propres
source_folders = [
    source_dir / "anemic",
    source_dir / "healthy"
]

print("Collecte des images...")
for folder in source_folders:
    if folder.exists():
        images = list(folder.glob("*.png")) + list(folder.glob("*.jpg")) + list(folder.glob("*.jpeg"))
        all_images.extend(images)
        print(f"  {folder.name}: {len(images)} images")

print(f"\nTotal images trouvées: {len(all_images)}")

# Mélanger et diviser en train/val (80/20)
random.seed(42)
random.shuffle(all_images)

split_idx = int(0.8 * len(all_images))
train_images = all_images[:split_idx]
val_images = all_images[split_idx:]

print(f"\nDivision:")
print(f"  Train: {len(train_images)} images")
print(f"  Val: {len(val_images)} images")

# Copier les images
print("\nCopie des images...")
for img in train_images:
    shutil.copy2(img, train_dir / img.name)

for img in val_images:
    shutil.copy2(img, val_dir / img.name)

print("\n✓ Organisation terminée!")
print(f"  data/train: {len(list(train_dir.glob('*')))} images")
print(f"  data/val: {len(list(val_dir.glob('*')))} images")
