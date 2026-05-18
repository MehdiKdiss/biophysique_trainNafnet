# Biophysics Blood Image Analysis with NAFNet

This project contains a small blood-smear image processing pipeline for educational/research use:

- Image restoration with a PyTorch NAFNet-style model.
- Binary anemia classification with a fine-tuned ResNet18.
- RBC/WBC segmentation and morphology extraction with Cellpose.
- A CustomTkinter GUI that combines restoration, segmentation, and analysis.
- Classical filtering and architecture/result diagrams for comparison and reporting.

> This project is not a medical device and must not be used for clinical diagnosis.

## Repository Structure

| Path | Purpose |
| --- | --- |
| `trainingNafnet.py` | Trains the NAFNet restoration model and saves `best_nafnet.pth`. |
| `train_classifier.py` | Trains a ResNet18 anemia classifier and saves `best_classifier.pth`. |
| `gui_nafnet.py` | Main desktop GUI for restoration and anemia analysis. |
| `anemia_segmentation.py` | Cellpose segmentation, RBC feature extraction, and anemia interpretation helpers. |
| `test_nafnet.py` | Standalone architecture and metric sanity checks. |
| `test_classifier_gui.py` | Simple GUI for testing the classifier checkpoint. |
| `classic_filter_demo.py` | Classical bilateral-filter plus sharpening baseline. |
| `organize_data.py` | Creates `data/train` and `data/val` splits from the local resized dataset. |
| `redimensionnement/resize_images.py` | Resizes the original local dataset. |
| `*.png` at the project root | Kept result images and diagrams, including training curves. |

## What Is Not Committed

The repository is prepared for GitHub by ignoring files that are too large, private, or generated:

- Local datasets under `redimensionnement/` and `data/`.
- Model checkpoints such as `best_nafnet.pth` and `best_classifier.pth`.
- Screenshots/captures.
- Local report documents such as `.docx`.
- Generated CSV outputs such as `rbc_features.csv`.

If you want to share trained weights, use GitHub Releases or Git LFS. The NAFNet checkpoint is larger than GitHub's normal file-size limit.

## Installation

Create and activate a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

For GPU training, install the PyTorch build that matches your CUDA version before running the project.

## Expected Local Files

Training and inference expect local files that are intentionally ignored by Git:

```text
best_nafnet.pth
best_classifier.pth
data/train/
data/val/
redimensionnement/Original_images_240x306/anemic/
redimensionnement/Original_images_240x306/healthy/
```

Use `organize_data.py` to create `data/train` and `data/val` from the local resized dataset.

## Common Commands

Check the NAFNet architecture:

```powershell
python test_nafnet.py
```

Prepare train/validation folders:

```powershell
python organize_data.py
```

Train the restoration model:

```powershell
python trainingNafnet.py
```

Train the anemia classifier:

```powershell
python train_classifier.py
```

Run the full GUI:

```powershell
python gui_nafnet.py
```

Run the classifier-only GUI:

```powershell
python test_classifier_gui.py
```

Run the classical filter baseline:

```powershell
python classic_filter_demo.py path\to\image.png
```

## Publish on GitHub

From the project folder:

```powershell
git init
git add .
git commit -m "Initial public release"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPOSITORY.git
git push -u origin main
```

Before pushing, check what will be committed:

```powershell
git status
git ls-files
```

If you need to publish checkpoints, do not add `.pth` files directly to normal Git history. Prefer a GitHub Release or configure Git LFS:

```powershell
git lfs install
git lfs track "*.pth"
git add .gitattributes
```
