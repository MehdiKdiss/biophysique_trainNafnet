import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as T
import matplotlib.pyplot as plt
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim
import random
import torch.nn.functional as F
from tqdm import tqdm

# ==========================================================
# 0) REPRODUCTIBILITÉ
# ==========================================================
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ==========================================================
# 1) DATASET
# ==========================================================
class ImageDataset(Dataset):
    def __init__(self, root):
        self.images = []
        for f in os.listdir(root):
            if f.lower().endswith(('png', 'jpg', 'jpeg')):
                self.images.append(os.path.join(root, f))

        self.transform = T.Compose([
            T.Resize((256, 256)),
            T.ToTensor()
        ])

    def add_gaussian_noise(self, img, mean=0, std=0.05):
        noise = torch.randn_like(img) * std + mean
        return torch.clamp(img + noise, 0, 1)

    def __getitem__(self, idx):
        img = Image.open(self.images[idx]).convert("RGB")
        clean = self.transform(img)
        noisy = self.add_gaussian_noise(clean)
        return noisy, clean

    def __len__(self):
        return len(self.images)


# ==========================================================
# 2) TRUE NAFNet Architecture
# ==========================================================
class LayerNorm2d(nn.Module):
    """LayerNorm for 2D feature maps (channels-first)"""
    def __init__(self, channels, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))
        self.eps = eps
    
    def forward(self, x):
        # x: [B, C, H, W]
        mean = x.mean(dim=1, keepdim=True)
        var = x.var(dim=1, keepdim=True, unbiased=False)
        x = (x - mean) / torch.sqrt(var + self.eps)
        x = x * self.weight[None, :, None, None] + self.bias[None, :, None, None]
        return x


class SimpleGate(nn.Module):
    """SimpleGate: split channels and multiply (core NAFNet innovation)"""
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


class SimplifiedChannelAttention(nn.Module):
    """Simplified Channel Attention without nonlinearities"""
    def __init__(self, channels):
        super().__init__()
        self.fc = nn.Conv2d(channels, channels, 1, bias=True)
    
    def forward(self, x):
        y = F.adaptive_avg_pool2d(x, 1)
        y = self.fc(y)
        return x * y


class NAFBlock(nn.Module):
    """True NAFNet Block with SimpleGate and SCA"""
    def __init__(self, channels, dw_expand=2, ffn_expand=2):
        super().__init__()
        dw_channels = channels * dw_expand
        
        # First path: spatial mixing with depthwise conv
        self.norm1 = LayerNorm2d(channels)
        self.conv1 = nn.Conv2d(channels, dw_channels, 1)
        self.conv2 = nn.Conv2d(dw_channels, dw_channels, 3, padding=1, groups=dw_channels)
        self.sg = SimpleGate()
        self.conv3 = nn.Conv2d(dw_channels // 2, channels, 1)
        
        # Second path: channel mixing (FFN)
        self.norm2 = LayerNorm2d(channels)
        self.conv4 = nn.Conv2d(channels, ffn_expand * channels, 1)
        self.sg2 = SimpleGate()
        self.conv5 = nn.Conv2d(ffn_expand * channels // 2, channels, 1)
        
        # Simplified Channel Attention
        self.sca = SimplifiedChannelAttention(channels)
        
        # Learnable scaling parameters
        self.beta = nn.Parameter(torch.zeros((1, channels, 1, 1)))
        self.gamma = nn.Parameter(torch.zeros((1, channels, 1, 1)))
    
    def forward(self, x):
        # First path: spatial mixing
        shortcut = x
        x = self.norm1(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)
        x = self.conv3(x)
        x = self.sca(x)
        x = shortcut + x * self.beta
        
        # Second path: channel mixing (FFN)
        shortcut = x
        x = self.norm2(x)
        x = self.conv4(x)
        x = self.sg2(x)
        x = self.conv5(x)
        x = shortcut + x * self.gamma
        
        return x


class NAFNet(nn.Module):
    """True NAFNet with U-Net encoder-decoder architecture"""
    def __init__(self, img_channels=3, width=64, middle_blocks=12, 
                 enc_blocks=[2, 2, 4, 8], dec_blocks=[2, 2, 2, 2]):
        super().__init__()
        
        self.intro = nn.Conv2d(img_channels, width, 3, padding=1)
        
        # Encoder
        self.encoders = nn.ModuleList()
        self.downs = nn.ModuleList()
        chan = width
        for num in enc_blocks:
            self.encoders.append(nn.Sequential(*[NAFBlock(chan) for _ in range(num)]))
            self.downs.append(nn.Conv2d(chan, chan * 2, 2, stride=2))
            chan *= 2
        
        # Middle (bottleneck)
        self.middle = nn.Sequential(*[NAFBlock(chan) for _ in range(middle_blocks)])
        
        # Decoder
        self.ups = nn.ModuleList()
        self.decoders = nn.ModuleList()
        for num in dec_blocks:
            self.ups.append(nn.Sequential(
                nn.Conv2d(chan, chan * 2, 1),
                nn.PixelShuffle(2)
            ))
            chan //= 2
            self.decoders.append(nn.Sequential(*[NAFBlock(chan) for _ in range(num)]))
        
        self.ending = nn.Conv2d(width, img_channels, 3, padding=1)
        self.features = []
    
    def forward(self, x):
        self.features = []
        
        x = self.intro(x)
        
        # Encoder with skip connections
        encs = []
        for encoder, down in zip(self.encoders, self.downs):
            x = encoder(x)
            encs.append(x)
            x = down(x)
            self.features.append(x.detach().cpu())
        
        # Middle
        x = self.middle(x)
        
        # Decoder with skip connections
        for decoder, up, enc_skip in zip(self.decoders, self.ups, reversed(encs)):
            x = up(x)
            x = x + enc_skip  # Skip connection
            x = decoder(x)
        
        x = self.ending(x)
        return x


# ==========================================================
# 3) BATCH-LEVEL METRICS (PSNR, MSE, SSIM, Accuracy)
# ==========================================================
def batch_psnr(pred, target, max_val=1.0):
    """Calculate PSNR averaged over entire batch"""
    mse = F.mse_loss(pred, target, reduction='none')
    mse = mse.view(mse.size(0), -1).mean(dim=1)  # Per-image MSE
    psnr = 20 * torch.log10(max_val / torch.sqrt(mse + 1e-8))
    return psnr.mean()


def batch_mse(pred, target):
    """Calculate MSE averaged over entire batch"""
    return F.mse_loss(pred, target)


def batch_ssim(pred, target):
    """Calculate SSIM averaged over entire batch using skimage"""
    batch_size = pred.size(0)
    ssim_sum = 0.0
    for i in range(batch_size):
        p = pred[i].permute(1, 2, 0).detach().cpu().numpy()
        t = target[i].permute(1, 2, 0).detach().cpu().numpy()
        ssim_sum += compare_ssim(p, t, channel_axis=2, data_range=1.0)
    return ssim_sum / batch_size


def gradient_accuracy(pred, target):
    """Comparaison des gradients Sobel sur tout le batch"""
    Gx = torch.tensor([[1, 0, -1],
                       [2, 0, -2],
                       [1, 0, -1]], dtype=torch.float32).view(1,1,3,3).to(device)

    def sobel(x):
        x_gray = x.mean(dim=1, keepdim=True)
        return F.conv2d(x_gray, Gx, padding=1)

    g1 = sobel(pred)
    g2 = sobel(target)

    return 1 - torch.mean(torch.abs(g1 - g2))


def texture_accuracy(pred, target):
    """Loi de puissance des textures via statistiques sur tout le batch"""
    p = torch.var(pred, dim=[1,2,3])
    t = torch.var(target, dim=[1,2,3])
    return 1 - torch.mean(torch.abs(p - t) / (t + 1e-6))


def combined_accuracy(pred, target):
    """Accuracy combinée (gradient + texture) sur tout le batch"""
    return 0.5 * gradient_accuracy(pred, target) + 0.5 * texture_accuracy(pred, target)


# ==========================================================
# 4) TRAIN WITH MIXED PRECISION & BATCH METRICS
# ==========================================================
def train_model(model, loader, val_loader, epochs=50, lr=1e-4, patience=7):
    criterion = nn.L1Loss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scaler = torch.amp.GradScaler('cuda') if device.type == 'cuda' else None

    train_loss_history, val_loss_history = [], []
    psnr_list, mse_list, ssim_list, acc_list = [], [], [], []

    best_loss = float('inf')
    wait = 0

    for epoch in range(epochs):
        model.train()
        running_loss = 0

        for noisy, clean in tqdm(loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False):
            noisy, clean = noisy.to(device), clean.to(device)

            optimizer.zero_grad()
            
            # Mixed precision training
            if scaler is not None:
                with torch.amp.autocast('cuda'):
                    output = model(noisy)
                    loss = criterion(output, clean)
                
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # Gradient clipping
                scaler.step(optimizer)
                scaler.update()
            else:
                output = model(noisy)
                loss = criterion(output, clean)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            running_loss += loss.item()

        train_loss = running_loss / len(loader)
        train_loss_history.append(train_loss)

        # =============== Validation + Batch Metrics ===============
        model.eval()
        val_loss = 0
        psnr_epoch, mse_epoch, ssim_epoch, acc_epoch = 0, 0, 0, 0

        with torch.no_grad():
            for noisy, clean in val_loader:
                noisy, clean = noisy.to(device), clean.to(device)
                output = model(noisy)
                val_loss += criterion(output, clean).item()

                # ---- Batch-level metrics ----
                psnr_epoch += batch_psnr(output, clean).item()
                mse_epoch += batch_mse(output, clean).item()
                ssim_epoch += batch_ssim(output, clean)
                acc_epoch += combined_accuracy(output, clean).item()

        val_loss /= len(val_loader)
        val_loss_history.append(val_loss)
        psnr_list.append(psnr_epoch / len(val_loader))
        mse_list.append(mse_epoch / len(val_loader))
        ssim_list.append(ssim_epoch / len(val_loader))
        acc_list.append(acc_epoch / len(val_loader))

        print(f"Epoch {epoch+1}: Train={train_loss:.4f}, Val={val_loss:.4f}, "
              f"PSNR={psnr_list[-1]:.2f}, MSE={mse_list[-1]:.4f}, "
              f"SSIM={ssim_list[-1]:.3f}, ACC={acc_list[-1]:.3f}")

        # EARLY STOPPING
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), "best_nafnet.pth")
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                print("Early stopping activé !")
                break

    return train_loss_history, val_loss_history, psnr_list, mse_list, ssim_list, acc_list


# ==========================================================
# 5) VISUALISER LES FEATURE MAPS
# ==========================================================
def visualize_feature_maps(model, n=4):
    if len(model.features) == 0:
        print("Aucune feature map enregistrée !")
        return

    fmap = model.features[-1][0]  # dernière couche, première image
    fmap = fmap[:n*n].detach().numpy()

    plt.figure(figsize=(8, 8))
    for i in range(n*n):
        plt.subplot(n, n, i+1)
        plt.imshow(fmap[i], cmap="gray")
        plt.axis("off")
    plt.show()



if __name__ == '__main__':
    # ==========================================================
    # 6) CHARGEMENT DATASETS
    # ==========================================================
    train_ds = ImageDataset("data/train")
    val_ds   = ImageDataset("data/val")

    train_loader = DataLoader(train_ds, batch_size=4, shuffle=True, num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_ds, batch_size=4, num_workers=2, pin_memory=True)


    # ==========================================================
    # 7) TRAIN TRUE NAFNET
    # ==========================================================
    model = NAFNet(width=48, middle_blocks=8).to(device)

    print(f"Training on device: {device}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

    train_loss, val_loss, psnr_hist, mse_hist, ssim_hist, acc_hist = train_model(
        model, train_loader, val_loader, epochs=50
    )


    # ==========================================================
    # 8) COURBES DES MÉTRIQUES (PSNR, MSE, SSIM, Accuracy)
    # ==========================================================
    plt.figure(figsize=(12,8))
    plt.plot(train_loss, label="Train Loss")
    plt.plot(val_loss, label="Val Loss")
    plt.legend()
    plt.title("Training & Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(12,8))
    plt.plot(psnr_hist, label="PSNR", color='green')
    plt.title("PSNR (Peak Signal-to-Noise Ratio)")
    plt.xlabel("Epoch")
    plt.ylabel("PSNR (dB)")
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(12,8))
    plt.plot(mse_hist, label="MSE", color='red')
    plt.title("MSE (Mean Squared Error)")
    plt.xlabel("Epoch")
    plt.ylabel("MSE")
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(12,8))
    plt.plot(ssim_hist, label="SSIM", color='blue')
    plt.title("SSIM (Structural Similarity Index)")
    plt.xlabel("Epoch")
    plt.ylabel("SSIM")
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(12,8))
    plt.plot(acc_hist, label="Texture/Gradient Accuracy", color='purple')
    plt.title("Combined Accuracy (Gradient + Texture)")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.grid(True)
    plt.show()

    # AFFICHER FEATURES
    visualize_feature_maps(model, n=4)

