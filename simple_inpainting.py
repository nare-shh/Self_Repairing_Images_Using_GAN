"""
Simple Image Inpainting GAN
==========================
A straightforward GAN that fills in missing parts of images.

Usage on Google Colab:
    1. Go to colab.research.google.com
    2. Runtime -> Change runtime type -> GPU
    3. Upload this file and run: !python simple_inpainting.py

Author: Your Name
"""

import os
import random
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm

# ============================================
# CONFIGURATION
# ============================================

IMAGE_SIZE = 128  # Smaller = faster training
BATCH_SIZE = 8
EPOCHS = 30
LEARNING_RATE = 0.0002
DATA_DIR = './data'
SAVE_DIR = './output'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


# ============================================
# SIMPLE U-NET GENERATOR
# ============================================

class UNetGenerator(nn.Module):
    """
    Simple U-Net for image inpainting.
    Input: masked image (3 channels) + mask (1 channel) = 4 channels
    Output: completed image (3 channels)
    """
    
    def __init__(self):
        super().__init__()
        
        # Encoder (downsampling)
        self.enc1 = self._block(4, 64, normalize=False)    # 128 -> 64
        self.enc2 = self._block(64, 128)                    # 64 -> 32
        self.enc3 = self._block(128, 256)                   # 32 -> 16
        self.enc4 = self._block(256, 512)                   # 16 -> 8
        self.enc5 = self._block(512, 512)                   # 8 -> 4
        
        # Decoder (upsampling with skip connections)
        self.dec5 = self._upblock(512, 512)                 # 4 -> 8
        self.dec4 = self._upblock(1024, 256)                # 8 -> 16 (512 + 512 from skip)
        self.dec3 = self._upblock(512, 128)                 # 16 -> 32 (256 + 256 from skip)
        self.dec2 = self._upblock(256, 64)                  # 32 -> 64 (128 + 128 from skip)
        self.dec1 = self._upblock(128, 64)                  # 64 -> 128 (64 + 64 from skip)
        
        # Output layer
        self.final = nn.Sequential(
            nn.Conv2d(64, 3, 3, padding=1),
            nn.Tanh()
        )
    
    def _block(self, in_ch, out_ch, normalize=True):
        layers = [nn.Conv2d(in_ch, out_ch, 4, 2, 1)]
        if normalize:
            layers.append(nn.BatchNorm2d(out_ch))
        layers.append(nn.LeakyReLU(0.2))
        return nn.Sequential(*layers)
    
    def _upblock(self, in_ch, out_ch):
        return nn.Sequential(
            nn.ConvTranspose2d(in_ch, out_ch, 4, 2, 1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU()
        )
    
    def forward(self, x, mask):
        # Combine input: masked image + mask
        inp = torch.cat([x * mask, mask], dim=1)
        
        # Encoder
        e1 = self.enc1(inp)     # 64 channels
        e2 = self.enc2(e1)      # 128 channels
        e3 = self.enc3(e2)      # 256 channels
        e4 = self.enc4(e3)      # 512 channels
        e5 = self.enc5(e4)      # 512 channels
        
        # Decoder with skip connections
        d5 = self.dec5(e5)
        d4 = self.dec4(torch.cat([d5, e4], 1))
        d3 = self.dec3(torch.cat([d4, e3], 1))
        d2 = self.dec2(torch.cat([d3, e2], 1))
        d1 = self.dec1(torch.cat([d2, e1], 1))
        
        # Output
        out = self.final(d1)
        
        # Blend: keep original where mask=1, use generated where mask=0
        result = x * mask + out * (1 - mask)
        return result


# ============================================
# SIMPLE DISCRIMINATOR
# ============================================

class Discriminator(nn.Module):
    """Simple PatchGAN discriminator."""
    
    def __init__(self):
        super().__init__()
        
        self.model = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1),
            nn.LeakyReLU(0.2),
            
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            
            nn.Conv2d(256, 1, 4, 1, 1)
        )
    
    def forward(self, x):
        return self.model(x)


# ============================================
# DATASET
# ============================================

def create_random_mask(size):
    """Create a random rectangular mask."""
    mask = np.ones((size, size), dtype=np.float32)
    
    # Random rectangle
    h = random.randint(size // 4, size // 2)
    w = random.randint(size // 4, size // 2)
    y = random.randint(0, size - h)
    x = random.randint(0, size - w)
    mask[y:y+h, x:x+w] = 0
    
    return mask


def create_demo_images(path, count=200):
    """Create simple demo images for training."""
    os.makedirs(path, exist_ok=True)
    
    if len(os.listdir(path)) >= count:
        print(f"Demo images already exist in {path}")
        return
    
    print(f"Creating {count} demo images...")
    for i in tqdm(range(count)):
        # Random gradient background
        img = Image.new('RGB', (IMAGE_SIZE, IMAGE_SIZE))
        pixels = img.load()
        
        r0, g0, b0 = random.randint(50, 200), random.randint(50, 200), random.randint(50, 200)
        
        for y in range(IMAGE_SIZE):
            for x in range(IMAGE_SIZE):
                r = min(255, r0 + (x + y) // 4 + random.randint(-10, 10))
                g = min(255, g0 + (x - y) // 4 + random.randint(-10, 10))
                b = min(255, b0 + (y - x) // 4 + random.randint(-10, 10))
                pixels[x, y] = (r, g, b)
        
        # Add random shapes
        draw = ImageDraw.Draw(img)
        for _ in range(random.randint(2, 5)):
            x1, y1 = random.randint(0, IMAGE_SIZE-40), random.randint(0, IMAGE_SIZE-40)
            x2, y2 = x1 + random.randint(20, 60), y1 + random.randint(20, 60)
            color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
            draw.ellipse([x1, y1, x2, y2], fill=color)
        
        img = img.filter(ImageFilter.GaussianBlur(1))
        img.save(os.path.join(path, f'{i:04d}.png'))


class ImageDataset(Dataset):
    """Simple dataset that loads images and generates masks."""
    
    def __init__(self, folder):
        self.files = [os.path.join(folder, f) for f in os.listdir(folder) 
                      if f.endswith(('.png', '.jpg', '.jpeg'))]
        self.transform = transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize([0.5]*3, [0.5]*3)  # Scale to [-1, 1]
        ])
    
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        img = Image.open(self.files[idx]).convert('RGB')
        img = self.transform(img)
        mask = torch.from_numpy(create_random_mask(IMAGE_SIZE)).unsqueeze(0)
        return img, mask


# ============================================
# TRAINING
# ============================================

def train():
    print("=" * 50)
    print("SIMPLE IMAGE INPAINTING GAN")
    print("=" * 50)
    
    # Create directories
    os.makedirs(SAVE_DIR, exist_ok=True)
    
    # Create demo data if needed
    create_demo_images(DATA_DIR)
    
    # Load data
    dataset = ImageDataset(DATA_DIR)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    print(f"Dataset: {len(dataset)} images")
    
    # Create models
    G = UNetGenerator().to(device)
    D = Discriminator().to(device)
    
    print(f"Generator params: {sum(p.numel() for p in G.parameters()):,}")
    print(f"Discriminator params: {sum(p.numel() for p in D.parameters()):,}")
    
    # Optimizers
    opt_G = torch.optim.Adam(G.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))
    opt_D = torch.optim.Adam(D.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))
    
    # Loss function
    criterion = nn.BCEWithLogitsLoss()
    l1_loss = nn.L1Loss()
    
    print(f"\nTraining for {EPOCHS} epochs...")
    print("-" * 50)
    
    for epoch in range(EPOCHS):
        G.train()
        D.train()
        
        total_g_loss = 0
        total_d_loss = 0
        
        for real_img, mask in tqdm(loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
            real_img = real_img.to(device)
            mask = mask.to(device)
            batch = real_img.size(0)
            
            # Labels
            real_label = torch.ones(batch, 1, 14, 14).to(device)
            fake_label = torch.zeros(batch, 1, 14, 14).to(device)
            
            # =================
            # Train Discriminator
            # =================
            opt_D.zero_grad()
            
            # Real images
            d_real = D(real_img)
            loss_d_real = criterion(d_real, real_label)
            
            # Fake images
            fake_img = G(real_img, mask)
            d_fake = D(fake_img.detach())
            loss_d_fake = criterion(d_fake, fake_label)
            
            loss_D = (loss_d_real + loss_d_fake) / 2
            loss_D.backward()
            opt_D.step()
            
            # =================
            # Train Generator
            # =================
            opt_G.zero_grad()
            
            fake_img = G(real_img, mask)
            d_fake = D(fake_img)
            
            # Adversarial loss (fool discriminator)
            loss_adv = criterion(d_fake, real_label)
            
            # Reconstruction loss (match original in hole region)
            loss_rec = l1_loss(fake_img, real_img) * 100
            
            loss_G = loss_adv + loss_rec
            loss_G.backward()
            opt_G.step()
            
            total_g_loss += loss_G.item()
            total_d_loss += loss_D.item()
        
        # Print epoch stats
        avg_g = total_g_loss / len(loader)
        avg_d = total_d_loss / len(loader)
        print(f"  G Loss: {avg_g:.4f} | D Loss: {avg_d:.4f}")
        
        # Save sample every 5 epochs
        if (epoch + 1) % 5 == 0:
            save_samples(G, dataset, epoch + 1)
    
    # Save model
    torch.save(G.state_dict(), os.path.join(SAVE_DIR, 'generator.pth'))
    print(f"\nModel saved to {SAVE_DIR}/generator.pth")
    print("Training complete!")


def save_samples(G, dataset, epoch):
    """Save sample outputs."""
    G.eval()
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(3, 4, figsize=(12, 9))
    
    with torch.no_grad():
        for i in range(3):
            img, mask = dataset[random.randint(0, len(dataset)-1)]
            img = img.unsqueeze(0).to(device)
            mask = mask.unsqueeze(0).to(device)
            
            masked = img * mask
            output = G(img, mask)
            
            # Convert to displayable format
            def to_img(t):
                t = (t.squeeze().cpu().numpy().transpose(1, 2, 0) + 1) / 2
                return np.clip(t, 0, 1)
            
            axes[i, 0].imshow(to_img(img))
            axes[i, 0].set_title('Original')
            axes[i, 1].imshow(mask.squeeze().cpu(), cmap='gray')
            axes[i, 1].set_title('Mask')
            axes[i, 2].imshow(to_img(masked))
            axes[i, 2].set_title('Input')
            axes[i, 3].imshow(to_img(output))
            axes[i, 3].set_title('Output')
            
            for ax in axes[i]:
                ax.axis('off')
    
    plt.suptitle(f'Epoch {epoch}')
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, f'epoch_{epoch:03d}.png'))
    plt.close()
    print(f"  Saved sample: epoch_{epoch:03d}.png")


# ============================================
# INFERENCE
# ============================================

def repair_image(image_path, mask_path=None, output_path='repaired.png'):
    """
    Repair a damaged image.
    
    Args:
        image_path: Path to damaged image
        mask_path: Path to mask (white=keep, black=repair). If None, detects automatically.
        output_path: Where to save the result
    """
    # Load model
    G = UNetGenerator().to(device)
    G.load_state_dict(torch.load(os.path.join(SAVE_DIR, 'generator.pth'), map_location=device))
    G.eval()
    
    # Load and preprocess image
    img = Image.open(image_path).convert('RGB')
    original_size = img.size
    
    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])
    
    img_tensor = transform(img).unsqueeze(0).to(device)
    
    # Load or create mask
    if mask_path:
        mask = Image.open(mask_path).convert('L')
        mask = mask.resize((IMAGE_SIZE, IMAGE_SIZE))
        mask = np.array(mask) / 255.0
    else:
        # Auto-detect: treat near-white pixels as holes
        img_np = np.array(img.resize((IMAGE_SIZE, IMAGE_SIZE)))
        brightness = img_np.mean(axis=2)
        mask = (brightness < 250).astype(np.float32)
    
    mask_tensor = torch.from_numpy(mask).unsqueeze(0).unsqueeze(0).float().to(device)
    
    # Run inpainting
    with torch.no_grad():
        output = G(img_tensor, mask_tensor)
    
    # Convert back to image
    output_np = output.squeeze().cpu().numpy().transpose(1, 2, 0)
    output_np = ((output_np + 1) / 2 * 255).clip(0, 255).astype(np.uint8)
    output_img = Image.fromarray(output_np)
    output_img = output_img.resize(original_size, Image.LANCZOS)
    output_img.save(output_path)
    
    print(f"Repaired image saved to {output_path}")


# ============================================
# MAIN
# ============================================

if __name__ == '__main__':
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == 'repair':
        # Usage: python simple_inpainting.py repair image.jpg [mask.png] [output.png]
        img = sys.argv[2] if len(sys.argv) > 2 else 'test.jpg'
        mask = sys.argv[3] if len(sys.argv) > 3 else None
        out = sys.argv[4] if len(sys.argv) > 4 else 'repaired.png'
        repair_image(img, mask, out)
    else:
        train()
