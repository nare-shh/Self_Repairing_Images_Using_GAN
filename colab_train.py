"""
Google Colab Training Script

Copy this entire file content into a Colab notebook cell and run it.
This is a self-contained script that:
1. Sets up the environment
2. Creates/downloads the dataset
3. Trains the model
4. Saves the results

Just run this cell and wait!
"""

# ============================================
# SETUP
# ============================================
print("="*50)
print("INPAINTING GAN - COLAB TRAINING")
print("="*50)

import subprocess
import sys

# Install dependencies
print("\n[1/5] Installing dependencies...")
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", 
                       "torch", "torchvision", "pillow", "numpy", "tqdm", "matplotlib", "gdown"])

import os
import random
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.spectral_norm as spectral_norm
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torchvision import transforms, models
from torchvision.utils import make_grid
from tqdm import tqdm
import matplotlib.pyplot as plt

# Check GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\n[2/5] Device: {device}")
if device.type == 'cuda':
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
else:
    print("WARNING: No GPU detected! Training will be very slow.")

# ============================================
# CONFIGURATION
# ============================================
class Config:
    # Data
    data_dir = './training_data'
    image_size = 256
    mask_type = 'mixed'  # rectangular, irregular, center, mixed
    
    # Training
    epochs = 50  # Increase for better results
    batch_size = 4  # Reduce if OOM
    lr_g = 1e-4
    lr_d = 1e-4
    beta1 = 0.5
    beta2 = 0.999
    
    # Loss weights
    l1_weight = 1.0
    adversarial_weight = 0.1
    perceptual_weight = 0.1
    style_weight = 250.0
    
    # Saving
    save_dir = './checkpoints'
    save_freq = 10  # Save every N epochs
    vis_freq = 50   # Visualize every N steps
    
    # Demo
    use_demo_data = True  # Set to False and provide your own data
    demo_size = 200  # Number of demo images

config = Config()

# Create directories
os.makedirs(config.save_dir, exist_ok=True)
os.makedirs(os.path.join(config.save_dir, 'visualizations'), exist_ok=True)

# ============================================
# MODEL DEFINITIONS
# ============================================

class PartialConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)
        self.mask_conv = nn.Conv2d(1, 1, kernel_size, stride, padding, bias=False)
        nn.init.constant_(self.mask_conv.weight, 1.0)
        for param in self.mask_conv.parameters():
            param.requires_grad = False
    
    def forward(self, x, mask):
        x_masked = x * mask
        output = self.conv(x_masked)
        with torch.no_grad():
            mask_sum = self.mask_conv(mask)
            mask_sum = torch.clamp(mask_sum, min=1e-8)
        kernel_size = self.conv.kernel_size[0] * self.conv.kernel_size[1]
        output = output * (kernel_size / mask_sum)
        new_mask = (mask_sum > 0).float()
        return output, new_mask


class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, use_partial=True):
        super().__init__()
        self.use_partial = use_partial
        if use_partial:
            self.conv = PartialConv2d(in_channels, out_channels, 4, stride=2, padding=1)
        else:
            self.conv = nn.Conv2d(in_channels, out_channels, 4, stride=2, padding=1)
        self.norm = nn.InstanceNorm2d(out_channels)
        self.activation = nn.LeakyReLU(0.2, inplace=True)
    
    def forward(self, x, mask=None):
        if self.use_partial and mask is not None:
            x, mask = self.conv(x, mask)
            x = self.norm(x)
            x = self.activation(x)
            return x, mask
        else:
            x = self.conv(x)
            x = self.norm(x)
            x = self.activation(x)
            return x, mask


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, use_dropout=False):
        super().__init__()
        self.conv = nn.ConvTranspose2d(in_channels, out_channels, 4, stride=2, padding=1)
        self.norm = nn.InstanceNorm2d(out_channels)
        self.activation = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(0.5) if use_dropout else None
    
    def forward(self, x, skip=None):
        x = self.conv(x)
        x = self.norm(x)
        if self.dropout:
            x = self.dropout(x)
        x = self.activation(x)
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        return x


class InpaintingGenerator(nn.Module):
    def __init__(self, in_channels=4, out_channels=3, base_channels=64):
        super().__init__()
        self.enc1 = EncoderBlock(in_channels, base_channels, use_partial=True)
        self.enc2 = EncoderBlock(base_channels, base_channels*2, use_partial=True)
        self.enc3 = EncoderBlock(base_channels*2, base_channels*4, use_partial=True)
        self.enc4 = EncoderBlock(base_channels*4, base_channels*8, use_partial=True)
        self.enc5 = EncoderBlock(base_channels*8, base_channels*8, use_partial=True)
        self.enc6 = EncoderBlock(base_channels*8, base_channels*8, use_partial=True)
        self.enc7 = EncoderBlock(base_channels*8, base_channels*8, use_partial=True)
        
        # Fixed decoder channel counts to match skip connections:
        # dec7: 512 -> 512, + e6(512) = 1024
        # dec6: 1024 -> 512, + e5(512) = 1024
        # dec5: 1024 -> 512, + e4(512) = 1024
        # dec4: 1024 -> 256, + e3(256) = 512
        # dec3: 512 -> 128, + e2(128) = 256
        # dec2: 256 -> 64, + e1(64) = 128
        # dec1: 128 -> 64, + input(4) = 68
        self.dec7 = DecoderBlock(base_channels*8, base_channels*8, use_dropout=True)
        self.dec6 = DecoderBlock(base_channels*16, base_channels*8, use_dropout=True)
        self.dec5 = DecoderBlock(base_channels*16, base_channels*8, use_dropout=True)
        self.dec4 = DecoderBlock(base_channels*16, base_channels*4)
        self.dec3 = DecoderBlock(base_channels*8, base_channels*2)
        self.dec2 = DecoderBlock(base_channels*4, base_channels)
        self.dec1 = DecoderBlock(base_channels*2, base_channels)
        
        self.output = nn.Sequential(
            nn.Conv2d(base_channels + in_channels, out_channels, 3, padding=1),
            nn.Tanh()
        )
    
    def forward(self, x, mask):
        input_combined = torch.cat([x * mask, mask], dim=1)
        e1, m1 = self.enc1(input_combined, mask)
        e2, m2 = self.enc2(e1, m1)
        e3, m3 = self.enc3(e2, m2)
        e4, m4 = self.enc4(e3, m3)
        e5, m5 = self.enc5(e4, m4)
        e6, m6 = self.enc6(e5, m5)
        e7, _ = self.enc7(e6, m6)
        
        d7 = self.dec7(e7, e6)
        d6 = self.dec6(d7, e5)
        d5 = self.dec5(d6, e4)
        d4 = self.dec4(d5, e3)
        d3 = self.dec3(d4, e2)
        d2 = self.dec2(d3, e1)
        d1 = self.dec1(d2, input_combined)
        
        output = self.output(d1)
        completed = x * mask + output * (1 - mask)
        return completed


class PatchDiscriminator(nn.Module):
    def __init__(self, in_channels=3, base_channels=64, n_layers=4):
        super().__init__()
        
        layers = []
        layers.append(nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels, base_channels, 4, 2, 1)),
            nn.LeakyReLU(0.2, inplace=True)
        ))
        
        channels = base_channels
        for i in range(1, n_layers):
            prev_channels = channels
            channels = min(base_channels * (2 ** i), 512)
            stride = 2 if i < n_layers - 1 else 1
            layers.append(nn.Sequential(
                spectral_norm(nn.Conv2d(prev_channels, channels, 4, stride, 1)),
                nn.InstanceNorm2d(channels),
                nn.LeakyReLU(0.2, inplace=True)
            ))
        
        layers.append(spectral_norm(nn.Conv2d(channels, 1, 4, 1, 1)))
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x)


# ============================================
# LOSS FUNCTIONS
# ============================================

class VGGFeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        vgg = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1).features
        self.slice1 = nn.Sequential(*list(vgg.children())[:2])
        self.slice2 = nn.Sequential(*list(vgg.children())[2:7])
        self.slice3 = nn.Sequential(*list(vgg.children())[7:12])
        self.slice4 = nn.Sequential(*list(vgg.children())[12:21])
        for param in self.parameters():
            param.requires_grad = False
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
    
    def forward(self, x):
        x = (x + 1) / 2
        x = (x - self.mean) / self.std
        h1 = self.slice1(x)
        h2 = self.slice2(h1)
        h3 = self.slice3(h2)
        h4 = self.slice4(h3)
        return [h1, h2, h3, h4]


# ============================================
# DATASET
# ============================================

class RandomMaskGenerator:
    def __init__(self, image_size=256, mask_type='mixed'):
        self.image_size = image_size
        self.mask_type = mask_type
    
    def generate_rectangular_mask(self):
        mask = np.ones((self.image_size, self.image_size), dtype=np.float32)
        h = random.randint(self.image_size // 4, self.image_size // 2)
        w = random.randint(self.image_size // 4, self.image_size // 2)
        x = random.randint(0, self.image_size - w)
        y = random.randint(0, self.image_size - h)
        mask[y:y+h, x:x+w] = 0
        return mask
    
    def generate_irregular_mask(self):
        mask = Image.new('L', (self.image_size, self.image_size), 255)
        draw = ImageDraw.Draw(mask)
        num_strokes = random.randint(1, 5)
        for _ in range(num_strokes):
            x, y = random.randint(0, self.image_size), random.randint(0, self.image_size)
            vertices = [(x, y)]
            for _ in range(random.randint(5, 15)):
                angle = random.uniform(0, 2 * np.pi)
                length = random.randint(10, 40)
                x = max(0, min(self.image_size - 1, x + int(length * np.cos(angle))))
                y = max(0, min(self.image_size - 1, y + int(length * np.sin(angle))))
                vertices.append((x, y))
            width = random.randint(10, 30)
            for i in range(len(vertices) - 1):
                draw.line([vertices[i], vertices[i+1]], fill=0, width=width)
        return np.array(mask).astype(np.float32) / 255.0
    
    def __call__(self):
        if self.mask_type == 'mixed':
            return random.choice([self.generate_rectangular_mask, self.generate_irregular_mask])()
        elif self.mask_type == 'rectangular':
            return self.generate_rectangular_mask()
        else:
            return self.generate_irregular_mask()


def create_demo_dataset(data_dir, num_images=200):
    os.makedirs(data_dir, exist_ok=True)
    if len([f for f in os.listdir(data_dir) if f.endswith('.png')]) >= num_images:
        print(f"Demo dataset already exists in {data_dir}")
        return
    
    print(f"Creating {num_images} demo images...")
    for i in tqdm(range(num_images)):
        img = Image.new('RGB', (256, 256))
        pixels = img.load()
        base_color = (random.randint(50, 255), random.randint(50, 255), random.randint(50, 255))
        for y in range(256):
            for x in range(256):
                r = max(0, min(255, base_color[0] + random.randint(-30, 30) + (x + y) // 8))
                g = max(0, min(255, base_color[1] + random.randint(-30, 30) + (x - y) // 8))
                b = max(0, min(255, base_color[2] + random.randint(-30, 30) + (y - x) // 8))
                pixels[x, y] = (r, g, b)
        draw = ImageDraw.Draw(img)
        for _ in range(random.randint(2, 6)):
            x1, y1 = random.randint(0, 200), random.randint(0, 200)
            x2, y2 = x1 + random.randint(20, 80), y1 + random.randint(20, 80)
            color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
            if random.choice([True, False]):
                draw.ellipse([x1, y1, x2, y2], fill=color)
            else:
                draw.rectangle([x1, y1, x2, y2], fill=color)
        img = img.filter(ImageFilter.GaussianBlur(radius=2))
        img.save(os.path.join(data_dir, f"demo_{i:05d}.png"))


class InpaintingDataset(Dataset):
    def __init__(self, data_dir, image_size=256, mask_type='mixed'):
        self.data_dir = data_dir
        self.image_size = image_size
        self.mask_generator = RandomMaskGenerator(image_size, mask_type)
        self.image_paths = [os.path.join(data_dir, f) for f in os.listdir(data_dir) 
                           if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert('RGB')
        image = self.transform(image)
        mask = torch.from_numpy(self.mask_generator()).unsqueeze(0)
        incomplete = image * mask
        return {'image': image, 'mask': mask, 'incomplete': incomplete}


# ============================================
# TRAINING
# ============================================

print("\n[3/5] Creating dataset...")
if config.use_demo_data:
    create_demo_dataset(config.data_dir, config.demo_size)

dataset = InpaintingDataset(config.data_dir, config.image_size, config.mask_type)
dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True, num_workers=2, pin_memory=True)
print(f"Dataset size: {len(dataset)}")

print("\n[4/5] Initializing models...")
generator = InpaintingGenerator().to(device)
discriminator = PatchDiscriminator().to(device)
vgg = VGGFeatureExtractor().to(device)

g_optimizer = Adam(generator.parameters(), lr=config.lr_g, betas=(config.beta1, config.beta2))
d_optimizer = Adam(discriminator.parameters(), lr=config.lr_d, betas=(config.beta1, config.beta2))

print(f"Generator params: {sum(p.numel() for p in generator.parameters()):,}")
print(f"Discriminator params: {sum(p.numel() for p in discriminator.parameters()):,}")

def compute_losses(fake, real, mask, d_fake, d_real=None):
    # L1 loss (weighted by mask)
    l1_loss = F.l1_loss(fake * (1-mask), real * (1-mask)) * 6 + F.l1_loss(fake * mask, real * mask)
    
    # Perceptual loss
    fake_features = vgg(fake)
    real_features = vgg(real)
    perc_loss = sum(F.l1_loss(f, r) for f, r in zip(fake_features, real_features))
    
    # Style loss
    style_loss = 0
    for f, r in zip(fake_features, real_features):
        b, c, h, w = f.size()
        f_gram = torch.bmm(f.view(b, c, -1), f.view(b, c, -1).transpose(1, 2)) / (c*h*w)
        r_gram = torch.bmm(r.view(b, c, -1), r.view(b, c, -1).transpose(1, 2)) / (c*h*w)
        style_loss += F.l1_loss(f_gram, r_gram)
    
    # Adversarial loss
    g_adv = F.relu(1 - d_fake).mean()
    
    if d_real is not None:
        d_real_loss = F.relu(1 - d_real).mean()
        d_fake_loss = F.relu(1 + d_fake).mean()
        d_loss = d_real_loss + d_fake_loss
    else:
        d_loss = None
    
    g_loss = (config.l1_weight * l1_loss + 
              config.perceptual_weight * perc_loss + 
              config.style_weight * style_loss +
              config.adversarial_weight * g_adv)
    
    return {'g_loss': g_loss, 'd_loss': d_loss, 'l1': l1_loss, 'perc': perc_loss}

def save_vis(real, incomplete, fake, mask, path):
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    for i, (img, title) in enumerate([(real, 'Original'), ((mask[0,0]*2-1).unsqueeze(0).repeat(3,1,1), 'Mask'), 
                                       (incomplete, 'Input'), (fake, 'Output')]):
        img = (img.cpu().detach() + 1) / 2
        img = img.clamp(0, 1).permute(1, 2, 0).numpy()
        axes[i].imshow(img)
        axes[i].set_title(title)
        axes[i].axis('off')
    plt.tight_layout()
    plt.savefig(path)
    plt.close()

# Training loop
print("\n[5/5] Starting training...")
print("="*50)

global_step = 0
for epoch in range(config.epochs):
    generator.train()
    discriminator.train()
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{config.epochs}")
    for batch in pbar:
        image = batch['image'].to(device)
        mask = batch['mask'].to(device)
        incomplete = batch['incomplete'].to(device)
        
        # Train discriminator
        d_optimizer.zero_grad()
        with torch.no_grad():
            fake = generator(incomplete, mask)
        d_real = discriminator(image)
        d_fake = discriminator(fake)
        losses = compute_losses(fake, image, mask, d_fake, d_real)
        losses['d_loss'].backward()
        d_optimizer.step()
        
        # Train generator
        g_optimizer.zero_grad()
        fake = generator(incomplete, mask)
        d_fake = discriminator(fake)
        losses = compute_losses(fake, image, mask, d_fake)
        losses['g_loss'].backward()
        g_optimizer.step()
        
        pbar.set_postfix({'G': f"{losses['g_loss'].item():.3f}", 'L1': f"{losses['l1'].item():.3f}"})
        
        if global_step % config.vis_freq == 0:
            save_vis(image[0], incomplete[0], fake[0], mask[0], 
                    os.path.join(config.save_dir, 'visualizations', f'step_{global_step:06d}.png'))
        
        global_step += 1
    
    if (epoch + 1) % config.save_freq == 0:
        torch.save({
            'generator': generator.state_dict(),
            'discriminator': discriminator.state_dict(),
            'epoch': epoch
        }, os.path.join(config.save_dir, f'checkpoint_epoch_{epoch+1}.pth'))

# Save final model
torch.save({
    'generator': generator.state_dict(),
    'discriminator': discriminator.state_dict(),
    'epoch': config.epochs
}, os.path.join(config.save_dir, 'final_model.pth'))

print("\n" + "="*50)
print("TRAINING COMPLETE!")
print(f"Model saved to: {config.save_dir}/final_model.pth")
print("="*50)
