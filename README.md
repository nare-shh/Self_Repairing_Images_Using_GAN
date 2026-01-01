# Self-Repairing Image - GAN Inpainting

A deep learning system that uses a Generative Adversarial Network (GAN) to automatically repair damaged or incomplete images.

![Inpainting Demo](https://raw.githubusercontent.com/user/repo/main/demo.png)

## Features

- **U-Net Generator** with partial convolutions for mask-aware inpainting
- **PatchGAN Discriminator** with spectral normalization
- **Multiple loss functions**: L1 reconstruction, perceptual (VGG), style (Gram matrix), adversarial
- **Automatic damage detection** for scratches, missing regions, and artifacts
- **Multiple mask types**: rectangular, irregular brush strokes, center masks

## Project Structure

```
slef reparing image/
├── models/
│   ├── __init__.py
│   ├── generator.py      # U-Net with partial convolutions
│   └── discriminator.py  # PatchGAN with spectral norm
├── dataset.py            # Dataset and mask generation
├── losses.py             # Loss functions (L1, perceptual, style, adversarial)
├── train.py              # Training script
├── inference.py          # Run inference on images
├── utils.py              # Utility functions
├── requirements.txt      # Dependencies
└── README.md
```

## Quick Start (Google Colab - FREE GPU)

### Step 1: Open Google Colab
Go to [Google Colab](https://colab.research.google.com/) and create a new notebook.

### Step 2: Enable GPU
- Go to `Runtime` → `Change runtime type`
- Select `GPU` (T4 is free)
- Click `Save`

### Step 3: Upload and Run

```python
# Cell 1: Upload files
from google.colab import files
import zipfile
import os

# Create project directory
!mkdir -p inpainting_gan
%cd inpainting_gan

# Option A: Upload as zip
uploaded = files.upload()  # Upload your zip file
for fn in uploaded.keys():
    if fn.endswith('.zip'):
        with zipfile.ZipFile(fn, 'r') as z:
            z.extractall('.')

# Option B: Clone from GitHub (if you upload to GitHub)
# !git clone https://github.com/yourusername/inpainting-gan.git .
```

```python
# Cell 2: Install dependencies
!pip install torch torchvision pillow numpy tqdm gdown matplotlib
```

```python
# Cell 3: Run training with demo data
!python train.py --demo --epochs 50 --batch_size 4 --save_freq 10
```

```python
# Cell 4: Run inference
!python inference.py \
    --checkpoint ./checkpoints/final_model.pth \
    --image test_image.jpg \
    --auto_detect \
    --output repaired.png
```

## Training on Your Own Dataset

### Option 1: Use your own images

```bash
# Create a folder with your images
mkdir -p data/my_images
# Copy your images there, then:
python train.py --data_dir ./data/my_images --epochs 100
```

### Option 2: Download CelebA-HQ (faces)

```python
from dataset import download_celeba_hq
download_celeba_hq('./data/celeba')
```

### Training Command

```bash
python train.py \
    --data_dir ./data/my_images \
    --epochs 100 \
    --batch_size 8 \
    --image_size 256 \
    --mask_type mixed \
    --lr_g 0.0001 \
    --lr_d 0.0001 \
    --save_dir ./checkpoints \
    --save_freq 5 \
    --vis_freq 100
```

### Training Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--data_dir` | `./data` | Directory with training images |
| `--epochs` | `100` | Number of training epochs |
| `--batch_size` | `8` | Batch size (reduce if OOM) |
| `--image_size` | `256` | Image size for training |
| `--mask_type` | `mixed` | Mask type: rectangular, irregular, center, mixed |
| `--lr_g` | `0.0001` | Generator learning rate |
| `--lr_d` | `0.0001` | Discriminator learning rate |
| `--l1_weight` | `1.0` | L1 reconstruction loss weight |
| `--adversarial_weight` | `0.1` | Adversarial loss weight |
| `--perceptual_weight` | `0.1` | Perceptual loss weight |
| `--style_weight` | `250.0` | Style loss weight |

## Inference

### Auto-detect damaged regions

```bash
python inference.py \
    --checkpoint ./checkpoints/final_model.pth \
    --image damaged_photo.jpg \
    --auto_detect \
    --output repaired.jpg
```

### Use manual mask

```bash
python inference.py \
    --checkpoint ./checkpoints/final_model.pth \
    --image damaged_photo.jpg \
    --mask damage_mask.png \
    --output repaired.jpg
```

The mask should be:
- White (255) = Keep original pixels
- Black (0) = Inpaint/repair these pixels

## Model Architecture

### Generator (U-Net with Partial Convolutions)

```
Input: Incomplete image [B, 3, 256, 256] + Mask [B, 1, 256, 256]
  ↓
Encoder (7 blocks): 256 → 128 → 64 → 32 → 16 → 8 → 4 → 2
  ↓
Decoder (7 blocks with skip connections): 2 → 4 → 8 → 16 → 32 → 64 → 128 → 256
  ↓
Output: Completed image [B, 3, 256, 256]
```

### Discriminator (PatchGAN)

- 70×70 receptive field
- Spectral normalization for stability
- Outputs real/fake score for each patch

### Losses

| Loss | Purpose |
|------|---------|
| L1 Reconstruction | Pixel-wise accuracy |
| Perceptual (VGG) | High-level feature matching |
| Style (Gram matrix) | Texture matching |
| Adversarial (Hinge) | Realistic textures |

## Tips for Better Results

1. **More training data = better results** - Use at least 1000+ images
2. **Train longer** - 100-200 epochs for good quality
3. **Match your domain** - Train on similar images to what you'll repair
4. **Use mixed masks** - Helps the model generalize
5. **Reduce batch size if OOM** - 4 or even 2 works fine

## Troubleshooting

### Out of Memory (OOM)
```bash
python train.py --batch_size 4  # or even 2
```

### Training too slow
- Make sure GPU is enabled in Colab
- Reduce `--vis_freq` to save less frequently

### Poor results
- Train for more epochs
- Use more training data
- Check that your images are high quality

## License

MIT License - Feel free to use for any purpose.
