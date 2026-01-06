# Simple Image Inpainting GAN

A simple GAN that fills in missing parts of images automatically.

## Files

| File | Purpose |
|------|---------|
| `simple_inpainting.py` | **Everything in one file!** |
| `requirements.txt` | Dependencies |

## Quick Start

### 1. Install dependencies
```bash
pip install torch torchvision pillow numpy tqdm matplotlib
```

### 2. Train the model
```bash
python simple_inpainting.py
```
This will:
- Create demo training images automatically
- Train for 30 epochs
- Save the model to `./output/generator.pth`

### 3. Repair an image
```bash
python simple_inpainting.py repair damaged_image.jpg
```

## Training on Google Colab (FREE GPU)

1. Go to [colab.research.google.com](https://colab.research.google.com)
2. `Runtime` → `Change runtime type` → `GPU`
3. Upload `simple_inpainting.py`
4. Run:
```python
!pip install torch torchvision pillow numpy tqdm matplotlib
!python simple_inpainting.py
```

## How It Works

1. **U-Net Generator**: Takes masked image as input, outputs completed image
2. **Discriminator**: Tells real images from generated ones
3. **Training**: Generator learns to fool discriminator while matching missing regions

## Settings

Edit these at the top of `simple_inpainting.py`:

```python
IMAGE_SIZE = 128    # Image size (smaller = faster)
BATCH_SIZE = 8      # Reduce if out of memory
EPOCHS = 30         # More = better results
DATA_DIR = './data' # Training images folder
```

## Use Your Own Images

Put your images in `./data/` folder before training:
```bash
mkdir data
cp your_images/*.jpg data/
python simple_inpainting.py
```
