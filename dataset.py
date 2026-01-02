"""
Dataset and Mask Generation for Inpainting GAN
Supports automatic dataset download and random mask generation.
"""

import os
import random
import numpy as np
from PIL import Image, ImageDraw
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import gdown
from zipfile import ZipFile
from tqdm import tqdm


class RandomMaskGenerator:
    """
    Generates random masks for inpainting training.
    Supports: rectangular, irregular (brush strokes), and center masks.
    """
    
    def __init__(self, image_size=256, mask_type='mixed'):
        self.image_size = image_size
        self.mask_type = mask_type
    
    def generate_rectangular_mask(self):
        """Generate random rectangular mask."""
        mask = np.ones((self.image_size, self.image_size), dtype=np.float32)
        
        # Random rectangle
        h = random.randint(self.image_size // 4, self.image_size // 2)
        w = random.randint(self.image_size // 4, self.image_size // 2)
        x = random.randint(0, self.image_size - w)
        y = random.randint(0, self.image_size - h)
        
        mask[y:y+h, x:x+w] = 0
        
        return mask
    
    def generate_center_mask(self):
        """Generate mask in center of image."""
        mask = np.ones((self.image_size, self.image_size), dtype=np.float32)
        
        h = self.image_size // 2
        w = self.image_size // 2
        x = (self.image_size - w) // 2
        y = (self.image_size - h) // 2
        
        mask[y:y+h, x:x+w] = 0
        
        return mask
    
    def generate_irregular_mask(self, min_strokes=1, max_strokes=5):
        """Generate irregular mask using random brush strokes."""
        mask = Image.new('L', (self.image_size, self.image_size), 255)
        draw = ImageDraw.Draw(mask)
        
        num_strokes = random.randint(min_strokes, max_strokes)
        
        for _ in range(num_strokes):
            # Random starting point
            x = random.randint(0, self.image_size)
            y = random.randint(0, self.image_size)
            
            # Random walk for brush stroke
            num_vertices = random.randint(5, 20)
            vertices = [(x, y)]
            
            for _ in range(num_vertices):
                angle = random.uniform(0, 2 * np.pi)
                length = random.randint(10, 50)
                x = max(0, min(self.image_size - 1, x + int(length * np.cos(angle))))
                y = max(0, min(self.image_size - 1, y + int(length * np.sin(angle))))
                vertices.append((x, y))
            
            # Draw thick line
            width = random.randint(10, 40)
            for i in range(len(vertices) - 1):
                draw.line([vertices[i], vertices[i+1]], fill=0, width=width)
            
            # Draw circles at vertices for rounded strokes
            for v in vertices:
                r = width // 2
                draw.ellipse([v[0]-r, v[1]-r, v[0]+r, v[1]+r], fill=0)
        
        mask = np.array(mask).astype(np.float32) / 255.0
        return mask
    
    def generate_multi_rect_mask(self, min_rects=1, max_rects=4):
        """Generate multiple random rectangular masks."""
        mask = np.ones((self.image_size, self.image_size), dtype=np.float32)
        
        num_rects = random.randint(min_rects, max_rects)
        
        for _ in range(num_rects):
            h = random.randint(20, self.image_size // 3)
            w = random.randint(20, self.image_size // 3)
            x = random.randint(0, self.image_size - w)
            y = random.randint(0, self.image_size - h)
            mask[y:y+h, x:x+w] = 0
        
        return mask
    
    def __call__(self):
        """Generate a random mask based on mask_type."""
        if self.mask_type == 'mixed':
            choice = random.choice(['rectangular', 'irregular', 'multi_rect', 'center'])
        else:
            choice = self.mask_type
        
        if choice == 'rectangular':
            return self.generate_rectangular_mask()
        elif choice == 'irregular':
            return self.generate_irregular_mask()
        elif choice == 'multi_rect':
            return self.generate_multi_rect_mask()
        elif choice == 'center':
            return self.generate_center_mask()
        else:
            return self.generate_rectangular_mask()


class InpaintingDataset(Dataset):
    """
    Dataset for inpainting training.
    Loads images from a folder and generates random masks.
    """
    
    def __init__(
        self,
        data_dir,
        image_size=256,
        mask_type='mixed',
        augment=True
    ):
        self.data_dir = data_dir
        self.image_size = image_size
        self.mask_generator = RandomMaskGenerator(image_size, mask_type)
        self.augment = augment
        
        # Find all images
        self.image_paths = []
        valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
        
        for root, _, files in os.walk(data_dir):
            for f in files:
                if os.path.splitext(f)[1].lower() in valid_extensions:
                    self.image_paths.append(os.path.join(root, f))
        
        print(f"Found {len(self.image_paths)} images in {data_dir}")
        
        # Transforms
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        
        self.augment_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
        ])
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # Load image
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        
        # Augmentation
        if self.augment:
            image = self.augment_transform(image)
        
        # Transform to tensor
        image = self.transform(image)
        
        # Generate mask
        mask = self.mask_generator()
        mask = torch.from_numpy(mask).unsqueeze(0)  # [1, H, W]
        
        # Create incomplete image
        incomplete = image * mask
        
        return {
            'image': image,           # Ground truth [3, H, W]
            'mask': mask,             # Mask [1, H, W] where 1=valid, 0=hole
            'incomplete': incomplete  # Incomplete image [3, H, W]
        }


def download_celeba_hq(data_dir, sample_size=5000):
    """
    Download a sample of CelebA-HQ dataset for training.
    This downloads a subset to keep the file size manageable.
    """
    os.makedirs(data_dir, exist_ok=True)
    
    # Check if already downloaded
    if len(os.listdir(data_dir)) >= sample_size:
        print(f"Dataset already exists in {data_dir}")
        return
    
    print("Downloading CelebA-HQ sample dataset...")
    
    # Google Drive link for CelebA-HQ (256x256)
    # Using a publicly available preprocessed version
    url = "https://drive.google.com/uc?id=1badu11NqxGf6qM3PTTooQDJvQbejgbTv"
    zip_path = os.path.join(data_dir, "celeba_hq_256.zip")
    
    try:
        gdown.download(url, zip_path, quiet=False)
        
        # Extract
        print("Extracting dataset...")
        with ZipFile(zip_path, 'r') as zf:
            zf.extractall(data_dir)
        
        # Clean up
        os.remove(zip_path)
        print(f"Dataset ready in {data_dir}")
        
    except Exception as e:
        print(f"Failed to download CelebA-HQ: {e}")
        print("Please manually download images to the data directory.")
        print("Alternative: Use your own images for training.")


def download_places365_sample(data_dir, sample_size=5000):
    """
    Download a sample of Places365 dataset for training.
    Good for general scene inpainting.
    """
    os.makedirs(data_dir, exist_ok=True)
    
    if len(os.listdir(data_dir)) >= 100:
        print(f"Dataset already exists in {data_dir}")
        return
    
    print("Downloading Places365 sample dataset...")
    
    # Using validation set (smaller)
    url = "http://data.csail.mit.edu/places/places365/val_256.tar"
    tar_path = os.path.join(data_dir, "val_256.tar")
    
    try:
        import urllib.request
        import tarfile
        
        print(f"Downloading from {url}...")
        urllib.request.urlretrieve(url, tar_path)
        
        print("Extracting dataset...")
        with tarfile.open(tar_path, 'r') as tf:
            tf.extractall(data_dir)
        
        os.remove(tar_path)
        print(f"Dataset ready in {data_dir}")
        
    except Exception as e:
        print(f"Failed to download Places365: {e}")
        print("Please manually download images to the data directory.")


def create_demo_dataset(data_dir, num_images=100):
    """
    Create a small demo dataset with synthetic images.
    Useful for testing the training pipeline.
    """
    os.makedirs(data_dir, exist_ok=True)
    
    if len(os.listdir(data_dir)) >= num_images:
        print(f"Demo dataset already exists in {data_dir}")
        return
    
    print(f"Creating demo dataset with {num_images} synthetic images...")
    
    for i in tqdm(range(num_images)):
        # Create random colorful image
        img = Image.new('RGB', (256, 256))
        pixels = img.load()
        
        # Random gradient or pattern
        base_color = (
            random.randint(50, 255),
            random.randint(50, 255),
            random.randint(50, 255)
        )
        
        for y in range(256):
            for x in range(256):
                # Add some variation
                r = max(0, min(255, base_color[0] + random.randint(-30, 30) + (x + y) // 8))
                g = max(0, min(255, base_color[1] + random.randint(-30, 30) + (x - y) // 8))
                b = max(0, min(255, base_color[2] + random.randint(-30, 30) + (y - x) // 8))
                pixels[x, y] = (r, g, b)
        
        # Add some shapes
        draw = ImageDraw.Draw(img)
        for _ in range(random.randint(2, 8)):
            shape_type = random.choice(['ellipse', 'rectangle'])
            x1 = random.randint(0, 200)
            y1 = random.randint(0, 200)
            x2 = x1 + random.randint(20, 80)
            y2 = y1 + random.randint(20, 80)
            color = (
                random.randint(0, 255),
                random.randint(0, 255),
                random.randint(0, 255)
            )
            if shape_type == 'ellipse':
                draw.ellipse([x1, y1, x2, y2], fill=color)
            else:
                draw.rectangle([x1, y1, x2, y2], fill=color)
        
        # Apply blur for smoother look
        from PIL import ImageFilter
        img = img.filter(ImageFilter.GaussianBlur(radius=2))
        
        img.save(os.path.join(data_dir, f"demo_{i:05d}.png"))
    
    print(f"Demo dataset created in {data_dir}")


if __name__ == "__main__":
    # Test dataset creation
    print("Testing dataset and mask generation...")
    
    # Create demo dataset
    demo_dir = "./demo_data"
    create_demo_dataset(demo_dir, num_images=10)
    
    # Test dataset
    dataset = InpaintingDataset(demo_dir, image_size=256)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
    
    # Get a sample
    sample = next(iter(dataloader))
    print(f"Image shape: {sample['image'].shape}")
    print(f"Mask shape: {sample['mask'].shape}")
    print(f"Incomplete shape: {sample['incomplete'].shape}")
    print(f"Image range: [{sample['image'].min():.2f}, {sample['image'].max():.2f}]")
    print(f"Mask unique values: {sample['mask'].unique().tolist()}")
