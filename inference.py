"""
Inference Script for Inpainting GAN

Usage:
    python inference.py --checkpoint ./checkpoints/final_model.pth --image damaged.jpg --output repaired.jpg

For automatic mask detection:
    python inference.py --checkpoint model.pth --image damaged.jpg --auto_detect

For manual mask:
    python inference.py --checkpoint model.pth --image damaged.jpg --mask mask.png
"""

import argparse
import os
import numpy as np
from PIL import Image
import torch

from models.generator import InpaintingGenerator
from utils import load_generator_only, tensor_to_image


def parse_args():
    parser = argparse.ArgumentParser(description='Inpainting Inference')
    
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to trained model checkpoint')
    parser.add_argument('--image', type=str, required=True,
                        help='Path to input image')
    parser.add_argument('--mask', type=str, default=None,
                        help='Path to mask image (white=keep, black=inpaint)')
    parser.add_argument('--output', type=str, default='./output.png',
                        help='Path to save output image')
    parser.add_argument('--image_size', type=int, default=256,
                        help='Image size for processing')
    parser.add_argument('--auto_detect', action='store_true',
                        help='Automatically detect damaged regions')
    parser.add_argument('--device', type=str, default='auto',
                        help='Device to use (auto, cuda, cpu)')
    
    return parser.parse_args()


def load_image(path, size=256):
    """Load and preprocess image."""
    img = Image.open(path).convert('RGB')
    original_size = img.size
    img = img.resize((size, size), Image.LANCZOS)
    
    # Convert to tensor [-1, 1]
    np_img = np.array(img).astype(np.float32) / 255.0
    tensor = torch.from_numpy(np_img).permute(2, 0, 1)
    tensor = tensor * 2 - 1
    tensor = tensor.unsqueeze(0)
    
    return tensor, original_size


def load_mask(path, size=256):
    """Load mask image."""
    mask = Image.open(path).convert('L')
    mask = mask.resize((size, size), Image.NEAREST)
    
    # Convert to tensor [0, 1] where 1=valid, 0=hole
    np_mask = np.array(mask).astype(np.float32) / 255.0
    tensor = torch.from_numpy(np_mask).unsqueeze(0).unsqueeze(0)
    
    return tensor


def auto_detect_damage(image_tensor, threshold=0.95):
    """
    Automatically detect damaged regions in an image.
    Detects: white/black regions, very low contrast areas, noise.
    
    Returns mask where 1=valid, 0=damaged
    """
    # Convert from [-1, 1] to [0, 1]
    img = (image_tensor + 1) / 2
    
    # Detect very bright regions (potential white damage)
    bright = (img.mean(dim=1, keepdim=True) > threshold).float()
    
    # Detect very dark regions (potential black damage)
    dark = (img.mean(dim=1, keepdim=True) < (1 - threshold)).float()
    
    # Combine detections
    damage = bright + dark
    
    # Create mask (1=valid, 0=damaged)
    mask = 1 - damage.clamp(0, 1)
    
    # Dilate the mask slightly to ensure coverage
    kernel = torch.ones(1, 1, 5, 5, device=image_tensor.device)
    damage_dilated = torch.nn.functional.conv2d(
        1 - mask, kernel, padding=2
    ) > 0
    mask = 1 - damage_dilated.float()
    
    return mask


def detect_scratches(image_tensor, threshold=0.3):
    """
    Detect scratch-like damage using edge detection.
    """
    # Simple Sobel-like edge detection
    img = (image_tensor + 1) / 2  # [0, 1]
    gray = img.mean(dim=1, keepdim=True)  # Convert to grayscale
    
    # Horizontal and vertical gradients
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32)
    
    sobel_x = sobel_x.view(1, 1, 3, 3).to(image_tensor.device)
    sobel_y = sobel_y.view(1, 1, 3, 3).to(image_tensor.device)
    
    grad_x = torch.nn.functional.conv2d(gray, sobel_x, padding=1)
    grad_y = torch.nn.functional.conv2d(gray, sobel_y, padding=1)
    
    # Gradient magnitude
    grad_mag = torch.sqrt(grad_x**2 + grad_y**2)
    
    # High gradient areas might be scratches
    scratches = (grad_mag > threshold).float()
    
    # Thin line detection (scratches are usually thin)
    # If gradient is high but area is small, likely scratch
    
    mask = 1 - scratches
    
    return mask


def inpaint(generator, image, mask, device):
    """Run inpainting on an image."""
    generator.eval()
    
    with torch.no_grad():
        image = image.to(device)
        mask = mask.to(device)
        
        # Create incomplete image
        incomplete = image * mask
        
        # Run generator
        output = generator(incomplete, mask)
    
    return output


def main():
    args = parse_args()
    
    # Device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    
    # Load model
    generator = InpaintingGenerator(in_channels=4, out_channels=3)
    generator = load_generator_only(generator, args.checkpoint, device)
    generator = generator.to(device)
    generator.eval()
    
    # Load image
    print(f"Loading image: {args.image}")
    image, original_size = load_image(args.image, args.image_size)
    image = image.to(device)
    
    # Get mask
    if args.mask:
        print(f"Loading mask: {args.mask}")
        mask = load_mask(args.mask, args.image_size)
    elif args.auto_detect:
        print("Auto-detecting damaged regions...")
        mask = auto_detect_damage(image)
        
        # Also check for scratches
        scratch_mask = detect_scratches(image)
        mask = mask * scratch_mask  # Combine masks
        
        # Count damaged pixels
        damage_percent = (1 - mask.mean()).item() * 100
        print(f"Detected {damage_percent:.1f}% damaged area")
    else:
        # Default: create center mask for demo
        print("No mask provided, using center mask for demo")
        mask = torch.ones(1, 1, args.image_size, args.image_size)
        center = args.image_size // 4
        mask[:, :, center:center*3, center:center*3] = 0
    
    mask = mask.to(device)
    
    # Run inpainting
    print("Running inpainting...")
    output = inpaint(generator, image, mask, device)
    
    # Save output
    output_img = tensor_to_image(output[0])
    
    # Resize back to original size if needed
    if output_img.size != original_size:
        output_img = output_img.resize(original_size, Image.LANCZOS)
    
    # Create output directory if needed
    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    output_img.save(args.output)
    print(f"Saved repaired image to: {args.output}")
    
    # Also save comparison
    comparison_path = args.output.replace('.png', '_comparison.png').replace('.jpg', '_comparison.jpg')
    
    # Create side-by-side comparison
    input_img = tensor_to_image(image[0])
    masked_img = tensor_to_image((image * mask)[0])
    
    if input_img.size != original_size:
        input_img = input_img.resize(original_size, Image.LANCZOS)
        masked_img = masked_img.resize(original_size, Image.LANCZOS)
    
    comparison = Image.new('RGB', (original_size[0] * 3, original_size[1]))
    comparison.paste(input_img, (0, 0))
    comparison.paste(masked_img, (original_size[0], 0))
    comparison.paste(output_img, (original_size[0] * 2, 0))
    comparison.save(comparison_path)
    print(f"Saved comparison to: {comparison_path}")


if __name__ == "__main__":
    main()
