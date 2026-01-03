"""
Utility Functions for Inpainting GAN
Image processing, visualization, and checkpointing helpers.
"""

import os
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from torchvision.utils import make_grid


def tensor_to_image(tensor):
    """
    Convert tensor to PIL Image.
    
    Args:
        tensor: Tensor [C, H, W] or [B, C, H, W] in range [-1, 1]
    
    Returns:
        PIL Image or list of PIL Images
    """
    # Handle batch dimension
    if tensor.dim() == 4:
        return [tensor_to_image(t) for t in tensor]
    
    # Move to CPU and denormalize
    tensor = tensor.detach().cpu()
    tensor = (tensor + 1) / 2  # [-1, 1] -> [0, 1]
    tensor = tensor.clamp(0, 1)
    
    # Convert to numpy
    np_img = tensor.permute(1, 2, 0).numpy()
    np_img = (np_img * 255).astype(np.uint8)
    
    return Image.fromarray(np_img)


def image_to_tensor(image, device='cpu'):
    """
    Convert PIL Image to tensor.
    
    Args:
        image: PIL Image
        device: Target device
    
    Returns:
        Tensor [1, C, H, W] in range [-1, 1]
    """
    np_img = np.array(image).astype(np.float32) / 255.0
    tensor = torch.from_numpy(np_img).permute(2, 0, 1)
    tensor = tensor * 2 - 1  # [0, 1] -> [-1, 1]
    tensor = tensor.unsqueeze(0).to(device)
    return tensor


def mask_to_tensor(mask, device='cpu'):
    """
    Convert mask (PIL Image or numpy array) to tensor.
    
    Args:
        mask: PIL Image (L mode) or numpy array
        device: Target device
    
    Returns:
        Tensor [1, 1, H, W] where 1=valid, 0=hole
    """
    if isinstance(mask, Image.Image):
        mask = np.array(mask.convert('L'))
    
    # Normalize to [0, 1]
    mask = mask.astype(np.float32) / 255.0
    
    # Convert to tensor
    tensor = torch.from_numpy(mask).unsqueeze(0).unsqueeze(0)
    
    return tensor.to(device)


def save_visualization(
    image, incomplete, output, mask, 
    save_path, title=None
):
    """
    Save visualization comparing input, masked, and output images.
    
    Args:
        image: Ground truth [B, 3, H, W]
        incomplete: Masked input [B, 3, H, W]
        output: Generator output [B, 3, H, W]
        mask: Binary mask [B, 1, H, W]
        save_path: Path to save the visualization
        title: Optional title
    """
    batch_size = image.size(0)
    
    # Take first 4 samples if batch is larger
    n = min(4, batch_size)
    
    fig, axes = plt.subplots(n, 4, figsize=(16, 4*n))
    
    if n == 1:
        axes = [axes]
    
    for i in range(n):
        # Ground truth
        gt = tensor_to_image(image[i])
        axes[i][0].imshow(gt)
        axes[i][0].set_title('Ground Truth')
        axes[i][0].axis('off')
        
        # Mask
        m = mask[i, 0].detach().cpu().numpy()
        axes[i][1].imshow(m, cmap='gray')
        axes[i][1].set_title('Mask (white=valid)')
        axes[i][1].axis('off')
        
        # Incomplete (masked input)
        inc = tensor_to_image(incomplete[i])
        axes[i][2].imshow(inc)
        axes[i][2].set_title('Incomplete')
        axes[i][2].axis('off')
        
        # Output
        out = tensor_to_image(output[i])
        axes[i][3].imshow(out)
        axes[i][3].set_title('Inpainted')
        axes[i][3].axis('off')
    
    if title:
        fig.suptitle(title)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def save_checkpoint(
    generator, discriminator, 
    g_optimizer, d_optimizer,
    epoch, step, save_path
):
    """Save model checkpoint."""
    checkpoint = {
        'epoch': epoch,
        'step': step,
        'generator_state_dict': generator.state_dict(),
        'discriminator_state_dict': discriminator.state_dict(),
        'g_optimizer_state_dict': g_optimizer.state_dict(),
        'd_optimizer_state_dict': d_optimizer.state_dict(),
    }
    torch.save(checkpoint, save_path)
    print(f"Checkpoint saved to {save_path}")


def load_checkpoint(
    generator, discriminator,
    g_optimizer, d_optimizer,
    checkpoint_path, device='cuda'
):
    """Load model checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    generator.load_state_dict(checkpoint['generator_state_dict'])
    discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
    
    if g_optimizer is not None:
        g_optimizer.load_state_dict(checkpoint['g_optimizer_state_dict'])
    if d_optimizer is not None:
        d_optimizer.load_state_dict(checkpoint['d_optimizer_state_dict'])
    
    epoch = checkpoint['epoch']
    step = checkpoint['step']
    
    print(f"Loaded checkpoint from epoch {epoch}, step {step}")
    
    return epoch, step


def load_generator_only(generator, checkpoint_path, device='cuda'):
    """Load only the generator from a checkpoint (for inference)."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    generator.load_state_dict(checkpoint['generator_state_dict'])
    print(f"Generator loaded from {checkpoint_path}")
    return generator


class AverageMeter:
    """Tracks average and current value of a metric."""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def create_grid_image(tensors, nrow=4):
    """
    Create a grid image from a batch of tensors.
    
    Args:
        tensors: Tensor [B, C, H, W]
        nrow: Number of images per row
    
    Returns:
        PIL Image
    """
    tensors = (tensors + 1) / 2  # [-1, 1] -> [0, 1]
    tensors = tensors.clamp(0, 1)
    
    grid = make_grid(tensors, nrow=nrow, padding=2, normalize=False)
    
    np_img = grid.permute(1, 2, 0).cpu().numpy()
    np_img = (np_img * 255).astype(np.uint8)
    
    return Image.fromarray(np_img)


def count_parameters(model):
    """Count trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Test utilities
    print("Testing utility functions...")
    
    # Create dummy tensor
    tensor = torch.randn(3, 256, 256)
    img = tensor_to_image(tensor)
    print(f"tensor_to_image: {tensor.shape} -> {img.size}")
    
    # Convert back
    tensor2 = image_to_tensor(img)
    print(f"image_to_tensor: {img.size} -> {tensor2.shape}")
