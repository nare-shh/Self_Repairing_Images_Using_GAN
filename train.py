"""
Training Script for Inpainting GAN

Usage:
    python train.py --data_dir ./data --epochs 100 --batch_size 8

For Google Colab:
    1. Upload all files to Colab
    2. Run: !pip install -r requirements.txt
    3. Run: !python train.py --data_dir ./data --epochs 100
"""

import os
import argparse
from datetime import datetime
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm

from models.generator import InpaintingGenerator
from models.discriminator import PatchDiscriminator
from losses import InpaintingLoss
from dataset import InpaintingDataset, create_demo_dataset, download_celeba_hq
from utils import (
    save_checkpoint, load_checkpoint, save_visualization,
    AverageMeter, count_parameters
)


def parse_args():
    parser = argparse.ArgumentParser(description='Train Inpainting GAN')
    
    # Data
    parser.add_argument('--data_dir', type=str, default='./data',
                        help='Directory containing training images')
    parser.add_argument('--image_size', type=int, default=256,
                        help='Image size for training')
    parser.add_argument('--mask_type', type=str, default='mixed',
                        choices=['rectangular', 'irregular', 'center', 'mixed'],
                        help='Type of mask to use')
    
    # Training
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size')
    parser.add_argument('--lr_g', type=float, default=1e-4,
                        help='Generator learning rate')
    parser.add_argument('--lr_d', type=float, default=1e-4,
                        help='Discriminator learning rate')
    parser.add_argument('--beta1', type=float, default=0.5,
                        help='Adam beta1')
    parser.add_argument('--beta2', type=float, default=0.999,
                        help='Adam beta2')
    
    # Loss weights
    parser.add_argument('--l1_weight', type=float, default=1.0,
                        help='L1 reconstruction loss weight')
    parser.add_argument('--adversarial_weight', type=float, default=0.1,
                        help='Adversarial loss weight')
    parser.add_argument('--perceptual_weight', type=float, default=0.1,
                        help='Perceptual loss weight')
    parser.add_argument('--style_weight', type=float, default=250.0,
                        help='Style loss weight')
    
    # Checkpointing
    parser.add_argument('--save_dir', type=str, default='./checkpoints',
                        help='Directory to save checkpoints')
    parser.add_argument('--save_freq', type=int, default=5,
                        help='Save checkpoint every N epochs')
    parser.add_argument('--vis_freq', type=int, default=100,
                        help='Save visualization every N steps')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    
    # Demo mode
    parser.add_argument('--demo', action='store_true',
                        help='Use synthetic demo data for testing')
    parser.add_argument('--demo_size', type=int, default=100,
                        help='Number of demo images to generate')
    
    return parser.parse_args()


def train_step(
    generator, discriminator,
    g_optimizer, d_optimizer,
    loss_fn, batch, device
):
    """Single training step."""
    
    image = batch['image'].to(device)
    mask = batch['mask'].to(device)
    incomplete = batch['incomplete'].to(device)
    
    # ==================
    # Train Discriminator
    # ==================
    d_optimizer.zero_grad()
    
    # Generate fake image
    with torch.no_grad():
        fake = generator(incomplete, mask)
    
    # Discriminator predictions
    d_real = discriminator(image)
    d_fake = discriminator(fake.detach())
    
    # Discriminator loss
    d_losses = loss_fn.discriminator_loss(d_real, d_fake)
    d_loss = d_losses['total']
    
    d_loss.backward()
    d_optimizer.step()
    
    # ==================
    # Train Generator
    # ==================
    g_optimizer.zero_grad()
    
    # Generate fake image
    fake = generator(incomplete, mask)
    
    # Discriminator prediction for fake
    d_fake = discriminator(fake)
    
    # Generator loss
    g_losses = loss_fn.generator_loss(fake, image, mask, d_fake)
    g_loss = g_losses['total']
    
    g_loss.backward()
    g_optimizer.step()
    
    return {
        'g_loss': g_losses['total'].item(),
        'g_l1': g_losses['l1'].item(),
        'g_perceptual': g_losses['perceptual'].item(),
        'g_style': g_losses['style'].item(),
        'g_adversarial': g_losses['adversarial'].item(),
        'd_loss': d_losses['total'].item(),
        'd_real': d_losses['real'].item(),
        'd_fake': d_losses['fake'].item(),
        'fake': fake,
        'image': image,
        'mask': mask,
        'incomplete': incomplete
    }


def train(args):
    """Main training loop."""
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    if device.type == 'cpu':
        print("WARNING: Training on CPU will be very slow!")
        print("Consider using Google Colab with GPU runtime.")
    
    # Create directories
    os.makedirs(args.save_dir, exist_ok=True)
    vis_dir = os.path.join(args.save_dir, 'visualizations')
    os.makedirs(vis_dir, exist_ok=True)
    
    # Prepare data
    if args.demo:
        print("Using demo dataset for testing...")
        demo_dir = os.path.join(args.data_dir, 'demo')
        create_demo_dataset(demo_dir, num_images=args.demo_size)
        args.data_dir = demo_dir
    elif not os.path.exists(args.data_dir) or len(os.listdir(args.data_dir)) == 0:
        print(f"Data directory {args.data_dir} is empty.")
        print("Creating demo dataset for testing...")
        create_demo_dataset(args.data_dir, num_images=100)
    
    # Dataset
    dataset = InpaintingDataset(
        args.data_dir,
        image_size=args.image_size,
        mask_type=args.mask_type,
        augment=True
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2 if device.type == 'cuda' else 0,
        pin_memory=True if device.type == 'cuda' else False
    )
    
    print(f"Dataset size: {len(dataset)}")
    print(f"Batches per epoch: {len(dataloader)}")
    
    # Models
    generator = InpaintingGenerator(in_channels=4, out_channels=3).to(device)
    discriminator = PatchDiscriminator(in_channels=3).to(device)
    
    print(f"Generator parameters: {count_parameters(generator):,}")
    print(f"Discriminator parameters: {count_parameters(discriminator):,}")
    
    # Optimizers
    g_optimizer = Adam(
        generator.parameters(),
        lr=args.lr_g,
        betas=(args.beta1, args.beta2)
    )
    d_optimizer = Adam(
        discriminator.parameters(),
        lr=args.lr_d,
        betas=(args.beta1, args.beta2)
    )
    
    # Learning rate scheduler
    g_scheduler = StepLR(g_optimizer, step_size=30, gamma=0.5)
    d_scheduler = StepLR(d_optimizer, step_size=30, gamma=0.5)
    
    # Loss
    loss_fn = InpaintingLoss(
        l1_weight=args.l1_weight,
        adversarial_weight=args.adversarial_weight,
        perceptual_weight=args.perceptual_weight,
        style_weight=args.style_weight
    ).to(device)
    
    # Resume from checkpoint
    start_epoch = 0
    global_step = 0
    
    if args.resume:
        start_epoch, global_step = load_checkpoint(
            generator, discriminator,
            g_optimizer, d_optimizer,
            args.resume, device
        )
        start_epoch += 1
    
    # Training loop
    print(f"\nStarting training from epoch {start_epoch}...")
    print(f"Running for {args.epochs} epochs")
    print("-" * 50)
    
    for epoch in range(start_epoch, args.epochs):
        generator.train()
        discriminator.train()
        
        # Metrics
        g_loss_meter = AverageMeter()
        d_loss_meter = AverageMeter()
        
        # Progress bar
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.epochs}")
        
        for batch_idx, batch in enumerate(pbar):
            # Training step
            results = train_step(
                generator, discriminator,
                g_optimizer, d_optimizer,
                loss_fn, batch, device
            )
            
            # Update metrics
            g_loss_meter.update(results['g_loss'])
            d_loss_meter.update(results['d_loss'])
            
            # Update progress bar
            pbar.set_postfix({
                'G': f"{results['g_loss']:.4f}",
                'D': f"{results['d_loss']:.4f}",
                'L1': f"{results['g_l1']:.4f}"
            })
            
            # Save visualization
            if global_step % args.vis_freq == 0:
                vis_path = os.path.join(vis_dir, f"step_{global_step:06d}.png")
                save_visualization(
                    results['image'],
                    results['incomplete'],
                    results['fake'],
                    results['mask'],
                    vis_path,
                    title=f"Epoch {epoch+1}, Step {global_step}"
                )
            
            global_step += 1
        
        # Update learning rate
        g_scheduler.step()
        d_scheduler.step()
        
        # Epoch summary
        print(f"Epoch {epoch+1} | G Loss: {g_loss_meter.avg:.4f} | D Loss: {d_loss_meter.avg:.4f}")
        
        # Save checkpoint
        if (epoch + 1) % args.save_freq == 0 or epoch == args.epochs - 1:
            checkpoint_path = os.path.join(
                args.save_dir,
                f"checkpoint_epoch_{epoch+1:03d}.pth"
            )
            save_checkpoint(
                generator, discriminator,
                g_optimizer, d_optimizer,
                epoch, global_step, checkpoint_path
            )
    
    # Save final model
    final_path = os.path.join(args.save_dir, 'final_model.pth')
    save_checkpoint(
        generator, discriminator,
        g_optimizer, d_optimizer,
        args.epochs - 1, global_step, final_path
    )
    
    print("\nTraining complete!")
    print(f"Final model saved to {final_path}")


if __name__ == "__main__":
    args = parse_args()
    train(args)
