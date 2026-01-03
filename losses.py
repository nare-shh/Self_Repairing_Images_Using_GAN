"""
Loss Functions for Inpainting GAN
Includes reconstruction, adversarial, perceptual (VGG), and style losses.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class VGGFeatureExtractor(nn.Module):
    """
    Extracts features from pretrained VGG19 for perceptual loss.
    Uses features from relu1_1, relu2_1, relu3_1, relu4_1, relu5_1.
    """
    
    def __init__(self, requires_grad=False):
        super().__init__()
        
        vgg = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1).features
        
        # Extract feature layers
        self.slice1 = nn.Sequential(*list(vgg.children())[:2])   # relu1_1
        self.slice2 = nn.Sequential(*list(vgg.children())[2:7])  # relu2_1
        self.slice3 = nn.Sequential(*list(vgg.children())[7:12]) # relu3_1
        self.slice4 = nn.Sequential(*list(vgg.children())[12:21])# relu4_1
        self.slice5 = nn.Sequential(*list(vgg.children())[21:30])# relu5_1
        
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False
        
        # Normalization for VGG input
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
    
    def forward(self, x):
        """
        Args:
            x: Image tensor [B, 3, H, W] in range [-1, 1]
        
        Returns:
            List of feature maps at different layers
        """
        # Normalize from [-1, 1] to [0, 1] then VGG normalization
        x = (x + 1) / 2
        x = (x - self.mean) / self.std
        
        h1 = self.slice1(x)
        h2 = self.slice2(h1)
        h3 = self.slice3(h2)
        h4 = self.slice4(h3)
        h5 = self.slice5(h4)
        
        return [h1, h2, h3, h4, h5]


class ReconstructionLoss(nn.Module):
    """L1 reconstruction loss with optional mask weighting."""
    
    def __init__(self, hole_weight=6.0, valid_weight=1.0):
        super().__init__()
        self.hole_weight = hole_weight
        self.valid_weight = valid_weight
        self.l1 = nn.L1Loss(reduction='none')
    
    def forward(self, output, target, mask):
        """
        Args:
            output: Generated image [B, 3, H, W]
            target: Ground truth image [B, 3, H, W]
            mask: Binary mask [B, 1, H, W] where 1 = valid, 0 = hole
        """
        loss = self.l1(output, target)
        
        # Weight hole pixels more heavily
        hole_loss = (loss * (1 - mask)).sum() / ((1 - mask).sum() + 1e-8)
        valid_loss = (loss * mask).sum() / (mask.sum() + 1e-8)
        
        return self.hole_weight * hole_loss + self.valid_weight * valid_loss


class PerceptualLoss(nn.Module):
    """Perceptual loss using VGG19 features."""
    
    def __init__(self, weights=[1.0, 1.0, 1.0, 1.0, 1.0]):
        super().__init__()
        self.vgg = VGGFeatureExtractor()
        self.weights = weights
        self.l1 = nn.L1Loss()
    
    def forward(self, output, target):
        """
        Args:
            output: Generated image [B, 3, H, W]
            target: Ground truth image [B, 3, H, W]
        """
        output_features = self.vgg(output)
        target_features = self.vgg(target)
        
        loss = 0
        for i, (out_f, tar_f) in enumerate(zip(output_features, target_features)):
            loss += self.weights[i] * self.l1(out_f, tar_f)
        
        return loss


class StyleLoss(nn.Module):
    """Style loss using Gram matrix of VGG features."""
    
    def __init__(self):
        super().__init__()
        self.vgg = VGGFeatureExtractor()
    
    def gram_matrix(self, x):
        """Compute Gram matrix for style matching."""
        b, c, h, w = x.size()
        features = x.view(b, c, -1)
        gram = torch.bmm(features, features.transpose(1, 2))
        return gram / (c * h * w)
    
    def forward(self, output, target):
        """
        Args:
            output: Generated image [B, 3, H, W]
            target: Ground truth image [B, 3, H, W]
        """
        output_features = self.vgg(output)
        target_features = self.vgg(target)
        
        loss = 0
        for out_f, tar_f in zip(output_features, target_features):
            out_gram = self.gram_matrix(out_f)
            tar_gram = self.gram_matrix(tar_f)
            loss += F.l1_loss(out_gram, tar_gram)
        
        return loss


class AdversarialLoss(nn.Module):
    """
    Adversarial loss for GAN training.
    Supports multiple loss types: vanilla, lsgan, hinge.
    """
    
    def __init__(self, loss_type='hinge'):
        super().__init__()
        self.loss_type = loss_type
        
        if loss_type == 'vanilla':
            self.bce = nn.BCEWithLogitsLoss()
    
    def forward(self, pred, target_is_real):
        """
        Args:
            pred: Discriminator prediction
            target_is_real: True if target is real, False if fake
        """
        if self.loss_type == 'hinge':
            if target_is_real:
                loss = F.relu(1 - pred).mean()
            else:
                loss = F.relu(1 + pred).mean()
        
        elif self.loss_type == 'lsgan':
            target = torch.ones_like(pred) if target_is_real else torch.zeros_like(pred)
            loss = F.mse_loss(pred, target)
        
        elif self.loss_type == 'vanilla':
            target = torch.ones_like(pred) if target_is_real else torch.zeros_like(pred)
            loss = self.bce(pred, target)
        
        return loss


class InpaintingLoss(nn.Module):
    """
    Combined loss for inpainting GAN.
    Totla loss = λ1 * L1 + λadv * Adversarial + λperc * Perceptual + λstyle * Style
    """
    
    def __init__(
        self,
        l1_weight=1.0,
        adversarial_weight=0.1,
        perceptual_weight=0.1,
        style_weight=250.0
    ):
        super().__init__()
        
        self.l1_weight = l1_weight
        self.adversarial_weight = adversarial_weight
        self.perceptual_weight = perceptual_weight
        self.style_weight = style_weight
        
        self.reconstruction_loss = ReconstructionLoss()
        self.perceptual_loss = PerceptualLoss()
        self.style_loss = StyleLoss()
        self.adversarial_loss = AdversarialLoss(loss_type='hinge')
    
    def generator_loss(self, output, target, mask, disc_fake):
        """
        Compute generator loss.
        
        Args:
            output: Generated image
            target: Ground truth image
            mask: Binary mask
            disc_fake: Discriminator output for fake image
        """
        # Reconstruction
        l1_loss = self.reconstruction_loss(output, target, mask)
        
        # Perceptual
        perc_loss = self.perceptual_loss(output, target)
        
        # Style
        style_loss = self.style_loss(output, target)
        
        # Adversarial (generator wants discriminator to think fake is real)
        adv_loss = self.adversarial_loss(disc_fake, target_is_real=True)
        
        # Total
        total_loss = (
            self.l1_weight * l1_loss +
            self.perceptual_weight * perc_loss +
            self.style_weight * style_loss +
            self.adversarial_weight * adv_loss
        )
        
        return {
            'total': total_loss,
            'l1': l1_loss,
            'perceptual': perc_loss,
            'style': style_loss,
            'adversarial': adv_loss
        }
    
    def discriminator_loss(self, disc_real, disc_fake):
        """
        Compute discriminator loss.
        
        Args:
            disc_real: Discriminator output for real image
            disc_fake: Discriminator output for fake image
        """
        real_loss = self.adversarial_loss(disc_real, target_is_real=True)
        fake_loss = self.adversarial_loss(disc_fake, target_is_real=False)
        
        return {
            'total': real_loss + fake_loss,
            'real': real_loss,
            'fake': fake_loss
        }


if __name__ == "__main__":
    # Test losses
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    loss_fn = InpaintingLoss().to(device)
    
    # Dummy data
    output = torch.randn(2, 3, 256, 256).to(device)
    target = torch.randn(2, 3, 256, 256).to(device)
    mask = torch.ones(2, 1, 256, 256).to(device)
    mask[:, :, 64:192, 64:192] = 0
    
    disc_fake = torch.randn(2, 1, 30, 30).to(device)
    disc_real = torch.randn(2, 1, 30, 30).to(device)
    
    g_loss = loss_fn.generator_loss(output, target, mask, disc_fake)
    d_loss = loss_fn.discriminator_loss(disc_real, disc_fake)
    
    print("Generator losses:")
    for k, v in g_loss.items():
        print(f"  {k}: {v.item():.4f}")
    
    print("\nDiscriminator losses:")
    for k, v in d_loss.items():
        print(f"  {k}: {v.item():.4f}")
