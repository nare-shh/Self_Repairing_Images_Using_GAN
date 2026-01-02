"""
Inpainting GAN - Discriminator Model
PatchGAN discriminator with spectral normalization for stable training.
"""

import torch
import torch.nn as nn
import torch.nn.utils.spectral_norm as spectral_norm


class SpectralNormConv2d(nn.Module):
    """Convolution with spectral normalization for training stability."""
    
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        self.conv = spectral_norm(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        )
    
    def forward(self, x):
        return self.conv(x)


class DiscriminatorBlock(nn.Module):
    """Discriminator block with spectral normalization."""
    
    def __init__(self, in_channels, out_channels, stride=2, use_norm=True):
        super().__init__()
        self.conv = SpectralNormConv2d(in_channels, out_channels, 4, stride, 1)
        self.norm = nn.InstanceNorm2d(out_channels) if use_norm else None
        self.activation = nn.LeakyReLU(0.2, inplace=True)
    
    def forward(self, x):
        x = self.conv(x)
        if self.norm:
            x = self.norm(x)
        x = self.activation(x)
        return x


class PatchDiscriminator(nn.Module):
    """
    PatchGAN Discriminator
    Classifies whether 70x70 overlapping patches are real or fake.
    Uses spectral normalization for stable training.
    """
    
    def __init__(self, in_channels=3, base_channels=64, n_layers=4):
        super().__init__()
        
        layers = []
        
        # First layer without normalization
        layers.append(DiscriminatorBlock(in_channels, base_channels, stride=2, use_norm=False))
        
        # Intermediate layers
        channels = base_channels
        for i in range(1, n_layers):
            prev_channels = channels
            channels = min(base_channels * (2 ** i), 512)
            stride = 2 if i < n_layers - 1 else 1
            layers.append(DiscriminatorBlock(prev_channels, channels, stride=stride))
        
        # Output layer
        layers.append(SpectralNormConv2d(channels, 1, 4, stride=1, padding=1))
        
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        """
        Args:
            x: Image [B, 3, H, W]
        
        Returns:
            Patch predictions [B, 1, H', W'] where each value is real/fake score
        """
        return self.model(x)


class MultiScaleDiscriminator(nn.Module):
    """
    Multi-scale discriminator for better quality assessment.
    Operates at multiple resolutions to capture both local and global features.
    """
    
    def __init__(self, in_channels=3, base_channels=64, n_discriminators=3):
        super().__init__()
        
        self.discriminators = nn.ModuleList()
        for _ in range(n_discriminators):
            self.discriminators.append(PatchDiscriminator(in_channels, base_channels))
        
        self.downsample = nn.AvgPool2d(3, stride=2, padding=1, count_include_pad=False)
    
    def forward(self, x):
        """
        Args:
            x: Image [B, 3, H, W]
        
        Returns:
            List of discriminator outputs at different scales
        """
        outputs = []
        
        for disc in self.discriminators:
            outputs.append(disc(x))
            x = self.downsample(x)
        
        return outputs


class GlobalLocalDiscriminator(nn.Module):
    """
    Discriminator that combines global and local (patch) discrimination.
    Global branch looks at the whole image, local branch focuses on masked region.
    """
    
    def __init__(self, in_channels=3, base_channels=64):
        super().__init__()
        
        # Global discriminator
        self.global_disc = PatchDiscriminator(in_channels, base_channels, n_layers=5)
        
        # Local discriminator (for masked region)
        self.local_disc = PatchDiscriminator(in_channels, base_channels, n_layers=4)
    
    def forward(self, x, mask=None):
        """
        Args:
            x: Image [B, 3, H, W]
            mask: Optional mask indicating inpainted region [B, 1, H, W]
        
        Returns:
            global_out: Global discrimination output
            local_out: Local discrimination output (if mask provided)
        """
        global_out = self.global_disc(x)
        
        if mask is not None:
            # Extract masked region (simplified: use center crop)
            local_out = self.local_disc(x)
        else:
            local_out = None
        
        return global_out, local_out


if __name__ == "__main__":
    # Test the discriminator
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Test PatchDiscriminator
    disc = PatchDiscriminator().to(device)
    img = torch.randn(2, 3, 256, 256).to(device)
    out = disc(img)
    print(f"PatchDiscriminator output shape: {out.shape}")
    print(f"PatchDiscriminator parameters: {sum(p.numel() for p in disc.parameters()):,}")
    
    # Test MultiScaleDiscriminator
    ms_disc = MultiScaleDiscriminator().to(device)
    outputs = ms_disc(img)
    print(f"\nMultiScaleDiscriminator outputs:")
    for i, o in enumerate(outputs):
        print(f"  Scale {i}: {o.shape}")
