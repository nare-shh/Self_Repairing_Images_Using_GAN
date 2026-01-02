"""
Inpainting GAN - Generator Model
U-Net style generator with partial convolutions for image inpainting.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PartialConv2d(nn.Module):
    """
    Partial Convolution layer for mask-aware convolutions.
    Only convolves over valid (non-masked) pixels and updates the mask.
    """
    
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)
        self.mask_conv = nn.Conv2d(1, 1, kernel_size, stride, padding, bias=False)
        
        # Initialize mask conv weights to 1
        nn.init.constant_(self.mask_conv.weight, 1.0)
        
        # Freeze mask conv weights
        for param in self.mask_conv.parameters():
            param.requires_grad = False
    
    def forward(self, x, mask):
        """
        Args:
            x: Input tensor [B, C, H, W]
            mask: Binary mask [B, 1, H, W] where 1 = valid, 0 = hole
        """
        # Apply mask to input
        x_masked = x * mask
        
        # Regular convolution on masked input
        output = self.conv(x_masked)
        
        # Calculate mask update
        with torch.no_grad():
            mask_sum = self.mask_conv(mask)
            # Avoid division by zero
            mask_sum = torch.clamp(mask_sum, min=1e-8)
        
        # Normalize by the number of valid pixels
        kernel_size = self.conv.kernel_size[0] * self.conv.kernel_size[1]
        output = output * (kernel_size / mask_sum)
        
        # Update mask: where any input was valid, output is valid
        new_mask = (mask_sum > 0).float()
        
        return output, new_mask


class GatedConv2d(nn.Module):
    """
    Gated Convolution for better inpainting.
    Uses a learned gating mechanism to control information flow.
    """
    
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        self.conv_features = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.conv_gate = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.norm = nn.InstanceNorm2d(out_channels)
    
    def forward(self, x):
        features = self.conv_features(x)
        gate = torch.sigmoid(self.conv_gate(x))
        return self.norm(features) * gate


class EncoderBlock(nn.Module):
    """Encoder block with partial convolution and downsampling."""
    
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
    """Decoder block with upsampling and skip connections."""
    
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
    """
    U-Net style generator for image inpainting.
    Takes incomplete image and mask, outputs completed image.
    
    Architecture:
    - Encoder: 7 blocks, each halves spatial dimensions
    - Decoder: 7 blocks with skip connections, each doubles spatial dimensions
    """
    
    def __init__(self, in_channels=4, out_channels=3, base_channels=64):
        super().__init__()
        
        # Input: image (3) + mask (1) = 4 channels
        # Encoder
        self.enc1 = EncoderBlock(in_channels, base_channels, use_partial=True)      # 256 -> 128
        self.enc2 = EncoderBlock(base_channels, base_channels*2, use_partial=True)  # 128 -> 64
        self.enc3 = EncoderBlock(base_channels*2, base_channels*4, use_partial=True) # 64 -> 32
        self.enc4 = EncoderBlock(base_channels*4, base_channels*8, use_partial=True) # 32 -> 16
        self.enc5 = EncoderBlock(base_channels*8, base_channels*8, use_partial=True) # 16 -> 8
        self.enc6 = EncoderBlock(base_channels*8, base_channels*8, use_partial=True) # 8 -> 4
        self.enc7 = EncoderBlock(base_channels*8, base_channels*8, use_partial=True) # 4 -> 2
        
        # Decoder (with skip connections)
        # Channel counts after each decoder + skip:
        # dec7: 512 -> 512, + e6(512) = 1024
        # dec6: 1024 -> 512, + e5(512) = 1024
        # dec5: 1024 -> 512, + e4(512) = 1024
        # dec4: 1024 -> 256, + e3(256) = 512
        # dec3: 512 -> 128, + e2(128) = 256
        # dec2: 256 -> 64, + e1(64) = 128
        # dec1: 128 -> 64, + input(4) = 68
        self.dec7 = DecoderBlock(base_channels*8, base_channels*8, use_dropout=True)   # 512 -> 512
        self.dec6 = DecoderBlock(base_channels*16, base_channels*8, use_dropout=True)  # 1024 -> 512
        self.dec5 = DecoderBlock(base_channels*16, base_channels*8, use_dropout=True)  # 1024 -> 512
        self.dec4 = DecoderBlock(base_channels*16, base_channels*4)                     # 1024 -> 256
        self.dec3 = DecoderBlock(base_channels*8, base_channels*2)                      # 512 -> 128
        self.dec2 = DecoderBlock(base_channels*4, base_channels)                        # 256 -> 64
        self.dec1 = DecoderBlock(base_channels*2, base_channels)                        # 128 -> 64
        
        # Output layer (64 + 4 = 68 channels from dec1 + input_combined)
        self.output = nn.Sequential(
            nn.Conv2d(base_channels + in_channels, out_channels, 3, padding=1),
            nn.Tanh()
        )
    
    def forward(self, x, mask):
        """
        Args:
            x: Incomplete image [B, 3, H, W]
            mask: Binary mask [B, 1, H, W] where 1 = valid, 0 = hole
        
        Returns:
            Completed image [B, 3, H, W]
        """
        # Concatenate image and mask
        input_combined = torch.cat([x * mask, mask], dim=1)
        
        # Encoder
        e1, m1 = self.enc1(input_combined, mask)
        e2, m2 = self.enc2(e1, m1)
        e3, m3 = self.enc3(e2, m2)
        e4, m4 = self.enc4(e3, m3)
        e5, m5 = self.enc5(e4, m4)
        e6, m6 = self.enc6(e5, m5)
        e7, _ = self.enc7(e6, m6)
        
        # Decoder with skip connections
        d7 = self.dec7(e7, e6)
        d6 = self.dec6(d7, e5)
        d5 = self.dec5(d6, e4)
        d4 = self.dec4(d5, e3)
        d3 = self.dec3(d4, e2)
        d2 = self.dec2(d3, e1)
        d1 = self.dec1(d2, input_combined)
        
        # Output
        output = self.output(d1)
        
        # Combine: keep original pixels where valid, use generated for holes
        completed = x * mask + output * (1 - mask)
        
        return completed


if __name__ == "__main__":
    # Test the generator
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    generator = InpaintingGenerator().to(device)
    
    # Create dummy inputs
    batch_size = 2
    img = torch.randn(batch_size, 3, 256, 256).to(device)
    mask = torch.ones(batch_size, 1, 256, 256).to(device)
    mask[:, :, 64:192, 64:192] = 0  # Create a hole in the center
    
    # Forward pass
    output = generator(img, mask)
    
    print(f"Input shape: {img.shape}")
    print(f"Mask shape: {mask.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Generator parameters: {sum(p.numel() for p in generator.parameters()):,}")
