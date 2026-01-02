"""
Models package for Inpainting GAN
"""

from .generator import InpaintingGenerator
from .discriminator import PatchDiscriminator, MultiScaleDiscriminator

__all__ = ['InpaintingGenerator', 'PatchDiscriminator', 'MultiScaleDiscriminator']
