from __future__ import annotations

from typing import Tuple

import torch
from torch import Tensor, nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .config import MoESRConfig
from .transformer_block import SRTransformerBlock


class PixelShuffleUpsampler(nn.Module):
    """Learned x2 upsampler for BCHW features.

    Input:
        x: [B, C, H, W]
    Output:
        out: [B, C, 2H, 2W]
    """

    def __init__(self, channels: int, scale: int) -> None:
        super().__init__()
        self.proj = nn.Conv2d(channels, channels * (scale**2), kernel_size=3, padding=1)
        self.shuffle = nn.PixelShuffle(scale)

    def forward(self, x: Tensor) -> Tensor:
        return self.shuffle(self.proj(x))


class NearestConvUpsampler(nn.Module):
    """Nearest-neighbor plus convolution upsampler for BCHW features.

    Input:
        x: [B, C, H, W]
    Output:
        out: [B, C, 2H, 2W]
    """

    def __init__(self, channels: int, scale: int) -> None:
        super().__init__()
        self.scale = scale
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x: Tensor) -> Tensor:
        x = F.interpolate(x, scale_factor=self.scale, mode="nearest")
        return self.conv(x)


class SRStage(nn.Module):
    """Single x2 upscaling stage with transformer trunk and residual upsampling.

    Input:
        x: [B, C, H, W]
        x_lr: [B, 3, H, W]
    Output:
        sr: [B, 3, 2H, 2W]
        features: [B, C, 2H, 2W]
        aux_loss: []
    """

    def __init__(self, config: MoESRConfig, num_blocks: int, block_offset: int, scale: int = 2) -> None:
        super().__init__()
        self.gradient_checkpointing = False
        self.embed_proj = nn.Conv2d(config.embed_dim, config.embed_dim, kernel_size=3, padding=1)
        self.blocks = nn.ModuleList(
            [SRTransformerBlock(config, block_idx=block_offset + idx) for idx in range(num_blocks)]
        )
        if config.upsampler == "pixelshuffle":
            self.upsampler = PixelShuffleUpsampler(config.embed_dim, scale)
        else:
            self.upsampler = NearestConvUpsampler(config.embed_dim, scale)
        self.to_rgb = nn.Conv2d(config.embed_dim, config.image_channels, kernel_size=3, padding=1)

    def set_gradient_checkpointing(self, enabled: bool) -> None:
        """Enable activation checkpointing for transformer blocks."""

        self.gradient_checkpointing = enabled

    def forward(self, x: Tensor, x_lr: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        feat = self.embed_proj(x)
        feat = feat.permute(0, 2, 3, 1).contiguous()
        aux_loss = feat.new_zeros(())
        num_blocks = max(len(self.blocks), 1)
        for block in self.blocks:
            if self.gradient_checkpointing and self.training:
                feat, block_aux = checkpoint(block, feat, use_reentrant=False)
            else:
                feat, block_aux = block(feat)
            aux_loss = aux_loss + block_aux
        aux_loss = aux_loss / num_blocks
        feat = feat.permute(0, 3, 1, 2).contiguous()
        feat_up = self.upsampler(feat)
        sr = self.to_rgb(feat_up)
        residual = F.interpolate(x_lr, scale_factor=2, mode="bicubic", align_corners=False)
        sr = sr + residual
        return sr, feat_up, aux_loss
