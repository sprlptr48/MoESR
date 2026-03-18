from __future__ import annotations

from torch import Tensor, nn

from .attention import OverlapCrossAttention, WindowAttention
from .config import MoESRConfig
from .moe import MoELayer


class SRTransformerBlock(nn.Module):
    """Hybrid SR transformer block with attention and MoE FFN.

    Input:
        x: [B, H, W, C]
    Output:
        out: [B, H, W, C]
    """

    def __init__(self, config: MoESRConfig, block_idx: int) -> None:
        super().__init__()
        shift = config.window_size // 2 if block_idx % 2 == 1 else 0
        self.norm1 = nn.LayerNorm(config.embed_dim, eps=config.norm_eps)
        self.attn = WindowAttention(config, shift_size=shift)
        self.use_overlap_attention = config.use_overlap_attention and block_idx % 4 == 0
        if self.use_overlap_attention:
            self.norm2 = nn.LayerNorm(config.embed_dim, eps=config.norm_eps)
            self.overlap_attn = OverlapCrossAttention(config)
        else:
            self.norm2 = None
            self.overlap_attn = None
        self.norm3 = nn.LayerNorm(config.embed_dim, eps=config.norm_eps)
        self.moe = MoELayer(config)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        x = x + self.attn(self.norm1(x))
        if self.use_overlap_attention and self.norm2 is not None and self.overlap_attn is not None:
            x = x + self.overlap_attn(self.norm2(x))
        b, h, w, c = x.shape
        moe_out = self.moe(self.norm3(x).view(b, h * w, c)).view(b, h, w, c)
        x = x + moe_out
        return x, self.moe.aux_loss

