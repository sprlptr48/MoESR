from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar


@dataclass
class MoESRConfig:
    """Configuration for the MoESR architecture."""

    embed_dim: int = 192
    window_size: int = 16
    num_heads: int = 8
    num_transformer_blocks: int = 28
    num_experts: int = 8
    experts_per_token: int = 2
    expert_capacity_factor: float = 1.25
    mlp_ratio: float = 4.0
    scale_factor: int = 4
    use_overlap_attention: bool = True
    overlap_window_size: int = 32
    use_channel_attention: bool = True
    auxiliary_loss_coeff: float = 0.01
    dropout: float = 0.0
    upsampler: str = "pixelshuffle"

    image_channels: int = 3
    norm_eps: float = 1e-6
    attention_dropout: float = 0.0
    proj_dropout: float = 0.0
    router_jitter_noise: float = 0.0
    stage_scale_factor: int = 2
    channel_attention_reduction: int = 16
    perceptual_layer: str = "relu3_4"
    ssim_kernel_size: int = 11
    ssim_sigma: float = 1.5
    PRESET_NAMES: ClassVar[tuple[str, ...]] = ("default", "debug_small", "base_large", "moe_1b")

    def __post_init__(self) -> None:
        if self.num_transformer_blocks % 2 != 0:
            raise ValueError("num_transformer_blocks must be divisible by 2.")
        if self.scale_factor not in {2, 4, 8}:
            raise ValueError("scale_factor must be one of {2, 4, 8}.")
        if self.experts_per_token < 1 or self.experts_per_token > self.num_experts:
            raise ValueError("experts_per_token must be in [1, num_experts].")
        if self.upsampler not in {"pixelshuffle", "nearest+conv"}:
            raise ValueError("upsampler must be 'pixelshuffle' or 'nearest+conv'.")
        if self.embed_dim % self.num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads.")

    @classmethod
    def debug_small(cls) -> "MoESRConfig":
        """Small preset for local debugging."""

        return cls(
            embed_dim=96,
            num_heads=4,
            num_transformer_blocks=8,
            num_experts=4,
            experts_per_token=2,
            mlp_ratio=3.0,
        )

    @classmethod
    def base_large(cls) -> "MoESRConfig":
        """Larger practical preset below the 1B target."""

        return cls(
            embed_dim=384,
            num_heads=6,
            num_transformer_blocks=24,
            num_experts=8,
            experts_per_token=2,
            mlp_ratio=4.0,
        )

    @classmethod
    def moe_1b(cls) -> "MoESRConfig":
        """1B-class preset for the current implementation."""

        return cls(
            embed_dim=512,
            num_heads=8,
            num_transformer_blocks=28,
            num_experts=12,
            experts_per_token=2,
            mlp_ratio=4.5,
        )

    @classmethod
    def from_preset(cls, name: str) -> "MoESRConfig":
        """Build a config from a named preset."""

        normalized = name.lower()
        if normalized == "default":
            return cls()
        if normalized == "debug_small":
            return cls.debug_small()
        if normalized == "base_large":
            return cls.base_large()
        if normalized == "moe_1b":
            return cls.moe_1b()
        raise ValueError(f"Unknown config preset: {name}")
