from __future__ import annotations

from typing import Dict

import torch
from torch import Tensor, nn

from .config import MoESRConfig
from .stages import SRStage


class MoESR(nn.Module):
    """Two-stage progressive MoE super-resolution network.

    Input:
        x: [B, 3, H, W]
    Output:
        dict with:
            output: [B, 3, 4H, 4W]
            stage1_output: [B, 3, 2H, 2W]
            aux_loss: []
    """

    def __init__(self, config: MoESRConfig) -> None:
        super().__init__()
        self.config = config
        self.shallow_extractor = nn.Conv2d(config.image_channels, config.embed_dim, kernel_size=3, padding=1)
        blocks_per_stage = config.num_transformer_blocks // 2
        self.stage1 = SRStage(config, num_blocks=blocks_per_stage, block_offset=0, scale=2)
        self.stage2 = SRStage(config, num_blocks=blocks_per_stage, block_offset=blocks_per_stage, scale=2)
        self.final_reconstruction = nn.Conv2d(config.embed_dim, config.image_channels, kernel_size=3, padding=1)

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        x_feat = self.shallow_extractor(x)
        stage1_output, feat1, aux1 = self.stage1(x_feat, x_lr=x)
        stage2_output, feat2, aux2 = self.stage2(feat1, x_lr=stage1_output)
        final_sr = self.final_reconstruction(feat2) + stage2_output
        total_aux = (aux1 + aux2) / 2.0
        return {
            "output": final_sr,
            "stage1_output": stage1_output,
            "aux_loss": total_aux,
        }

    def set_gradient_checkpointing(self, enabled: bool) -> None:
        """Enable gradient checkpointing in both SR stages."""

        self.stage1.set_gradient_checkpointing(enabled)
        self.stage2.set_gradient_checkpointing(enabled)


MoeSR = MoESR


if __name__ == "__main__":
    cfg = MoESRConfig()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = MoESR(cfg).to(device)
    x = torch.randn(1, 3, 64, 64, device=device)
    out = model(x)
    assert out["output"].shape == (1, 3, 256, 256)
    assert out["stage1_output"].shape == (1, 3, 128, 128)
    print("Forward pass OK. Output shape:", tuple(out["output"].shape))
    print("Aux loss:", float(out["aux_loss"].detach().item()))
