from __future__ import annotations

from typing import Dict, Optional

import torch
from torch import Tensor, nn
import torch.nn.functional as F
from torchvision.models import VGG19_Weights, vgg19

from moesr.models.config import MoESRConfig


class SSIMLoss(nn.Module):
    """SSIM loss for BCHW RGB tensors in [0, 1].

    Input:
        pred: [B, 3, H, W]
        target: [B, 3, H, W]
    Output:
        loss: []
    """

    def __init__(self, kernel_size: int = 11, sigma: float = 1.5) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.sigma = sigma
        self.register_buffer("kernel", self._build_kernel(kernel_size, sigma), persistent=False)

    @staticmethod
    def _build_kernel(kernel_size: int, sigma: float) -> Tensor:
        coords = torch.arange(kernel_size, dtype=torch.float32) - kernel_size // 2
        gauss = torch.exp(-(coords**2) / (2 * sigma**2))
        gauss = gauss / gauss.sum()
        kernel_2d = torch.outer(gauss, gauss)
        return kernel_2d.unsqueeze(0).unsqueeze(0)

    def _filter(self, x: Tensor) -> Tensor:
        channels = x.shape[1]
        kernel = self.kernel.expand(channels, 1, self.kernel_size, self.kernel_size)
        return F.conv2d(x, kernel, padding=self.kernel_size // 2, groups=channels)

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        c1 = 0.01**2
        c2 = 0.03**2
        mu_x = self._filter(pred)
        mu_y = self._filter(target)
        sigma_x = self._filter(pred * pred) - mu_x.square()
        sigma_y = self._filter(target * target) - mu_y.square()
        sigma_xy = self._filter(pred * target) - mu_x * mu_y
        ssim_map = ((2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)) / (
            (mu_x.square() + mu_y.square() + c1) * (sigma_x + sigma_y + c2)
        )
        return 1.0 - ssim_map.mean()


class VGGPerceptualLoss(nn.Module):
    """Perceptual loss using VGG19 relu3_4 features.

    Input:
        pred: [B, 3, H, W]
        target: [B, 3, H, W]
    Output:
        loss: []
    """

    def __init__(self) -> None:
        super().__init__()
        try:
            weights = VGG19_Weights.IMAGENET1K_V1
        except Exception:
            weights = None
        features = vgg19(weights=weights).features[:18]
        for param in features.parameters():
            param.requires_grad = False
        self.features = features.eval()
        self.register_buffer(
            "mean",
            torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "std",
            torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1),
            persistent=False,
        )

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        pred = (pred - self.mean) / self.std
        target = (target - self.mean) / self.std
        pred_feat = self.features(pred)
        target_feat = self.features(target)
        return F.l1_loss(pred_feat, target_feat)


class MoESRLoss(nn.Module):
    """Composite loss for progressive MoESR supervision.

    Input:
        outputs["output"]: [B, 3, 4H, 4W]
        outputs["stage1_output"]: [B, 3, 2H, 2W]
        target_x4: [B, 3, 4H, 4W]
        target_x2: [B, 3, 2H, 2W]
    Output:
        Dict[str, Tensor]
    """

    def __init__(self, config: MoESRConfig, use_perceptual: bool = True) -> None:
        super().__init__()
        self.config = config
        self.l1 = nn.L1Loss()
        self.ssim = SSIMLoss(config.ssim_kernel_size, config.ssim_sigma)
        self.perceptual = VGGPerceptualLoss() if use_perceptual else None

    def forward(
        self,
        outputs: Dict[str, Tensor],
        target_x4: Tensor,
        target_x2: Tensor,
        aux_loss: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        final_pixel = self.l1(outputs["output"], target_x4)
        stage1_pixel = self.l1(outputs["stage1_output"], target_x2) * 0.4
        ssim_loss = self.ssim(outputs["output"], target_x4) * 0.1
        perceptual_loss = (
            self.perceptual(outputs["output"], target_x4) * 0.05
            if self.perceptual is not None
            else target_x4.new_zeros(())
        )
        aux_term = (aux_loss if aux_loss is not None else outputs["aux_loss"]) * self.config.auxiliary_loss_coeff
        total = final_pixel + stage1_pixel + ssim_loss + perceptual_loss + aux_term
        return {
            "loss": total,
            "pixel_loss": final_pixel + stage1_pixel,
            "final_l1": final_pixel,
            "stage1_l1": stage1_pixel,
            "ssim_loss": ssim_loss,
            "perceptual_loss": perceptual_loss,
            "aux_loss": aux_term,
        }

