from __future__ import annotations

import math
import argparse
import gc
from dataclasses import asdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import torch
from PIL import Image
from torch import Tensor
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import functional as TF

from moesr.losses.loss import MoESRLoss
from moesr.infer import tiled_inference
from moesr.models.config import MoESRConfig
from moesr.models.moesr import MoESR
from moesr.utils.expert_monitor import ExpertUtilizationMonitor


@dataclass
class TrainerConfig:
    """Training hyperparameters."""

    train_lr_dir: str
    train_hr_dir: str
    val_lr_dir: str
    val_hr_dir: str
    output_dir: str = "runs/moesr"
    batch_size: int = 4
    num_workers: int = 4
    patch_size: int = 64
    max_steps: int = 500_000
    warmup_steps: int = 5_000
    grad_accum_steps: int = 4
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    grad_clip_norm: float = 1.0
    checkpoint_interval: int = 100
    val_interval: int = 2_000
    val_max_images: int = 0
    val_tile_size: int = 0
    val_tile_overlap: int = 16
    log_interval: int = 100
    compile_model: bool = False
    mixed_precision: bool = True


class PairedImageDataset(Dataset):
    """Folder-based paired SR dataset.

    Output:
        dict with lr [3, h, w], hr_x2 [3, 2h, 2w], hr [3, scale*h, scale*w]
    """

    def __init__(self, lr_dir: str, hr_dir: str, scale_factor: int, patch_size: int, train: bool) -> None:
        super().__init__()
        self.lr_dir = Path(lr_dir)
        self.hr_dir = Path(hr_dir)
        self.lr_paths = sorted(path for path in self.lr_dir.glob("*") if path.is_file())
        self.hr_paths = sorted(path for path in self.hr_dir.glob("*") if path.is_file())
        if not self.lr_dir.exists():
            raise FileNotFoundError(f"LR directory does not exist: {self.lr_dir}")
        if not self.hr_dir.exists():
            raise FileNotFoundError(f"HR directory does not exist: {self.hr_dir}")
        if len(self.lr_paths) == 0:
            raise ValueError(f"No LR images found in: {self.lr_dir}")
        if len(self.hr_paths) == 0:
            raise ValueError(f"No HR images found in: {self.hr_dir}")
        if len(self.lr_paths) != len(self.hr_paths):
            raise ValueError("LR and HR directories must contain the same number of files.")
        self.scale_factor = scale_factor
        self.patch_size = patch_size
        self.train = train

    def __len__(self) -> int:
        return len(self.lr_paths)

    def _load_image(self, path: Path) -> Tensor:
        image = Image.open(path).convert("RGB")
        return TF.to_tensor(image)

    def _random_crop(self, lr: Tensor, hr: Tensor) -> tuple[Tensor, Tensor]:
        _, h, w = lr.shape
        if h < self.patch_size or w < self.patch_size:
            raise ValueError("Patch size exceeds LR image size.")
        top = torch.randint(0, h - self.patch_size + 1, (1,)).item()
        left = torch.randint(0, w - self.patch_size + 1, (1,)).item()
        hr_top = top * self.scale_factor
        hr_left = left * self.scale_factor
        lr_patch = lr[:, top : top + self.patch_size, left : left + self.patch_size]
        hr_patch = hr[
            :,
            hr_top : hr_top + self.patch_size * self.scale_factor,
            hr_left : hr_left + self.patch_size * self.scale_factor,
        ]
        return lr_patch, hr_patch

    def _augment(self, lr: Tensor, hr: Tensor) -> tuple[Tensor, Tensor]:
        if torch.rand(()) < 0.5:
            lr = torch.flip(lr, dims=[2])
            hr = torch.flip(hr, dims=[2])
        if torch.rand(()) < 0.5:
            lr = torch.flip(lr, dims=[1])
            hr = torch.flip(hr, dims=[1])
        rotations = torch.randint(0, 4, (1,)).item()
        lr = torch.rot90(lr, rotations, dims=[1, 2])
        hr = torch.rot90(hr, rotations, dims=[1, 2])
        return lr, hr

    def __getitem__(self, index: int) -> Dict[str, Tensor]:
        lr = self._load_image(self.lr_paths[index])
        hr = self._load_image(self.hr_paths[index])
        if self.train:
            lr, hr = self._random_crop(lr, hr)
            lr, hr = self._augment(lr, hr)
        hr_x2 = TF.resize(hr, [lr.shape[1] * 2, lr.shape[2] * 2], antialias=True)
        return {"lr": lr, "hr": hr, "hr_x2": hr_x2}


def build_dataloader(
    lr_dir: str,
    hr_dir: str,
    model_config: MoESRConfig,
    trainer_config: TrainerConfig,
    train: bool,
) -> DataLoader:
    """Create a dataloader for paired SR images."""

    dataset = PairedImageDataset(
        lr_dir=lr_dir,
        hr_dir=hr_dir,
        scale_factor=model_config.scale_factor,
        patch_size=trainer_config.patch_size,
        train=train,
    )
    return DataLoader(
        dataset,
        batch_size=trainer_config.batch_size if train else 1,
        shuffle=train,
        num_workers=trainer_config.num_workers,
        pin_memory=True,
        drop_last=train,
    )


def build_cosine_schedule(optimizer: torch.optim.Optimizer, warmup_steps: int, total_steps: int) -> LambdaLR:
    """Cosine learning-rate schedule with linear warmup."""

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step + 1) / float(max(1, warmup_steps))
        progress = (step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return LambdaLR(optimizer, lr_lambda=lr_lambda)


def compute_psnr(pred: Tensor, target: Tensor) -> Tensor:
    """Compute PSNR for BCHW images in [0, 1]."""

    mse = torch.mean((pred - target) ** 2)
    return -10.0 * torch.log10(mse.clamp_min(1e-8))


class MoESRTrainer:
    """Training loop scaffold for MoESR."""

    def __init__(
        self,
        model: MoESR,
        model_config: MoESRConfig,
        trainer_config: TrainerConfig,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: torch.device,
    ) -> None:
        self.device = device
        self.config = trainer_config
        self.model = model.to(device)
        if trainer_config.compile_model and hasattr(torch, "compile"):
            self.model = torch.compile(self.model, mode="reduce-overhead")
        self.criterion = MoESRLoss(model_config).to(device)
        self.val_l1 = torch.nn.L1Loss().to(device)
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=trainer_config.learning_rate,
            weight_decay=trainer_config.weight_decay,
        )
        self.scheduler = build_cosine_schedule(
            self.optimizer,
            warmup_steps=trainer_config.warmup_steps,
            total_steps=trainer_config.max_steps,
        )
        scaler_device = "cuda" if device.type == "cuda" else "cpu"
        self.scaler = torch.amp.GradScaler(
            scaler_device,
            enabled=trainer_config.mixed_precision and device.type == "cuda",
        )
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.output_dir = Path(trainer_config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.monitor = ExpertUtilizationMonitor(self.model)
        self.best_psnr = float("-inf")
        self.global_step = 0

    def _memory_log(self) -> str:
        """Return a compact CUDA memory usage string."""

        if self.device.type != "cuda":
            return "mem_alloc=0.00GB mem_reserved=0.00GB mem_peak=0.00GB"
        allocated = torch.cuda.memory_allocated(self.device) / (1024**3)
        reserved = torch.cuda.memory_reserved(self.device) / (1024**3)
        peak = torch.cuda.max_memory_allocated(self.device) / (1024**3)
        return f"mem_alloc={allocated:.2f}GB mem_reserved={reserved:.2f}GB mem_peak={peak:.2f}GB"

    def load_checkpoint(self, checkpoint_path: str) -> None:
        """Resume training state from a checkpoint."""

        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.scheduler.load_state_dict(checkpoint["scheduler"])
        self.scaler.load_state_dict(checkpoint["scaler"])
        self.global_step = int(checkpoint.get("global_step", 0))
        self.best_psnr = float(checkpoint.get("best_psnr", float("-inf")))

    def _move_batch(self, batch: Dict[str, Tensor]) -> Dict[str, Tensor]:
        return {key: value.to(self.device, non_blocking=True) for key, value in batch.items()}

    def save_checkpoint(self, name: str) -> None:
        """Save a training checkpoint."""

        checkpoint = {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "scaler": self.scaler.state_dict(),
            "global_step": self.global_step,
            "best_psnr": self.best_psnr,
            "model_config": asdict(self.model.config),
            "trainer_config": asdict(self.config),
        }
        torch.save(checkpoint, self.output_dir / name)

    def validate(self) -> Dict[str, float]:
        """Run validation and return averaged metrics."""

        self.optimizer.zero_grad(set_to_none=True)
        if self.device.type == "cuda":
            gc.collect()
            torch.cuda.empty_cache()
        self.model.eval()
        psnr_values = []
        loss_values = []
        max_images = self.config.val_max_images if self.config.val_max_images > 0 else None
        with torch.inference_mode():
            for batch_idx, batch in enumerate(self.val_loader):
                if max_images is not None and batch_idx >= max_images:
                    break
                batch = self._move_batch(batch)
                with torch.amp.autocast(device_type=self.device.type, enabled=False):
                    if self.config.val_tile_size > 0:
                        outputs = tiled_inference(
                            self.model,
                            batch["lr"],
                            tile_size=self.config.val_tile_size,
                            use_half=False,
                            tile_overlap=self.config.val_tile_overlap,
                        )
                    else:
                        outputs = self.model(batch["lr"])
                    losses = self.val_l1(outputs["output"].float(), batch["hr"].float())
                psnr_values.append(compute_psnr(outputs["output"], batch["hr"]).item())
                loss_values.append(losses.item())
        avg_psnr = sum(psnr_values) / max(1, len(psnr_values))
        avg_loss = sum(loss_values) / max(1, len(loss_values))
        self.model.train()
        if self.device.type == "cuda":
            gc.collect()
            torch.cuda.empty_cache()
        return {"psnr": avg_psnr, "loss": avg_loss}

    def fit(self) -> None:
        """Train the model until max_steps."""

        self.model.train()
        train_iter = iter(self.train_loader)
        while self.global_step < self.config.max_steps:
            if self.device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(self.device)
            self.optimizer.zero_grad(set_to_none=True)
            accum_logs: Dict[str, float] = {}
            for _ in range(self.config.grad_accum_steps):
                try:
                    batch = next(train_iter)
                except StopIteration:
                    train_iter = iter(self.train_loader)
                    batch = next(train_iter)
                batch = self._move_batch(batch)
                with torch.amp.autocast(
                    device_type=self.device.type,
                    enabled=self.config.mixed_precision and self.device.type == "cuda",
                ):
                    outputs = self.model(batch["lr"])
                    losses = self.criterion(outputs, batch["hr"], batch["hr_x2"])
                    loss = losses["loss"] / self.config.grad_accum_steps
                self.scaler.scale(loss).backward()
                self.monitor.update()
                for key, value in losses.items():
                    accum_logs[key] = accum_logs.get(key, 0.0) + float(value.item())

            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip_norm)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.scheduler.step()
            self.global_step += 1

            if self.global_step % self.config.log_interval == 0:
                utilization = self.monitor.report(reset=True)
                avg_logs = {k: v / self.config.grad_accum_steps for k, v in accum_logs.items()}
                print(
                    f"step={self.global_step} "
                    f"loss={avg_logs['loss']:.4f} "
                    f"pixel={avg_logs['pixel_loss']:.4f} "
                    f"perc={avg_logs['perceptual_loss']:.4f} "
                    f"ssim={avg_logs['ssim_loss']:.4f} "
                    f"aux={avg_logs['aux_loss']:.4f} "
                    f"expert_std={utilization['utilization_std']:.4f} "
                    f"dispatch_counts={utilization['dispatch_counts']} "
                    f"{self._memory_log()}"
                )

            if self.config.val_interval > 0 and self.global_step % self.config.val_interval == 0:
                metrics = self.validate()
                print(f"[val] step={self.global_step} psnr={metrics['psnr']:.3f} loss={metrics['loss']:.4f}")
                if metrics["psnr"] > self.best_psnr:
                    self.best_psnr = metrics["psnr"]
                    self.save_checkpoint("best.pt")

            if self.global_step % self.config.checkpoint_interval == 0:
                self.save_checkpoint(f"step_{self.global_step}.pt")

        self.save_checkpoint("last.pt")


def build_default_trainer(
    model_config: Optional[MoESRConfig] = None,
    trainer_config: Optional[TrainerConfig] = None,
    device: Optional[str] = None,
) -> MoESRTrainer:
    """Convenience factory for the default trainer scaffold."""

    model_config = model_config or MoESRConfig()
    trainer_config = trainer_config or TrainerConfig(
        train_lr_dir="dataset/train/LR",
        train_hr_dir="dataset/train/HR",
        val_lr_dir="dataset/val/LR",
        val_hr_dir="dataset/val/HR",
    )
    model = MoESR(model_config)
    actual_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    train_loader = build_dataloader(
        trainer_config.train_lr_dir,
        trainer_config.train_hr_dir,
        model_config,
        trainer_config,
        train=True,
    )
    val_loader = build_dataloader(
        trainer_config.val_lr_dir,
        trainer_config.val_hr_dir,
        model_config,
        trainer_config,
        train=False,
    )
    return MoESRTrainer(model, model_config, trainer_config, train_loader, val_loader, actual_device)


def build_argparser() -> argparse.ArgumentParser:
    """Build the training CLI parser."""

    parser = argparse.ArgumentParser(description="Train MoESR on paired LR/HR folders.")
    parser.add_argument("--config", default="default", choices=MoESRConfig.PRESET_NAMES)
    parser.add_argument("--train-lr", required=True)
    parser.add_argument("--train-hr", required=True)
    parser.add_argument("--val-lr", required=True)
    parser.add_argument("--val-hr", required=True)
    parser.add_argument("--output-dir", default="runs/moesr")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--steps", type=int, default=500_000)
    parser.add_argument("--log-interval", type=int, default=100)
    parser.add_argument("--val-interval", type=int, default=2_000)
    parser.add_argument("--val-max-images", type=int, default=0)
    parser.add_argument("--val-tile-size", type=int, default=0)
    parser.add_argument("--val-tile-overlap", type=int, default=16)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--device", default=None)
    return parser
