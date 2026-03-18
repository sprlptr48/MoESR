from __future__ import annotations

import argparse
import math
from pathlib import Path

import torch
from PIL import Image
from torch import Tensor
from torchvision.transforms import functional as TF

from moesr.models.config import MoESRConfig
from moesr.models.moesr import MoESR


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}


def list_images(path: Path) -> list[Path]:
    """Collect image files from a path."""

    if path.is_file():
        return [path]
    return sorted(candidate for candidate in path.iterdir() if candidate.suffix.lower() in IMAGE_EXTENSIONS)


def load_image(path: Path) -> Tensor:
    """Load an image as a BCHW float tensor in [0, 1]."""

    image = Image.open(path).convert("RGB")
    return TF.to_tensor(image).unsqueeze(0)


def save_image(tensor: Tensor, path: Path) -> None:
    """Save a BCHW or CHW tensor as an image."""

    image = tensor.detach().cpu().clamp(0.0, 1.0)
    if image.dim() == 4:
        image = image.squeeze(0)
    path.parent.mkdir(parents=True, exist_ok=True)
    TF.to_pil_image(image).save(path)


def resolve_model_config(checkpoint: dict, preset_name: str) -> MoESRConfig:
    """Load config from checkpoint if present, otherwise use the preset."""

    if "model_config" in checkpoint:
        return MoESRConfig(**checkpoint["model_config"])
    return MoESRConfig.from_preset(preset_name)


def load_model(checkpoint_path: str, preset_name: str, device: torch.device) -> tuple[MoESR, MoESRConfig]:
    """Instantiate and load a checkpointed model."""

    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = resolve_model_config(checkpoint, preset_name)
    model = MoESR(config).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model, config


def run_model(model: MoESR, image: Tensor, use_half: bool) -> dict[str, Tensor]:
    """Run direct inference on a single BCHW image."""

    with torch.inference_mode():
        if use_half and image.device.type == "cuda":
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                return model(image)
        return model(image)


def _tile_weight(height: int, width: int, overlap: int, device: torch.device, dtype: torch.dtype) -> Tensor:
    """Create a 2D blending weight for overlap-add tiling."""

    if overlap <= 0:
        return torch.ones((1, 1, height, width), device=device, dtype=dtype)

    overlap_h = min(overlap, max(height // 2, 1))
    overlap_w = min(overlap, max(width // 2, 1))
    y = torch.ones(height, device=device, dtype=dtype)
    x = torch.ones(width, device=device, dtype=dtype)

    if overlap_h > 0 and height > 1:
        ramp = torch.linspace(0.0, 1.0, overlap_h + 2, device=device, dtype=dtype)[1:-1]
        y[:overlap_h] = ramp
        y[-overlap_h:] = torch.minimum(y[-overlap_h:], ramp.flip(0))
    if overlap_w > 0 and width > 1:
        ramp = torch.linspace(0.0, 1.0, overlap_w + 2, device=device, dtype=dtype)[1:-1]
        x[:overlap_w] = ramp
        x[-overlap_w:] = torch.minimum(x[-overlap_w:], ramp.flip(0))

    weight = torch.outer(y, x).clamp_min(1e-3)
    return weight.view(1, 1, height, width)


def tiled_inference(
    model: MoESR,
    image: Tensor,
    tile_size: int,
    use_half: bool,
    tile_overlap: int = 0,
) -> dict[str, Tensor]:
    """Run tiled inference and merge tiles into a single output."""

    _, _, h, w = image.shape
    if max(h, w) <= tile_size:
        return run_model(model, image, use_half)

    scale = model.config.scale_factor
    stage1_scale = model.config.stage_scale_factor
    stride = max(tile_size - tile_overlap, 1)
    output = image.new_zeros((1, 3, h * scale, w * scale))
    stage1_output = image.new_zeros((1, 3, h * stage1_scale, w * stage1_scale))
    output_weight = image.new_zeros((1, 1, h * scale, w * scale))
    stage1_weight = image.new_zeros((1, 1, h * stage1_scale, w * stage1_scale))

    top_positions = list(range(0, max(h - tile_size, 0) + 1, stride))
    left_positions = list(range(0, max(w - tile_size, 0) + 1, stride))
    if not top_positions or top_positions[-1] != max(h - tile_size, 0):
        top_positions.append(max(h - tile_size, 0))
    if not left_positions or left_positions[-1] != max(w - tile_size, 0):
        left_positions.append(max(w - tile_size, 0))

    for top in top_positions:
        for left in left_positions:
            tile = image[:, :, top : min(top + tile_size, h), left : min(left + tile_size, w)]
            tile_outputs = run_model(model, tile, use_half)
            out_top = top * scale
            out_left = left * scale
            mid_top = top * stage1_scale
            mid_left = left * stage1_scale
            output_patch = tile_outputs["output"]
            stage1_patch = tile_outputs["stage1_output"]
            output_patch_weight = _tile_weight(
                output_patch.shape[-2],
                output_patch.shape[-1],
                tile_overlap * scale,
                output_patch.device,
                output_patch.dtype,
            )
            stage1_patch_weight = _tile_weight(
                stage1_patch.shape[-2],
                stage1_patch.shape[-1],
                tile_overlap * stage1_scale,
                stage1_patch.device,
                stage1_patch.dtype,
            )
            output[
                :,
                :,
                out_top : out_top + output_patch.shape[-2],
                out_left : out_left + output_patch.shape[-1],
            ] += output_patch * output_patch_weight
            output_weight[
                :,
                :,
                out_top : out_top + output_patch.shape[-2],
                out_left : out_left + output_patch.shape[-1],
            ] += output_patch_weight
            stage1_output[
                :,
                :,
                mid_top : mid_top + stage1_patch.shape[-2],
                mid_left : mid_left + stage1_patch.shape[-1],
            ] += stage1_patch * stage1_patch_weight
            stage1_weight[
                :,
                :,
                mid_top : mid_top + stage1_patch.shape[-2],
                mid_left : mid_left + stage1_patch.shape[-1],
            ] += stage1_patch_weight

    output = output / output_weight.clamp_min(1e-6)
    stage1_output = stage1_output / stage1_weight.clamp_min(1e-6)
    return {"output": output, "stage1_output": stage1_output, "aux_loss": image.new_zeros(())}


def infer_path(
    model: MoESR,
    input_path: Path,
    output_dir: Path,
    tile_size: int,
    tile_overlap: int,
    use_half: bool,
    save_passes: bool,
    device: torch.device,
) -> None:
    """Run inference for a file or folder of files."""

    for image_path in list_images(input_path):
        image = load_image(image_path).to(device)
        outputs = tiled_inference(
            model,
            image,
            tile_size=tile_size,
            tile_overlap=tile_overlap,
            use_half=use_half,
        )
        stem = image_path.stem
        suffix = image_path.suffix or ".png"
        save_image(outputs["output"], output_dir / f"{stem}_sr{suffix}")
        if save_passes:
            save_image(outputs["stage1_output"], output_dir / f"{stem}_stage1{suffix}")


def build_argparser() -> argparse.ArgumentParser:
    """Build the inference CLI parser."""

    parser = argparse.ArgumentParser(description="Run MoESR inference on an image or folder.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--config", default="default", choices=MoESRConfig.PRESET_NAMES)
    parser.add_argument("--tile-size", type=int, default=256)
    parser.add_argument("--tile-overlap", type=int, default=16)
    parser.add_argument("--half", action="store_true")
    parser.add_argument("--save-passes", action="store_true")
    parser.add_argument("--device", default=None)
    return parser


def main() -> None:
    """CLI entrypoint for inference."""

    args = build_argparser().parse_args()
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model, _ = load_model(args.checkpoint, args.config, device)
    infer_path(
        model=model,
        input_path=Path(args.input),
        output_dir=Path(args.output_dir),
        tile_size=args.tile_size,
        tile_overlap=args.tile_overlap,
        use_half=args.half,
        save_passes=args.save_passes,
        device=device,
    )


if __name__ == "__main__":
    main()
