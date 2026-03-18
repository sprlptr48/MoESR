from __future__ import annotations

import argparse
from pathlib import Path

import torch

from moesr.infer import list_images, load_image, load_model, save_image, tiled_inference
from moesr.models.config import MoESRConfig


def build_argparser() -> argparse.ArgumentParser:
    """Build the folder test CLI parser."""

    parser = argparse.ArgumentParser(description="Run MoESR over a default test folder and save outputs.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--input-dir", default="test_images")
    parser.add_argument("--output-dir", default="test_outputs")
    parser.add_argument("--config", default="default", choices=MoESRConfig.PRESET_NAMES)
    parser.add_argument("--tile-size", type=int, default=256)
    parser.add_argument("--tile-overlap", type=int, default=16)
    parser.add_argument("--half", action="store_true")
    parser.add_argument("--save-passes", action="store_true")
    parser.add_argument("--device", default=None)
    return parser


def main() -> None:
    """Run folder inference with simple defaults."""

    args = build_argparser().parse_args()
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model, _ = load_model(args.checkpoint, args.config, device)
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    inputs_dir = output_dir / "inputs"
    outputs_dir = output_dir / "outputs"
    stage1_dir = output_dir / "stage1"
    for image_path in list_images(input_dir):
        image = load_image(image_path).to(device)
        outputs = tiled_inference(
            model,
            image,
            tile_size=args.tile_size,
            tile_overlap=args.tile_overlap,
            use_half=args.half,
        )
        suffix = image_path.suffix or ".png"
        save_image(image, inputs_dir / f"{image_path.stem}{suffix}")
        save_image(outputs["output"], outputs_dir / f"{image_path.stem}_sr{suffix}")
        if args.save_passes:
            save_image(outputs["stage1_output"], stage1_dir / f"{image_path.stem}_stage1{suffix}")
    print(f"Processed images from {args.input_dir} into {args.output_dir}")


if __name__ == "__main__":
    main()
