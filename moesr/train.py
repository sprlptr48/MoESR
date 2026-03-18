from __future__ import annotations

from moesr.models.config import MoESRConfig
from moesr.training.trainer import TrainerConfig, build_argparser, build_default_trainer


def main() -> None:
    """CLI entrypoint for training."""

    args = build_argparser().parse_args()
    model_config = MoESRConfig.from_preset(args.config)
    trainer_config = TrainerConfig(
        train_lr_dir=args.train_lr,
        train_hr_dir=args.train_hr,
        val_lr_dir=args.val_lr,
        val_hr_dir=args.val_hr,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        max_steps=args.steps,
        log_interval=args.log_interval,
        checkpoint_interval=args.checkpoint_interval,
        val_interval=args.val_interval,
        val_max_images=args.val_max_images,
        val_tile_size=args.val_tile_size,
        val_tile_overlap=args.val_tile_overlap,
        full_checkpoints=args.full_checkpoints,
        gradient_checkpointing=args.gradient_checkpointing,
        compile_model=args.compile,
    )
    trainer = build_default_trainer(model_config, trainer_config, device=args.device)
    if args.resume:
        trainer.load_checkpoint(args.resume)
    trainer.fit()


if __name__ == "__main__":
    main()
