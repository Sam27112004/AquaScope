"""
scripts/train.py — CLI entry point for training an Aquascope model.

Usage::

    python scripts/train.py --config config/training_config.yaml
    python scripts/train.py --config config/training_config.yaml --epochs 100 --model yolo
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running as `python scripts/train.py` from the project root.
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch

from aquascope.config.config_manager import ConfigManager
from aquascope.datasets.augmentations import build_augmentation_pipeline
from aquascope.datasets.dataset_loader import build_dataloader
from aquascope.models.model_registry import ModelRegistry
from aquascope.training.callbacks import CheckpointSaver, EarlyStopping
from aquascope.training.losses import ClassificationLoss, DetectionLoss, SegmentationLoss
from aquascope.training.trainer import Trainer
from aquascope.utils.logging_utils import get_logger

logger = get_logger(__name__)

_LOSS_MAP = {
    "detection": DetectionLoss,
    "segmentation": SegmentationLoss,
    "classification": ClassificationLoss,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train an Aquascope model.")
    parser.add_argument("--config", default="config/training_config.yaml", help="Path to training YAML config.")
    parser.add_argument("--epochs", type=int, default=None, help="Override config epochs.")
    parser.add_argument("--batch-size", type=int, default=None, help="Override config batch size.")
    parser.add_argument("--model", default=None, help="Override config model slug.")
    parser.add_argument("--task", default=None, help="Override task (detection/segmentation/classification).")
    parser.add_argument("--output-dir", default=None, help="Override checkpoint directory.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    overrides: dict = {}
    if args.epochs:
        overrides.setdefault("training", {})["epochs"] = args.epochs
    if args.batch_size:
        overrides.setdefault("training", {})["batch_size"] = args.batch_size
    if args.model:
        overrides["model"] = args.model
    if args.task:
        overrides["task"] = args.task

    cfg = ConfigManager.load(args.config, overrides=overrides or None)

    task: str = cfg.task
    model_slug: str = cfg.model
    epochs: int = cfg.training.epochs
    batch_size: int = cfg.training.batch_size
    lr: float = cfg.training.learning_rate
    checkpoint_dir = args.output_dir or cfg.tracking.checkpoint_dir

    logger.info("Task: %s | Model: %s | Epochs: %d | Batch: %d", task, model_slug, epochs, batch_size)

    # --- Dataset ---
    train_transforms = build_augmentation_pipeline(image_size=cfg.datasets.image_size, mode="train")
    val_transforms = build_augmentation_pipeline(image_size=cfg.datasets.image_size, mode="val")

    train_loader = build_dataloader(
        annotations_file=Path(cfg.datasets.annotations_dir) / "train.json",
        images_dir=Path(cfg.datasets.processed_dir) / "images",
        batch_size=batch_size,
        shuffle=True,
        num_workers=cfg.datasets.num_workers,
        task=task,
    )
    val_loader = build_dataloader(
        annotations_file=Path(cfg.datasets.annotations_dir) / "val.json",
        images_dir=Path(cfg.datasets.processed_dir) / "images",
        batch_size=batch_size,
        shuffle=False,
        num_workers=cfg.datasets.num_workers,
        task=task,
    )

    num_classes = len(train_loader.dataset.class_names)  # type: ignore[attr-defined]

    # --- Model ---
    model = ModelRegistry.build(task, model_slug, num_classes=num_classes, config=cfg)
    params = model.param_count()
    logger.info("Model '%s' — %d trainable / %d total params.", model_slug, params["trainable"], params["total"])

    # --- Optimiser & Scheduler ---
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=cfg.training.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    loss_fn = _LOSS_MAP[task]()

    # --- Callbacks ---
    callbacks = [
        EarlyStopping(monitor="val_loss", patience=cfg.training.early_stopping_patience),
        CheckpointSaver(
            checkpoint_dir=checkpoint_dir,
            monitor="val_loss",
            save_best_only=cfg.training.save_best_only,
        ),
    ]

    # --- Train ---
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        loss_fn=loss_fn,
        scheduler=scheduler,
        config=cfg,
        callbacks=callbacks,
    )
    history = trainer.fit(train_loader, val_loader, epochs=epochs)
    logger.info("Training complete. Best val_loss: %.4f", min(history.get("val_loss", [0])))


if __name__ == "__main__":
    main()
