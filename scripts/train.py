"""
scripts/train.py — CLI entry point for training an Aquascope model.

Usage::

    python scripts/train.py --config config/training_config.yaml
    python scripts/train.py --config config/training_config.yaml --epochs 100 --model yolo
"""

from __future__ import annotations

import argparse
import os
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


def _train_with_ultralytics(cfg, epochs: int, batch_size: int, checkpoint_dir: str | Path) -> None:
    """Train detection model with Ultralytics YOLOv8 on unified dataset.yaml."""
    from ultralytics import YOLO

    dataset_yaml = Path(getattr(cfg.datasets, "dataset_yaml", "datasets/processed/dataset.yaml"))
    if not dataset_yaml.exists():
        raise FileNotFoundError(f"YOLO dataset YAML not found: {dataset_yaml}")

    model_init = getattr(cfg.training, "yolo_weights", "yolov8n.yaml")
    configured_device = str(getattr(cfg, "device", "cpu"))
    resolved_device = configured_device if (configured_device != "cuda" or torch.cuda.is_available()) else "cpu"
    project_dir = Path(checkpoint_dir)
    project_dir.mkdir(parents=True, exist_ok=True)

    # Ultralytics enables MLflow callback automatically if mlflow is installed.
    # Ensure Windows paths use a valid URI scheme for MLflow store resolution.
    mlruns_dir = (Path("runs") / "mlflow").resolve()
    mlruns_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MLFLOW_TRACKING_URI", mlruns_dir.as_uri())

    logger.info(f"Starting Ultralytics YOLO training with init='{model_init}'")
    logger.info(f"Dataset YAML: {dataset_yaml}")
    logger.info(f"Device: {resolved_device}")

    model = YOLO(model_init)
    model.train(
        data=str(dataset_yaml),
        epochs=epochs,
        batch=batch_size,
        imgsz=int(cfg.datasets.image_size[0]),
        lr0=float(cfg.training.learning_rate),
        weight_decay=float(cfg.training.weight_decay),
        project=str(project_dir),
        name="ultralytics_yolo",
        exist_ok=True,
        device=resolved_device,
        workers=int(cfg.datasets.num_workers),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train an Aquascope model.")
    parser.add_argument("--config", default="config/training_config.yaml", help="Path to training YAML config.")
    parser.add_argument("--epochs", type=int, default=None, help="Override config epochs.")
    parser.add_argument("--batch-size", type=int, default=None, help="Override config batch size.")
    parser.add_argument("--model", default=None, help="Override config model slug.")
    parser.add_argument("--task", default=None, help="Override task (detection/segmentation/classification).")
    parser.add_argument("--output-dir", default=None, help="Override checkpoint directory.")
    parser.add_argument("--max-train-batches", type=int, default=None, help="Limit train batches per epoch (smoke runs).")
    parser.add_argument("--max-val-batches", type=int, default=None, help="Limit val batches per epoch (smoke runs).")
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
    dataset_format = getattr(cfg.datasets, "format", "coco").lower()
    class_names = list(getattr(cfg.datasets, "class_names", []))
    use_ultralytics_yolo = (
        task == "detection"
        and model_slug == "yolo"
        and dataset_format == "yolo"
        and args.max_train_batches is None
        and args.max_val_batches is None
    )

    logger.info(f"Task: {task} | Model: {model_slug} | Epochs: {epochs} | Batch: {batch_size}")

    if use_ultralytics_yolo:
        try:
            _train_with_ultralytics(cfg, epochs=epochs, batch_size=batch_size, checkpoint_dir=checkpoint_dir)
            logger.info("Ultralytics YOLO training completed.")
            return
        except ImportError as exc:
            raise RuntimeError(
                "Ultralytics is required for real YOLO training. Install with: pip install ultralytics"
            ) from exc

    # --- Dataset ---
    train_transforms = build_augmentation_pipeline(image_size=cfg.datasets.image_size, mode="train")
    val_transforms = build_augmentation_pipeline(image_size=cfg.datasets.image_size, mode="val")

    if dataset_format == "yolo":
        train_loader = build_dataloader(
            annotations_file=None,
            images_dir=Path(cfg.datasets.processed_dir) / "images" / "train",
            labels_dir=Path(cfg.datasets.processed_dir) / "labels" / "train",
            batch_size=batch_size,
            shuffle=True,
            num_workers=cfg.datasets.num_workers,
            transforms=train_transforms,
            task=task,
            split="train",
            class_names=class_names,
        )
        val_loader = build_dataloader(
            annotations_file=None,
            images_dir=Path(cfg.datasets.processed_dir) / "images" / "val",
            labels_dir=Path(cfg.datasets.processed_dir) / "labels" / "val",
            batch_size=batch_size,
            shuffle=False,
            num_workers=cfg.datasets.num_workers,
            transforms=val_transforms,
            task=task,
            split="val",
            class_names=class_names,
        )
    else:
        train_loader = build_dataloader(
            annotations_file=Path(cfg.datasets.annotations_dir) / "train.json",
            images_dir=Path(cfg.datasets.processed_dir) / "images",
            batch_size=batch_size,
            shuffle=True,
            num_workers=cfg.datasets.num_workers,
            transforms=train_transforms,
            task=task,
        )
        val_loader = build_dataloader(
            annotations_file=Path(cfg.datasets.annotations_dir) / "val.json",
            images_dir=Path(cfg.datasets.processed_dir) / "images",
            batch_size=batch_size,
            shuffle=False,
            num_workers=cfg.datasets.num_workers,
            transforms=val_transforms,
            task=task,
        )

    num_classes = len(train_loader.dataset.class_names)  # type: ignore[attr-defined]

    # --- Model ---
    model = ModelRegistry.build(task, model_slug, num_classes=num_classes, config=cfg)
    params = model.param_count()
    logger.info(
        f"Model '{model_slug}' — {params['trainable']} trainable / {params['total']} total params."
    )

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
    history = trainer.fit(
        train_loader,
        val_loader,
        epochs=epochs,
        max_train_batches=args.max_train_batches,
        max_val_batches=args.max_val_batches,
    )
    logger.info(f"Training complete. Best val_loss: {min(history.get('val_loss', [0])):.4f}")


if __name__ == "__main__":
    main()
