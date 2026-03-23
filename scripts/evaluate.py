"""
scripts/evaluate.py — CLI entry point for evaluating a trained checkpoint.

Usage::

    python scripts/evaluate.py --checkpoint experiments/checkpoints/best.pt
    python scripts/evaluate.py --checkpoint experiments/checkpoints/best.pt --split val
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch

from aquascope.config.config_manager import ConfigManager
from aquascope.datasets.dataset_loader import build_dataloader
from aquascope.models.model_registry import ModelRegistry
from aquascope.training.metrics import compute_detection_metrics, compute_classification_metrics
from aquascope.utils.logging_utils import get_logger

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate an Aquascope checkpoint.")
    parser.add_argument("--checkpoint", required=True, help="Path to the .pt checkpoint file.")
    parser.add_argument("--config", default="config/training_config.yaml", help="Config YAML to use.")
    parser.add_argument("--split", default="val", choices=["val", "test"], help="Dataset split to evaluate on.")
    parser.add_argument("--batch-size", type=int, default=16)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = ConfigManager.load(args.config)

    task: str = cfg.task
    model_slug: str = cfg.model

    # Load checkpoint metadata to determine num_classes.
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    payload = torch.load(checkpoint_path, map_location="cpu")
    num_classes: int = payload.get("num_classes", 10)

    model = ModelRegistry.build(task, model_slug, num_classes=num_classes, config=cfg)
    model.load_checkpoint(checkpoint_path)
    model.eval()
    logger.info("Loaded checkpoint '%s' (task=%s, model=%s, classes=%d).", checkpoint_path, task, model_slug, num_classes)

    # Dataset.
    loader = build_dataloader(
        annotations_file=Path(cfg.datasets.annotations_dir) / f"{args.split}.json",
        images_dir=Path(cfg.datasets.processed_dir) / "images",
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=cfg.datasets.num_workers,
        task=task,
    )

    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    model.to(device)

    all_preds: list[dict] = []
    all_gts: list[dict] = []

    with torch.no_grad():
        for images, targets in loader:
            if isinstance(images, list):
                images = [img.to(device) for img in images]
            else:
                images = images.to(device)

            outputs = model(images)

            if isinstance(outputs, list):
                all_preds.extend(outputs)
            else:
                # For single-output models, wrap each sample.
                for i in range(len(targets)):
                    all_preds.append({"logits": outputs[i:i+1]})

            all_gts.extend(targets if isinstance(targets, list) else [targets])

    # Compute metrics.
    if task == "detection":
        metrics = compute_detection_metrics(all_preds, all_gts, num_classes=num_classes)
    elif task == "classification":
        logits = torch.cat([p["logits"] for p in all_preds])
        gt_labels = torch.cat([t["labels"] for t in all_gts])
        metrics = compute_classification_metrics(logits, gt_labels)
    else:
        logger.warning("Metric computation for task '%s' is not yet implemented.", task)
        metrics = {}

    logger.info("Evaluation results (%s split):", args.split)
    for key, value in metrics.items():
        logger.info("  %s: %.4f", key, value)


if __name__ == "__main__":
    main()
