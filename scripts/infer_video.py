"""
scripts/infer_video.py — CLI entry point for running video inference.

Usage::

    python scripts/infer_video.py --source path/to/video.mp4
    python scripts/infer_video.py --source 0  # webcam device index
    python scripts/infer_video.py --source path/to/video.mp4 --output experiments/results/out.mp4
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from aquascope.config.config_manager import ConfigManager
from aquascope.inference.video_inference import VideoInferencePipeline
from aquascope.models.model_registry import ModelRegistry
from aquascope.utils.logging_utils import get_logger

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Aquascope video inference.")
    parser.add_argument("--source", required=True, help="Path to video file or camera device index.")
    parser.add_argument("--config", default="config/inference_config.yaml", help="Inference config YAML.")
    parser.add_argument("--checkpoint", default=None, help="Path to model checkpoint (.pt).")
    parser.add_argument("--model", default=None, help="Override model slug.")
    parser.add_argument("--task", default=None, help="Override task.")
    parser.add_argument("--output", default=None, help="Path to write annotated output video.")
    parser.add_argument("--show", action="store_true", help="Display live annotated output.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    overrides: dict = {}
    if args.model:
        overrides["model"] = args.model
    if args.task:
        overrides["task"] = args.task
    if args.checkpoint:
        overrides["checkpoint"] = args.checkpoint
    if args.show:
        overrides.setdefault("video", {})["show_live"] = True

    cfg = ConfigManager.load(args.config, overrides=overrides or None)

    task: str = cfg.task
    model_slug: str = cfg.model

    model = ModelRegistry.build(task, model_slug, num_classes=10, config=cfg)

    checkpoint = args.checkpoint or getattr(cfg, "checkpoint", None)
    if checkpoint and Path(checkpoint).exists():
        model.load_checkpoint(checkpoint)
        logger.info("Loaded checkpoint from '%s'.", checkpoint)
    else:
        logger.warning("No checkpoint provided or found — running with random weights.")

    # Try to resolve source as an integer (webcam index).
    source: str | int = args.source
    try:
        source = int(args.source)
    except ValueError:
        pass

    pipeline = VideoInferencePipeline(model=model, config=cfg)
    logger.info("Starting inference on '%s'...", source)

    output_path = args.output or (
        Path(cfg.tracking.results_dir) / (Path(str(source)).stem + "_annotated.mp4")
        if not isinstance(source, int)
        else None
    )

    results = pipeline.run(source=source, output_path=output_path)
    logger.info("Inference complete. Processed %d frames.", len(results))

    if output_path:
        logger.info("Annotated video saved → %s", output_path)


if __name__ == "__main__":
    main()
