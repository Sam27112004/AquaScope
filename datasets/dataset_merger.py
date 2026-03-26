"""
Dataset merger for YOLO unification pipeline.

Orchestrates processing and merging of multiple raw datasets into:
  datasets/processed/images/{train,val,test}
  datasets/processed/labels/{train,val,test}
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import yaml

from datasets.class_mapper import ClassMapper
from datasets.dataset_processors import (
    TrashCANProcessor,
    YOLOBBoxProcessor,
    YOLOOBBProcessor,
)
from utils.file_utils import clean_dir, ensure_dir


class DatasetMerger:
    """Merges multiple raw datasets into unified YOLO format."""

    def __init__(self, raw_dir: Path, output_processed_dir: Path, logger):
        self.raw_dir = Path(raw_dir)
        self.output_processed_dir = Path(output_processed_dir)
        self.images_root = self.output_processed_dir / "images"
        self.labels_root = self.output_processed_dir / "labels"
        self.logger = logger

        self.processors = [
            TrashCANProcessor(
                source_dir=self.raw_dir / "trashcan" / "dataset" / "instance_version",
                dataset_name="trashcan",
                logger=logger,
            ),
            YOLOOBBProcessor(
                source_dir=self.raw_dir / "roboflow" / "underwater-trash.v3i.yolov8-obb",
                dataset_name="underwater_trash",
                class_name="trash_plastic",
                logger=logger,
            ),
            YOLOBBoxProcessor(
                source_dir=self.raw_dir / "roboflow" / "Underwater Crack Detection.v11i.yolov8",
                dataset_name="underwater_crack",
                class_name="crack",
                logger=logger,
            ),
            YOLOOBBProcessor(
                source_dir=self.raw_dir / "roboflow" / "Concrete Crack.v1i.yolov8-obb",
                dataset_name="concrete_crack",
                class_name="Concrete-Crack",
                logger=logger,
            ),
        ]

    def merge_split(self, split: str) -> dict:
        """Merge all datasets for one split and write YOLO images/labels."""
        images_dir = ensure_dir(self.images_root / split)
        labels_dir = ensure_dir(self.labels_root / split)

        per_dataset_images: dict[str, int] = {}
        num_images = 0
        num_labels = 0

        for processor in self.processors:
            self.logger.info(f"Processing {processor.dataset_name} ({split})...")
            try:
                samples = processor.process_split(split)
            except FileNotFoundError:
                self.logger.warning(
                    f"  Split '{split}' not found for {processor.dataset_name}, skipping"
                )
                continue

            written_for_dataset = 0
            for sample in samples:
                source_image_path = sample["source_image_path"]
                target_name = sample["target_file_name"]

                if not source_image_path.exists():
                    self.logger.warning(f"  Source image not found: {source_image_path}")
                    continue

                dst_image_path = images_dir / target_name
                dst_label_path = labels_dir / f"{Path(target_name).stem}.txt"

                if not dst_image_path.exists():
                    shutil.copy2(source_image_path, dst_image_path)

                # Always write a label file, even if empty, to preserve sample parity.
                label_text = "\n".join(sample["yolo_lines"])
                if label_text:
                    label_text += "\n"
                dst_label_path.write_text(label_text, encoding="utf-8")

                num_images += 1
                num_labels += len(sample["yolo_lines"])
                written_for_dataset += 1

            per_dataset_images[processor.dataset_name] = (
                per_dataset_images.get(processor.dataset_name, 0) + written_for_dataset
            )

        self.logger.info(f"{split.upper()}: {num_images} images, {num_labels} object labels")

        return {
            "split": split,
            "images": num_images,
            "labels": num_labels,
            "images_dir": str(images_dir),
            "labels_dir": str(labels_dir),
            "per_dataset_images": per_dataset_images,
        }

    def write_dataset_yaml(self) -> Path:
        """Write YOLO dataset.yaml describing split paths and class names."""
        dataset_yaml_path = self.output_processed_dir / "dataset.yaml"
        yolo_data = {
            "path": str(self.output_processed_dir.resolve()),
            "train": "images/train",
            "val": "images/val",
            "test": "images/test",
            "names": {idx: name for idx, name in enumerate(ClassMapper.get_yolo_class_names())},
        }
        with open(dataset_yaml_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(yolo_data, f, sort_keys=False)
        return dataset_yaml_path

    def merge_all(self, clean_output: bool = True) -> dict[str, dict]:
        """Merge train/val/test into unified YOLO format and return summary stats."""
        ensure_dir(self.output_processed_dir)
        ensure_dir(self.images_root)
        ensure_dir(self.labels_root)

        if clean_output:
            clean_dir(self.images_root)
            clean_dir(self.labels_root)
            ensure_dir(self.images_root)
            ensure_dir(self.labels_root)

        results: dict[str, dict] = {}
        for split in ["train", "val", "test"]:
            self.logger.info(f"\n{'=' * 60}")
            self.logger.info(f"Merging {split.upper()} split")
            self.logger.info(f"{'=' * 60}")
            results[split] = self.merge_split(split)

        dataset_yaml_path = self.write_dataset_yaml()

        summary_path = self.output_processed_dir / "summary.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)

        self.logger.info(f"\nWrote YOLO dataset config: {dataset_yaml_path}")
        self.logger.info(f"Wrote unification summary: {summary_path}")
        return results
