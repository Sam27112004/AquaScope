"""
Dataset merger for unification pipeline.

Orchestrates processing and merging of multiple raw datasets.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from utils.file_utils import ensure_dir

from datasets.class_mapper import ClassMapper
from datasets.dataset_processors import (
    TrashCANProcessor,
    YOLOBBoxProcessor,
    YOLOOBBProcessor,
)


class DatasetMerger:
    """Merges multiple raw datasets into unified COCO format."""

    def __init__(
        self,
        raw_dir: Path,
        output_images_dir: Path,
        output_annotations_dir: Path,
        logger,
    ):
        """Initialize merger.

        Args:
            raw_dir: Directory containing raw datasets
            output_images_dir: Output directory for unified images
            output_annotations_dir: Output directory for COCO JSON files
            logger: Logger instance
        """
        self.raw_dir = Path(raw_dir)
        self.output_images_dir = Path(output_images_dir)
        self.output_annotations_dir = Path(output_annotations_dir)
        self.logger = logger

        # Initialize processors for all datasets
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

    def merge_split(self, split: str) -> Path:
        """Merge all datasets for one split (train/val/test).

        Args:
            split: Split name ("train", "val", or "test")

        Returns:
            Path to output COCO JSON file
        """
        # Initialize COCO structure
        merged_coco = {
            "info": {
                "description": "AquaScope Unified Underwater Inspection Dataset",
                "version": "1.0",
                "year": 2026,
                "contributor": "AquaScope-AI",
                "date_created": "2026-03-23",
            },
            "licenses": [],
            "images": [],
            "annotations": [],
            "categories": ClassMapper.get_coco_categories(),
        }

        next_image_id = 1
        next_ann_id = 1

        # Process each dataset
        for processor in self.processors:
            self.logger.info(f"Processing {processor.dataset_name} ({split})...")

            try:
                result = processor.process_split(split, next_image_id, next_ann_id)
            except FileNotFoundError:
                # Some datasets don't have all splits (e.g., TrashCAN has no test)
                self.logger.warning(
                    f"  Split '{split}' not found for {processor.dataset_name}, skipping"
                )
                continue

            # Merge images and annotations
            merged_coco["images"].extend(result["images"])
            merged_coco["annotations"].extend(result["annotations"])

            # Copy images
            self._copy_images(processor, split, result["images"])

            # Update counters
            next_image_id = result["next_image_id"]
            next_ann_id = result["next_ann_id"]

        # Write merged JSON
        output_path = self.output_annotations_dir / f"{split}.json"
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(merged_coco, f, indent=2)

        self.logger.info(
            f"{split.upper()}: {len(merged_coco['images'])} images, "
            f"{len(merged_coco['annotations'])} annotations → {output_path.name}"
        )

        return output_path

    def _copy_images(
        self, processor, split: str, images: list[dict]
    ) -> None:
        """Copy images from source to unified output directory.

        Args:
            processor: Processor instance
            split: Split name
            images: List of image dicts with prefixed filenames
        """
        for img_dict in images:
            # Determine source image path based on dataset format
            if processor.dataset_name == "trashcan":
                # TrashCAN COCO format: images in split subdirectory
                # Strip prefix to get original filename
                original_name = img_dict["file_name"].replace(
                    f"{processor.dataset_name}_", "", 1
                )
                src_path = processor.source_dir / split / original_name
            else:
                # YOLO format: images in {split}/images/
                original_name = img_dict["file_name"].replace(
                    f"{processor.dataset_name}_", "", 1
                )
                split_dir = "valid" if split == "val" else split
                src_path = processor.source_dir / split_dir / "images" / original_name

            dst_path = self.output_images_dir / img_dict["file_name"]

            # Copy if not already present
            if not dst_path.exists():
                if not src_path.exists():
                    self.logger.warning(f"  Source image not found: {src_path}")
                    continue
                try:
                    shutil.copy2(src_path, dst_path)
                except Exception as e:
                    self.logger.error(f"  Failed to copy {src_path}: {e}")

    def merge_all(self) -> dict[str, Path]:
        """Merge all splits (train, val, test).

        Returns:
            Dictionary mapping split names to output JSON paths
        """
        # Ensure output directories exist
        ensure_dir(self.output_images_dir)
        ensure_dir(self.output_annotations_dir)

        results = {}
        for split in ["train", "val", "test"]:
            self.logger.info(f"\n{'=' * 60}")
            self.logger.info(f"Merging {split.upper()} split")
            self.logger.info("=" * 60)

            try:
                results[split] = self.merge_split(split)
            except Exception as e:
                self.logger.error(f"Failed to merge {split}: {e}", exc_info=True)
                raise

        return results
