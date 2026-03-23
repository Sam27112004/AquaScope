"""
Dataset processors for unification pipeline.

Each processor handles one raw dataset format and converts to COCO.
"""

from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

from datasets.class_mapper import ClassMapper
from datasets.format_converters import (
    parse_yolo_label_file,
    yolo_bbox_to_coco,
    yolo_obb_to_coco,
)


class BaseDatasetProcessor:
    """Base class for dataset-specific processors."""

    def __init__(self, source_dir: Path, dataset_name: str, logger):
        """Initialize processor.

        Args:
            source_dir: Path to dataset's root directory
            dataset_name: Short name for prefixing filenames
            logger: Logger instance
        """
        self.source_dir = Path(source_dir)
        self.dataset_name = dataset_name
        self.logger = logger

    def process_split(
        self, split: str, start_image_id: int, start_ann_id: int
    ) -> dict:
        """Process one split and return COCO-format data.

        Args:
            split: Split name ("train", "val", or "test")
            start_image_id: Starting image ID for this processor
            start_ann_id: Starting annotation ID for this processor

        Returns:
            Dictionary with:
                - images: List of COCO image dicts
                - annotations: List of COCO annotation dicts
                - next_image_id: Next available image ID
                - next_ann_id: Next available annotation ID
        """
        raise NotImplementedError


class TrashCANProcessor(BaseDatasetProcessor):
    """Process TrashCAN COCO JSON dataset."""

    def __init__(self, source_dir: Path, dataset_name: str, logger):
        super().__init__(source_dir, dataset_name, logger)

    def process_split(
        self, split: str, start_image_id: int, start_ann_id: int
    ) -> dict:
        """Load TrashCAN COCO JSON, remap classes, prefix filenames."""

        # Map split name to file name
        json_file = self.source_dir / f"instances_{split}_trashcan.json"

        if not json_file.exists():
            raise FileNotFoundError(f"TrashCAN {split} file not found: {json_file}")

        self.logger.info(f"  Loading {json_file.name}...")

        # Load COCO JSON
        with open(json_file, "r", encoding="utf-8") as f:
            coco_data = json.load(f)

        # Build category mapping: original_id → target_id
        cat_map = {}
        for cat in coco_data.get("categories", []):
            target_id = ClassMapper.map_trashcan_class(cat["name"])
            cat_map[cat["id"]] = target_id  # Can be None for excluded classes

        # Build image ID mapping: original_id → new_id
        image_id_map = {}
        images = []
        next_image_id = start_image_id

        for img in coco_data.get("images", []):
            original_id = img["id"]
            new_id = next_image_id

            image_id_map[original_id] = new_id

            # Create new image dict with prefixed filename
            new_img = {
                "id": new_id,
                "file_name": f"{self.dataset_name}_{img['file_name']}",
                "width": img["width"],
                "height": img["height"],
                "source_dataset": self.dataset_name,
            }
            images.append(new_img)
            next_image_id += 1

        # Process annotations
        annotations = []
        next_ann_id = start_ann_id

        for ann in coco_data.get("annotations", []):
            # Map category
            new_cat_id = cat_map.get(ann["category_id"])

            # Skip if class is excluded (e.g., rov)
            if new_cat_id is None:
                continue

            # Skip if image was not processed
            new_image_id = image_id_map.get(ann["image_id"])
            if new_image_id is None:
                continue

            # Create new annotation
            bbox = ann["bbox"]
            new_ann = {
                "id": next_ann_id,
                "image_id": new_image_id,
                "category_id": new_cat_id,
                "bbox": bbox,
                "area": ann.get("area", bbox[2] * bbox[3]),
                "iscrowd": 0,
            }

            # Preserve segmentation if present
            if "segmentation" in ann:
                new_ann["segmentation"] = ann["segmentation"]

            annotations.append(new_ann)
            next_ann_id += 1

        self.logger.info(
            f"    Processed {len(images)} images, {len(annotations)} annotations"
        )

        return {
            "images": images,
            "annotations": annotations,
            "next_image_id": next_image_id,
            "next_ann_id": next_ann_id,
        }


class YOLOBBoxProcessor(BaseDatasetProcessor):
    """Process YOLO standard bounding box dataset."""

    def __init__(self, source_dir: Path, dataset_name: str, class_name: str, logger):
        """Initialize YOLO bbox processor.

        Args:
            source_dir: Path to YOLO dataset root (contains train/, valid/, test/)
            dataset_name: Name for prefixing filenames
            class_name: Original class name to map (e.g., "crack")
            logger: Logger instance
        """
        super().__init__(source_dir, dataset_name, logger)
        self.class_name = class_name

    def process_split(
        self, split: str, start_image_id: int, start_ann_id: int
    ) -> dict:
        """Convert YOLO bbox format to COCO format."""

        # YOLO uses "valid" instead of "val"
        split_dir = "valid" if split == "val" else split
        images_dir = self.source_dir / split_dir / "images"
        labels_dir = self.source_dir / split_dir / "labels"

        if not images_dir.exists():
            raise FileNotFoundError(
                f"{self.dataset_name} {split} not found: {images_dir}"
            )

        self.logger.info(f"  Processing {images_dir}...")

        # Map class
        target_class_id = ClassMapper.map_simple_class(self.class_name)
        if target_class_id is None:
            self.logger.warning(f"Class {self.class_name} unmapped, skipping dataset")
            return {
                "images": [],
                "annotations": [],
                "next_image_id": start_image_id,
                "next_ann_id": start_ann_id,
            }

        images = []
        annotations = []
        next_image_id = start_image_id
        next_ann_id = start_ann_id

        # Process each image
        for img_path in sorted(images_dir.glob("*")):
            if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                continue

            # Get image dimensions
            try:
                with Image.open(img_path) as im:
                    img_width, img_height = im.size
            except Exception as e:
                self.logger.warning(f"Cannot open image {img_path}: {e}")
                continue

            # Create image entry
            new_filename = f"{self.dataset_name}_{img_path.name}"
            images.append(
                {
                    "id": next_image_id,
                    "file_name": new_filename,
                    "width": img_width,
                    "height": img_height,
                    "source_dataset": self.dataset_name,
                }
            )

            # Parse label file
            label_path = labels_dir / f"{img_path.stem}.txt"
            labels = parse_yolo_label_file(label_path, format_type="bbox")

            # Convert each bbox to COCO format
            for label in labels:
                class_id, center_x, center_y, width, height = label

                # Convert to COCO bbox
                x, y, w, h = yolo_bbox_to_coco(
                    center_x, center_y, width, height, img_width, img_height
                )

                # Create annotation
                annotations.append(
                    {
                        "id": next_ann_id,
                        "image_id": next_image_id,
                        "category_id": target_class_id,
                        "bbox": [x, y, w, h],
                        "area": w * h,
                        "iscrowd": 0,
                    }
                )
                next_ann_id += 1

            next_image_id += 1

        self.logger.info(
            f"    Processed {len(images)} images, {len(annotations)} annotations"
        )

        return {
            "images": images,
            "annotations": annotations,
            "next_image_id": next_image_id,
            "next_ann_id": next_ann_id,
        }


class YOLOOBBProcessor(BaseDatasetProcessor):
    """Process YOLO Oriented Bounding Box dataset."""

    def __init__(self, source_dir: Path, dataset_name: str, class_name: str, logger):
        """Initialize YOLO OBB processor.

        Args:
            source_dir: Path to YOLO dataset root (contains train/, valid/, test/)
            dataset_name: Name for prefixing filenames
            class_name: Original class name to map (e.g., "trash_plastic")
            logger: Logger instance
        """
        super().__init__(source_dir, dataset_name, logger)
        self.class_name = class_name

    def process_split(
        self, split: str, start_image_id: int, start_ann_id: int
    ) -> dict:
        """Convert YOLO OBB format to COCO bbox format."""

        # YOLO uses "valid" instead of "val"
        split_dir = "valid" if split == "val" else split
        images_dir = self.source_dir / split_dir / "images"
        labels_dir = self.source_dir / split_dir / "labels"

        if not images_dir.exists():
            raise FileNotFoundError(
                f"{self.dataset_name} {split} not found: {images_dir}"
            )

        self.logger.info(f"  Processing {images_dir}...")

        # Map class
        target_class_id = ClassMapper.map_simple_class(self.class_name)
        if target_class_id is None:
            self.logger.warning(f"Class {self.class_name} unmapped, skipping dataset")
            return {
                "images": [],
                "annotations": [],
                "next_image_id": start_image_id,
                "next_ann_id": start_ann_id,
            }

        images = []
        annotations = []
        next_image_id = start_image_id
        next_ann_id = start_ann_id

        # Process each image
        for img_path in sorted(images_dir.glob("*")):
            if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                continue

            # Get image dimensions
            try:
                with Image.open(img_path) as im:
                    img_width, img_height = im.size
            except Exception as e:
                self.logger.warning(f"Cannot open image {img_path}: {e}")
                continue

            # Create image entry
            new_filename = f"{self.dataset_name}_{img_path.name}"
            images.append(
                {
                    "id": next_image_id,
                    "file_name": new_filename,
                    "width": img_width,
                    "height": img_height,
                    "source_dataset": self.dataset_name,
                }
            )

            # Parse label file (OBB format)
            label_path = labels_dir / f"{img_path.stem}.txt"
            labels = parse_yolo_label_file(label_path, format_type="obb")

            # Convert each OBB to COCO bbox
            for label in labels:
                class_id, x1, y1, x2, y2, x3, y3, x4, y4 = label

                # Convert to axis-aligned bbox
                x, y, w, h = yolo_obb_to_coco(
                    x1, y1, x2, y2, x3, y3, x4, y4, img_width, img_height
                )

                # Create annotation
                annotations.append(
                    {
                        "id": next_ann_id,
                        "image_id": next_image_id,
                        "category_id": target_class_id,
                        "bbox": [x, y, w, h],
                        "area": w * h,
                        "iscrowd": 0,
                    }
                )
                next_ann_id += 1

            next_image_id += 1

        self.logger.info(
            f"    Processed {len(images)} images, {len(annotations)} annotations"
        )

        return {
            "images": images,
            "annotations": annotations,
            "next_image_id": next_image_id,
            "next_ann_id": next_ann_id,
        }
