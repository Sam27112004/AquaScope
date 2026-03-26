"""
Dataset processors for YOLO unification pipeline.

Each processor handles one raw dataset format and emits YOLO labels.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TypedDict

from PIL import Image

from datasets.class_mapper import ClassMapper
from datasets.format_converters import (
    coco_bbox_to_yolo,
    parse_yolo_label_file,
    yolo_obb_to_coco,
)


class UnifiedSample(TypedDict):
    """One unified sample ready to write into processed images/labels."""

    source_image_path: Path
    target_file_name: str
    yolo_lines: list[str]


class BaseDatasetProcessor:
    """Base class for dataset-specific processors."""

    def __init__(self, source_dir: Path, dataset_name: str, logger):
        self.source_dir = Path(source_dir)
        self.dataset_name = dataset_name
        self.logger = logger

    def process_split(self, split: str) -> list[UnifiedSample]:
        """Process one split and return unified YOLO-ready samples."""
        raise NotImplementedError


class TrashCANProcessor(BaseDatasetProcessor):
    """Process TrashCAN COCO JSON dataset into YOLO labels."""

    def process_split(self, split: str) -> list[UnifiedSample]:
        json_file = self.source_dir / f"instances_{split}_trashcan.json"
        if not json_file.exists():
            raise FileNotFoundError(f"TrashCAN {split} file not found: {json_file}")

        self.logger.info(f"  Loading {json_file.name}...")
        with open(json_file, "r", encoding="utf-8") as f:
            coco_data = json.load(f)

        cat_map: dict[int, int | None] = {}
        for cat in coco_data.get("categories", []):
            cat_map[cat["id"]] = ClassMapper.map_trashcan_class(cat["name"])

        ann_by_image: dict[int, list[dict]] = {}
        for ann in coco_data.get("annotations", []):
            ann_by_image.setdefault(ann["image_id"], []).append(ann)

        samples: list[UnifiedSample] = []
        for img in coco_data.get("images", []):
            image_path = self.source_dir / split / img["file_name"]
            target_name = f"{self.dataset_name}_{img['file_name']}"

            yolo_lines: list[str] = []
            for ann in ann_by_image.get(img["id"], []):
                coco_class_id = cat_map.get(ann["category_id"])
                if coco_class_id is None:
                    continue

                x, y, w, h = ann["bbox"]
                cx, cy, nw, nh = coco_bbox_to_yolo(
                    x=x,
                    y=y,
                    width=w,
                    height=h,
                    img_width=img["width"],
                    img_height=img["height"],
                )

                if nw <= 0 or nh <= 0:
                    continue

                yolo_class = ClassMapper.coco_id_to_yolo_id(coco_class_id)
                yolo_lines.append(
                    f"{yolo_class} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}"
                )

            samples.append(
                {
                    "source_image_path": image_path,
                    "target_file_name": target_name,
                    "yolo_lines": yolo_lines,
                }
            )

        self.logger.info(f"    Processed {len(samples)} images")
        return samples


class YOLOBBoxProcessor(BaseDatasetProcessor):
    """Process standard YOLO bbox dataset and remap class IDs."""

    def __init__(self, source_dir: Path, dataset_name: str, class_name: str, logger):
        super().__init__(source_dir, dataset_name, logger)
        self.class_name = class_name

    def process_split(self, split: str) -> list[UnifiedSample]:
        split_dir = "valid" if split == "val" else split
        images_dir = self.source_dir / split_dir / "images"
        labels_dir = self.source_dir / split_dir / "labels"

        if not images_dir.exists():
            raise FileNotFoundError(f"{self.dataset_name} {split} not found: {images_dir}")

        coco_class_id = ClassMapper.map_simple_class(self.class_name)
        if coco_class_id is None:
            self.logger.warning(f"Class {self.class_name} unmapped, skipping dataset")
            return []
        yolo_class = ClassMapper.coco_id_to_yolo_id(coco_class_id)

        samples: list[UnifiedSample] = []
        for img_path in sorted(images_dir.glob("*")):
            if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                continue

            label_path = labels_dir / f"{img_path.stem}.txt"
            labels = parse_yolo_label_file(label_path, format_type="bbox")

            yolo_lines: list[str] = []
            for label in labels:
                _, cx, cy, w, h = label
                if w <= 0 or h <= 0:
                    continue
                yolo_lines.append(f"{yolo_class} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")

            samples.append(
                {
                    "source_image_path": img_path,
                    "target_file_name": f"{self.dataset_name}_{img_path.name}",
                    "yolo_lines": yolo_lines,
                }
            )

        self.logger.info(f"    Processed {len(samples)} images")
        return samples


class YOLOOBBProcessor(BaseDatasetProcessor):
    """Process YOLO OBB dataset and convert to axis-aligned YOLO bbox labels."""

    def __init__(self, source_dir: Path, dataset_name: str, class_name: str, logger):
        super().__init__(source_dir, dataset_name, logger)
        self.class_name = class_name

    def process_split(self, split: str) -> list[UnifiedSample]:
        split_dir = "valid" if split == "val" else split
        images_dir = self.source_dir / split_dir / "images"
        labels_dir = self.source_dir / split_dir / "labels"

        if not images_dir.exists():
            raise FileNotFoundError(f"{self.dataset_name} {split} not found: {images_dir}")

        coco_class_id = ClassMapper.map_simple_class(self.class_name)
        if coco_class_id is None:
            self.logger.warning(f"Class {self.class_name} unmapped, skipping dataset")
            return []
        yolo_class = ClassMapper.coco_id_to_yolo_id(coco_class_id)

        samples: list[UnifiedSample] = []
        for img_path in sorted(images_dir.glob("*")):
            if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                continue

            # Image dimensions are required to project normalized OBB points.
            with Image.open(img_path) as img:
                img_width, img_height = img.size

            label_path = labels_dir / f"{img_path.stem}.txt"
            labels = parse_yolo_label_file(label_path, format_type="obb")

            yolo_lines: list[str] = []
            for label in labels:
                _, x1, y1, x2, y2, x3, y3, x4, y4 = label
                x, y, w, h = yolo_obb_to_coco(
                    x1=x1,
                    y1=y1,
                    x2=x2,
                    y2=y2,
                    x3=x3,
                    y3=y3,
                    x4=x4,
                    y4=y4,
                    img_width=img_width,
                    img_height=img_height,
                )
                if w <= 0 or h <= 0:
                    continue

                cx, cy, nw, nh = coco_bbox_to_yolo(
                    x=x,
                    y=y,
                    width=w,
                    height=h,
                    img_width=img_width,
                    img_height=img_height,
                )
                if nw <= 0 or nh <= 0:
                    continue

                yolo_lines.append(
                    f"{yolo_class} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}"
                )

            samples.append(
                {
                    "source_image_path": img_path,
                    "target_file_name": f"{self.dataset_name}_{img_path.name}",
                    "yolo_lines": yolo_lines,
                }
            )

        self.logger.info(f"    Processed {len(samples)} images")
        return samples
