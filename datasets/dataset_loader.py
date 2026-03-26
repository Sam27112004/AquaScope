"""
UnderwaterDataset loaders for COCO or YOLO label formats.

Supported layouts:

COCO:
    datasets/
        annotations/
            train.json
            val.json
        processed/
            images/
                <image_id>.jpg / .png

YOLO (recommended):
    datasets/
        processed/
            images/
                train/*.jpg
                val/*.jpg
                test/*.jpg
            labels/
                train/*.txt
                val/*.txt
                test/*.txt
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

DEFAULT_CLASS_NAMES = [
    "trash",
    "plastic",
    "fishing_net",
    "marine_growth",
    "surface_damage",
]


def _apply_detection_transform(image, target: dict, transform):
    """Apply Albumentations-like transform to detection sample safely."""
    boxes_tensor = target.get("boxes")
    labels_tensor = target.get("labels")

    img_h, img_w = image.shape[:2]

    bboxes = boxes_tensor.tolist() if boxes_tensor is not None else []
    labels = labels_tensor.tolist() if labels_tensor is not None else []

    sanitized_bboxes: list[list[float]] = []
    sanitized_labels: list[int] = []
    for box, label in zip(bboxes, labels):
        x1, y1, x2, y2 = [float(v) for v in box]

        x1 = max(0.0, min(x1, float(img_w)))
        y1 = max(0.0, min(y1, float(img_h)))
        x2 = max(0.0, min(x2, float(img_w)))
        y2 = max(0.0, min(y2, float(img_h)))

        if x2 <= x1 or y2 <= y1:
            continue

        sanitized_bboxes.append([x1, y1, x2, y2])
        sanitized_labels.append(int(label))

    transformed = transform(image=image, bboxes=sanitized_bboxes, labels=sanitized_labels)

    out_image = transformed["image"]
    out_target = dict(target)

    out_boxes = transformed.get("bboxes", [])
    out_labels = transformed.get("labels", [])

    out_target["labels"] = torch.as_tensor(out_labels, dtype=torch.long)
    if out_boxes:
        out_target["boxes"] = torch.as_tensor(out_boxes, dtype=torch.float32)
    else:
        out_target.pop("boxes", None)

    return out_image, out_target


class COCODataset(Dataset):
    """Loads processed underwater inspection images and COCO-format labels."""

    def __init__(
        self,
        annotations_file: str | Path,
        images_dir: str | Path,
        transforms: Callable | None = None,
        task: str = "detection",
    ) -> None:
        self.images_dir = Path(images_dir)
        self.transforms = transforms
        self.task = task

        with open(annotations_file, "r", encoding="utf-8") as fh:
            coco = json.load(fh)

        self.images_meta: list[dict] = coco.get("images", [])
        self.annotations: list[dict] = coco.get("annotations", [])
        self.categories: list[dict] = coco.get("categories", [])

        self._ann_by_image: dict[int, list[dict]] = {}
        for ann in self.annotations:
            self._ann_by_image.setdefault(ann["image_id"], []).append(ann)

    def __len__(self) -> int:
        return len(self.images_meta)

    def __getitem__(self, idx: int) -> tuple:
        meta = self.images_meta[idx]
        image_path = self.images_dir / meta["file_name"]
        image = np.array(Image.open(image_path).convert("RGB"))

        anns = self._ann_by_image.get(meta["id"], [])
        target = self._build_target(anns, meta)

        if self.transforms:
            image, target = _apply_detection_transform(image, target, self.transforms)

        return image, target

    def _build_target(self, anns: list[dict], meta: dict) -> dict:
        boxes = []
        labels = []
        masks = []

        for ann in anns:
            labels.append(ann.get("category_id", 0))
            if self.task in ("detection", "segmentation"):
                x, y, w, h = ann["bbox"]
                boxes.append([x, y, x + w, y + h])
            if self.task == "segmentation" and "segmentation" in ann:
                masks.append(ann["segmentation"])

        target: dict = {
            "image_id": meta["id"],
            "labels": torch.as_tensor(labels, dtype=torch.long),
        }
        if boxes:
            target["boxes"] = torch.as_tensor(boxes, dtype=torch.float32)
        if masks:
            target["masks"] = masks

        return target

    @property
    def num_classes(self) -> int:
        return len(self.categories)

    @property
    def class_names(self) -> list[str]:
        return [c["name"] for c in sorted(self.categories, key=lambda c: c["id"])]


class YOLODataset(Dataset):
    """Loads split-scoped YOLO images and labels and returns detection targets."""

    def __init__(
        self,
        images_dir: str | Path,
        labels_dir: str | Path,
        class_names: list[str],
        transforms: Callable | None = None,
    ) -> None:
        self.images_dir = Path(images_dir)
        self.labels_dir = Path(labels_dir)
        self.class_names_list = class_names
        self.transforms = transforms

        self.image_paths = sorted(
            p for p in self.images_dir.glob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
        )

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> tuple:
        image_path = self.image_paths[idx]
        image = np.array(Image.open(image_path).convert("RGB"))
        img_h, img_w = image.shape[:2]

        label_path = self.labels_dir / f"{image_path.stem}.txt"
        boxes: list[list[float]] = []
        labels: list[int] = []

        if label_path.exists():
            for line in label_path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) != 5:
                    continue

                class_id = int(float(parts[0]))
                cx = float(parts[1])
                cy = float(parts[2])
                w = float(parts[3])
                h = float(parts[4])

                x1 = (cx - w / 2.0) * img_w
                y1 = (cy - h / 2.0) * img_h
                x2 = (cx + w / 2.0) * img_w
                y2 = (cy + h / 2.0) * img_h

                boxes.append([x1, y1, x2, y2])
                # Keep labels 1-indexed for compatibility with existing training stack.
                labels.append(class_id + 1)

        target = {
            "image_id": idx,
            "labels": torch.as_tensor(labels, dtype=torch.long),
        }
        if boxes:
            target["boxes"] = torch.as_tensor(boxes, dtype=torch.float32)

        if self.transforms:
            image, target = _apply_detection_transform(image, target, self.transforms)

        return image, target

    @property
    def num_classes(self) -> int:
        return len(self.class_names_list)

    @property
    def class_names(self) -> list[str]:
        return self.class_names_list


class UnderwaterDataset(Dataset):
    """Compatibility wrapper around COCO or YOLO dataset implementations."""

    def __new__(
        cls,
        annotations_file: str | Path | None,
        images_dir: str | Path,
        transforms: Callable | None = None,
        task: str = "detection",
        labels_dir: str | Path | None = None,
        split: str | None = None,
        class_names: list[str] | None = None,
    ):
        if labels_dir is not None:
            if class_names is None:
                class_names = DEFAULT_CLASS_NAMES
            return YOLODataset(
                images_dir=images_dir,
                labels_dir=labels_dir,
                class_names=class_names,
                transforms=transforms,
            )

        if annotations_file is None:
            raise ValueError("annotations_file is required for COCO loading")

        ann_path = Path(annotations_file)
        if ann_path.exists() and ann_path.suffix.lower() == ".json":
            return COCODataset(
                annotations_file=ann_path,
                images_dir=images_dir,
                transforms=transforms,
                task=task,
            )

        if split is None:
            split = ann_path.stem

        images_root = Path(images_dir)
        candidate_images_dir = images_root / split
        candidate_labels_dir = images_root.parent / "labels" / split

        if candidate_images_dir.exists() and candidate_labels_dir.exists():
            if class_names is None:
                class_names = DEFAULT_CLASS_NAMES
            return YOLODataset(
                images_dir=candidate_images_dir,
                labels_dir=candidate_labels_dir,
                class_names=class_names,
                transforms=transforms,
            )

        raise FileNotFoundError(
            f"Could not load dataset from annotations_file={annotations_file}. "
            "Expected existing COCO JSON or YOLO split under processed/images and processed/labels."
        )


def build_dataloader(
    annotations_file: str | Path | None,
    images_dir: str | Path,
    batch_size: int = 16,
    shuffle: bool = True,
    num_workers: int = 4,
    transforms: Callable | None = None,
    task: str = "detection",
    labels_dir: str | Path | None = None,
    split: str | None = None,
    class_names: list[str] | None = None,
) -> DataLoader:
    """Convenience factory returning a configured DataLoader.

    For YOLO mode, either pass ``labels_dir`` explicitly or pass an
    ``annotations_file`` like "train.json" and ensure split folders exist under
    ``images_dir/<split>`` and sibling ``labels/<split>``.
    """
    dataset = UnderwaterDataset(
        annotations_file=annotations_file,
        images_dir=images_dir,
        transforms=transforms,
        task=task,
        labels_dir=labels_dir,
        split=split,
        class_names=class_names,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=_collate_fn,
    )


def _collate_fn(batch: list) -> tuple:
    images, targets = zip(*batch)
    return list(images), list(targets)
