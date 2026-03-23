"""
UnderwaterDataset — PyTorch Dataset for underwater inspection imagery.

Expects the following on-disk layout::

    datasets/
        annotations/
            train.json   ← COCO-format annotation file
            val.json
        processed/
            images/
                <image_id>.jpg / .png
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset


class UnderwaterDataset(Dataset):
    """Loads processed underwater inspection images and COCO-format labels.

    Args:
        annotations_file: Path to a COCO-format JSON annotation file.
        images_dir: Directory containing the image files.
        transforms: Optional callable applied to (image, target) pairs.
        task: One of ``"detection"``, ``"segmentation"``, ``"classification"``.
    """

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

        # Index annotations by image_id for fast lookup.
        self._ann_by_image: dict[int, list[dict]] = {}
        for ann in self.annotations:
            self._ann_by_image.setdefault(ann["image_id"], []).append(ann)

    # ------------------------------------------------------------------
    # Dataset contract
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.images_meta)

    def __getitem__(self, idx: int) -> tuple:
        meta = self.images_meta[idx]
        image_path = self.images_dir / meta["file_name"]
        image = np.array(Image.open(image_path).convert("RGB"))

        anns = self._ann_by_image.get(meta["id"], [])
        target = self._build_target(anns, meta)

        if self.transforms:
            image, target = self.transforms(image, target)

        return image, target

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_target(self, anns: list[dict], meta: dict) -> dict:
        """Convert COCO annotations to a task-appropriate target dict."""
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
            target["masks"] = masks  # further processing in transforms

        return target

    @property
    def num_classes(self) -> int:
        return len(self.categories)

    @property
    def class_names(self) -> list[str]:
        return [c["name"] for c in sorted(self.categories, key=lambda c: c["id"])]


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_dataloader(
    annotations_file: str | Path,
    images_dir: str | Path,
    batch_size: int = 16,
    shuffle: bool = True,
    num_workers: int = 4,
    transforms: Callable | None = None,
    task: str = "detection",
) -> DataLoader:
    """Convenience factory returning a configured :class:`DataLoader`."""
    dataset = UnderwaterDataset(
        annotations_file=annotations_file,
        images_dir=images_dir,
        transforms=transforms,
        task=task,
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
