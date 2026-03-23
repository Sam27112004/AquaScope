"""
Albumentations-based augmentation pipelines for underwater imagery.

Underwater images often suffer from colour distortion, low contrast, and
particulate haze — the augmentations here are tuned to simulate and
improve robustness against those degradations.
"""

from __future__ import annotations

from typing import Any

import albumentations as A
from albumentations.pytorch import ToTensorV2


def build_augmentation_pipeline(
    image_size: tuple[int, int] = (640, 640),
    mode: str = "train",
) -> A.Compose:
    """Build an Albumentations pipeline for the given *mode*.

    Args:
        image_size: Target ``(height, width)`` after resize.
        mode: One of ``"train"`` or ``"val"``/``"test"``.

    Returns:
        A configured :class:`albumentations.Compose` transform.
    """
    h, w = image_size

    if mode == "train":
        return A.Compose(
            [
                A.RandomResizedCrop(height=h, width=w, scale=(0.7, 1.0)),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.1),
                A.RandomRotate90(p=0.2),
                # Underwater colour correction simulation
                A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1, p=0.5),
                A.RandomFog(fog_coef_lower=0.05, fog_coef_upper=0.2, p=0.2),
                A.GaussianBlur(blur_limit=(3, 5), p=0.2),
                A.GaussNoise(var_limit=(10.0, 50.0), p=0.2),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ],
            bbox_params=A.BboxParams(format="pascal_voc", label_fields=["labels"]),
        )

    # val / test — deterministic resize + normalise only
    return A.Compose(
        [
            A.Resize(height=h, width=w),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ],
        bbox_params=A.BboxParams(format="pascal_voc", label_fields=["labels"]),
    )


def apply_transforms(
    image: Any,
    bboxes: list,
    labels: list,
    transform: A.Compose,
) -> dict:
    """Apply an Albumentations *transform* to a single sample.

    Returns a dict with keys ``image``, ``bboxes``, and ``labels``.
    """
    result = transform(image=image, bboxes=bboxes, labels=labels)
    return {
        "image": result["image"],
        "bboxes": result["bboxes"],
        "labels": result["labels"],
    }
