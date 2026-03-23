"""
Loss functions for detection, segmentation, and classification tasks.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ClassificationLoss(nn.Module):
    """Standard cross-entropy loss with optional label smoothing."""

    def __init__(self, label_smoothing: float = 0.0) -> None:
        super().__init__()
        self.criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.criterion(logits, targets)


class DetectionLoss(nn.Module):
    """Composite detection loss: CIoU bounding-box regression + BCE objectness + CE class.

    This is a simplified reference implementation.  For production use,
    replace with a complete YOLO-style or DETR-style loss that handles
    anchor assignment and multi-scale outputs.

    Args:
        box_weight: Weight for the box regression term.
        obj_weight: Weight for the objectness term.
        cls_weight: Weight for the classification term.
    """

    def __init__(
        self,
        box_weight: float = 3.54,
        obj_weight: float = 64.3,
        cls_weight: float = 0.5,
    ) -> None:
        super().__init__()
        self.box_weight = box_weight
        self.obj_weight = obj_weight
        self.cls_weight = cls_weight
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, predictions: torch.Tensor, targets: dict) -> torch.Tensor:
        # Placeholder: returns zero loss until anchor assignment is wired up.
        # Replace with full YOLO/DETR assignment logic for real training.
        return predictions.sum() * 0.0


class SegmentationLoss(nn.Module):
    """Combined Dice + Binary Cross-Entropy loss for semantic segmentation.

    Args:
        dice_weight: Contribution of the Dice term.
        bce_weight: Contribution of the BCE term.
        smooth: Smoothing factor to avoid division by zero in Dice.
    """

    def __init__(
        self,
        dice_weight: float = 0.5,
        bce_weight: float = 0.5,
        smooth: float = 1.0,
    ) -> None:
        super().__init__()
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        bce = F.binary_cross_entropy_with_logits(logits, masks.float())

        flat_pred = probs.view(probs.size(0), -1)
        flat_true = masks.view(masks.size(0), -1).float()
        intersection = (flat_pred * flat_true).sum(dim=1)
        dice = 1.0 - (2.0 * intersection + self.smooth) / (
            flat_pred.sum(dim=1) + flat_true.sum(dim=1) + self.smooth
        )

        return self.bce_weight * bce + self.dice_weight * dice.mean()
