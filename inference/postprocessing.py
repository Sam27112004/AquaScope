"""
Postprocessing utilities for model outputs.

Includes NMS, box coordinate scaling, and confidence filtering.
"""

from __future__ import annotations

import torch
import torchvision


def apply_nms(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    labels: torch.Tensor,
    iou_threshold: float = 0.45,
    score_threshold: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Apply class-aware Non-Maximum Suppression.

    Args:
        boxes: Float tensor ``(N, 4)`` in xyxy format.
        scores: Float tensor ``(N,)`` of confidence scores.
        labels: Long tensor ``(N,)`` of class indices.
        iou_threshold: IoU threshold for NMS.
        score_threshold: Discard boxes below this confidence score.

    Returns:
        Tuple of filtered ``(boxes, scores, labels)``.
    """
    if boxes.numel() == 0:
        return boxes, scores, labels

    # Filter by score threshold first.
    if score_threshold > 0:
        keep_score = scores >= score_threshold
        boxes, scores, labels = boxes[keep_score], scores[keep_score], labels[keep_score]

    if boxes.numel() == 0:
        return boxes, scores, labels

    # Class-aware NMS: offset boxes by class index so different classes never suppress each other.
    max_coord = boxes.max()
    offsets = labels.float() * (max_coord + 1)
    boxes_shifted = boxes + offsets[:, None]

    keep = torchvision.ops.nms(boxes_shifted, scores, iou_threshold)
    return boxes[keep], scores[keep], labels[keep]


def scale_boxes(
    boxes: torch.Tensor,
    from_size: tuple[int, int],
    to_size: tuple[int, int],
) -> torch.Tensor:
    """Scale bounding boxes from one image resolution to another.

    Args:
        boxes: Float tensor ``(N, 4)`` in xyxy format.
        from_size: ``(height, width)`` of the source resolution.
        to_size: ``(height, width)`` of the target resolution.

    Returns:
        Scaled boxes tensor.
    """
    if boxes.numel() == 0:
        return boxes

    fh, fw = from_size
    th, tw = to_size
    scale_x = tw / fw
    scale_y = th / fh

    scale = torch.tensor([scale_x, scale_y, scale_x, scale_y], dtype=boxes.dtype, device=boxes.device)
    return boxes * scale


def filter_by_confidence(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    labels: torch.Tensor,
    threshold: float = 0.35,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Remove detections with scores below *threshold*."""
    keep = scores >= threshold
    return boxes[keep], scores[keep], labels[keep]
