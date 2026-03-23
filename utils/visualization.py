"""
Visualisation helpers for bounding boxes, segmentation masks, and metrics.
"""

from __future__ import annotations

import cv2
import numpy as np
import torch

# A palette of BGR colours — one per class index.
_PALETTE = [
    (0, 200, 255),   # amber
    (0, 80, 255),    # red
    (0, 255, 80),    # green
    (255, 80, 0),    # blue
    (200, 0, 255),   # purple
    (0, 255, 200),   # teal
    (255, 200, 0),   # cyan
    (80, 255, 0),    # lime
    (255, 0, 200),   # magenta
    (128, 128, 255), # lavender
]


def _get_colour(label: int) -> tuple[int, int, int]:
    return _PALETTE[label % len(_PALETTE)]


def draw_detections(
    frame: np.ndarray,
    result: dict,
    class_names: list[str] | None = None,
    thickness: int = 2,
    font_scale: float = 0.55,
) -> np.ndarray:
    """Draw bounding boxes and labels onto *frame* (in-place copy).

    Args:
        frame: BGR numpy array as returned by OpenCV.
        result: Dict with ``"boxes"``, ``"scores"``, and ``"labels"`` keys.
                Values may be PyTorch tensors or plain Python lists.
        class_names: Optional list mapping label index → class name string.
        thickness: Line thickness in pixels.
        font_scale: Font scale for label text.

    Returns:
        Annotated BGR numpy array.
    """
    annotated = frame.copy()
    boxes = result.get("boxes", [])
    scores = result.get("scores", [])
    labels = result.get("labels", [])

    # Convert tensors to lists for indexing.
    if isinstance(boxes, torch.Tensor):
        boxes = boxes.cpu().tolist()
    if isinstance(scores, torch.Tensor):
        scores = scores.cpu().tolist()
    if isinstance(labels, torch.Tensor):
        labels = labels.cpu().tolist()

    for box, score, label in zip(boxes, scores, labels):
        x1, y1, x2, y2 = map(int, box)
        colour = _get_colour(int(label))
        cv2.rectangle(annotated, (x1, y1), (x2, y2), colour, thickness)

        name = class_names[int(label)] if class_names and int(label) < len(class_names) else str(int(label))
        text = f"{name} {float(score):.2f}"
        (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
        cv2.rectangle(annotated, (x1, y1 - th - baseline - 4), (x1 + tw, y1), colour, -1)
        cv2.putText(
            annotated, text, (x1, y1 - baseline - 2),
            cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 1, cv2.LINE_AA
        )

    return annotated


def draw_segmentation_mask(
    frame: np.ndarray,
    mask: np.ndarray,
    label: int = 0,
    alpha: float = 0.45,
) -> np.ndarray:
    """Overlay a binary segmentation *mask* on *frame*.

    Args:
        frame: BGR numpy array.
        mask: Binary 2-D uint8 array (H x W), same spatial size as *frame*.
        label: Class index used to pick a colour from the palette.
        alpha: Blend factor for the mask overlay (0 = invisible, 1 = solid).

    Returns:
        Annotated BGR numpy array.
    """
    annotated = frame.copy()
    colour = _get_colour(label)
    overlay = np.zeros_like(frame, dtype=np.uint8)
    overlay[mask.astype(bool)] = colour
    cv2.addWeighted(overlay, alpha, annotated, 1.0 - alpha, 0, annotated)
    return annotated
