"""
Evaluation metrics for detection, segmentation, and classification.
"""

from __future__ import annotations

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------

def compute_iou(box_a: torch.Tensor, box_b: torch.Tensor) -> torch.Tensor:
    """Compute pairwise IoU between two sets of boxes (xyxy format).

    Args:
        box_a: Tensor of shape ``(N, 4)``.
        box_b: Tensor of shape ``(M, 4)``.

    Returns:
        IoU matrix of shape ``(N, M)``.
    """
    area_a = (box_a[:, 2] - box_a[:, 0]) * (box_a[:, 3] - box_a[:, 1])
    area_b = (box_b[:, 2] - box_b[:, 0]) * (box_b[:, 3] - box_b[:, 1])

    inter_x1 = torch.max(box_a[:, None, 0], box_b[None, :, 0])
    inter_y1 = torch.max(box_a[:, None, 1], box_b[None, :, 1])
    inter_x2 = torch.min(box_a[:, None, 2], box_b[None, :, 2])
    inter_y2 = torch.min(box_a[:, None, 3], box_b[None, :, 3])

    inter_area = (inter_x2 - inter_x1).clamp(0) * (inter_y2 - inter_y1).clamp(0)
    union_area = area_a[:, None] + area_b[None, :] - inter_area
    return inter_area / union_area.clamp(min=1e-6)


def compute_detection_metrics(
    predictions: list[dict],
    ground_truths: list[dict],
    iou_threshold: float = 0.5,
    num_classes: int | None = None,
) -> dict[str, float]:
    """Compute per-class AP and mean AP (mAP@iou_threshold).

    Each element of *predictions* and *ground_truths* should be a dict with:
    - ``"boxes"`` — FloatTensor ``(N, 4)`` in xyxy format
    - ``"labels"`` — LongTensor ``(N,)``
    - ``"scores"`` — FloatTensor ``(N,)``  [predictions only]

    Returns:
        Dict with ``"mAP"`` and per-class ``"AP_<label>"`` entries.
    """
    if num_classes is None:
        all_labels = [l.item() for gt in ground_truths for l in gt["labels"]]
        num_classes = max(all_labels) + 1 if all_labels else 0

    aps: list[float] = []
    results: dict[str, float] = {}

    for cls in range(num_classes):
        tp_list, fp_list, n_gt = [], [], 0

        for pred, gt in zip(predictions, ground_truths):
            gt_mask = gt["labels"] == cls
            pred_mask = pred["labels"] == cls

            gt_boxes = gt["boxes"][gt_mask]
            pred_boxes = pred["boxes"][pred_mask]
            pred_scores = pred["scores"][pred_mask]
            n_gt += gt_boxes.shape[0]

            if pred_boxes.shape[0] == 0:
                continue

            order = pred_scores.argsort(descending=True)
            pred_boxes = pred_boxes[order]
            matched = torch.zeros(gt_boxes.shape[0], dtype=torch.bool)

            for pb in pred_boxes:
                if gt_boxes.shape[0] > 0:
                    ious = compute_iou(pb.unsqueeze(0), gt_boxes)[0]
                    best_idx = ious.argmax()
                    if ious[best_idx] >= iou_threshold and not matched[best_idx]:
                        tp_list.append(1)
                        fp_list.append(0)
                        matched[best_idx] = True
                        continue
                tp_list.append(0)
                fp_list.append(1)

        if n_gt == 0:
            continue

        tp_arr = np.cumsum(tp_list)
        fp_arr = np.cumsum(fp_list)
        recall = tp_arr / n_gt
        precision = tp_arr / (tp_arr + fp_arr + 1e-9)
        ap = float(np.trapz(precision, recall)) if len(recall) > 1 else 0.0
        aps.append(ap)
        results[f"AP_{cls}"] = ap

    results["mAP"] = float(np.mean(aps)) if aps else 0.0
    return results


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------

def compute_classification_metrics(
    predictions: torch.Tensor,
    targets: torch.Tensor,
) -> dict[str, float]:
    """Compute top-1 accuracy and top-5 accuracy.

    Args:
        predictions: Logit tensor ``(N, num_classes)``.
        targets: Ground-truth class indices ``(N,)``.
    """
    with torch.no_grad():
        top1 = (predictions.argmax(dim=1) == targets).float().mean().item()
        k = min(5, predictions.shape[1])
        top5_pred = predictions.topk(k, dim=1).indices
        top5 = (top5_pred == targets.unsqueeze(1)).any(dim=1).float().mean().item()
    return {"top1_accuracy": top1, "top5_accuracy": top5}
