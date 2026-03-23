"""
Faster R-CNN two-stage detector for underwater inspection.

Uses torchvision's Faster R-CNN with a ResNet-50 FPN backbone as the
default architecture. Override ``build()`` in a subclass to swap the
backbone or head.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from aquascope.models.base_model import BaseModel
from aquascope.models.model_registry import ModelRegistry


@ModelRegistry.register("detection", "faster_rcnn")
class FasterRCNNDetector(BaseModel):
    """Faster R-CNN detector pre-configured for underwater inspection.

    Args:
        num_classes: Number of target categories (background is included
                     automatically by torchvision, so pass the count of
                     *foreground* classes only).
        pretrained_backbone: If True, load ImageNet-pretrained backbone weights.
        config: Optional config object.
    """

    def __init__(
        self,
        num_classes: int,
        pretrained_backbone: bool = True,
        config: Any | None = None,
    ) -> None:
        self.pretrained_backbone = pretrained_backbone
        super().__init__(num_classes=num_classes, config=config)

    # ------------------------------------------------------------------
    # BaseModel interface
    # ------------------------------------------------------------------

    def build(self) -> None:
        weights = (
            torchvision.models.detection.FasterRCNN_ResNet50_FPN_Weights.DEFAULT
            if self.pretrained_backbone
            else None
        )
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=weights)

        # Replace the classifier head for our number of classes.
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(
            in_features, self.num_classes + 1  # +1 for background
        )
        # Store as backbone/head split for API consistency.
        self.backbone = model.backbone
        self.head = model.roi_heads
        # Keep the full model for forward pass convenience.
        self._model: FasterRCNN = model

    def forward(self, images: list, targets: list | None = None):
        return self._model(images, targets)

    def load_checkpoint(self, checkpoint_path: str | Path) -> None:
        payload = torch.load(checkpoint_path, map_location="cpu")
        self._model.load_state_dict(payload["state_dict"])
