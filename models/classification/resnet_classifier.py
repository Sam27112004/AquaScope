"""
ResNet-based image classifier for underwater inspection.

Uses torchvision's ResNet family with a replaced classification head.
Useful for scene-level tagging (e.g. "corrosion", "biofouling", "clear").
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torchvision.models as tv_models

from aquascope.models.base_model import BaseModel
from aquascope.models.model_registry import ModelRegistry

_RESNET_VARIANTS = {
    "resnet18": (tv_models.resnet18, tv_models.ResNet18_Weights.DEFAULT),
    "resnet34": (tv_models.resnet34, tv_models.ResNet34_Weights.DEFAULT),
    "resnet50": (tv_models.resnet50, tv_models.ResNet50_Weights.DEFAULT),
}


@ModelRegistry.register("classification", "resnet")
class ResNetClassifier(BaseModel):
    """ResNet classifier with a swappable backbone variant.

    Args:
        num_classes: Number of output classes.
        variant: One of ``"resnet18"``, ``"resnet34"``, ``"resnet50"``.
        pretrained: Use ImageNet pre-trained weights for the backbone.
        freeze_backbone: If True, freeze all backbone parameters and only
                         train the final fully-connected layer.
        config: Optional config object.
    """

    def __init__(
        self,
        num_classes: int,
        variant: str = "resnet50",
        pretrained: bool = True,
        freeze_backbone: bool = False,
        config: Any | None = None,
    ) -> None:
        if variant not in _RESNET_VARIANTS:
            raise ValueError(f"Unknown variant '{variant}'. Choose from {list(_RESNET_VARIANTS)}.")
        self.variant = variant
        self.pretrained = pretrained
        self.freeze_backbone = freeze_backbone
        super().__init__(num_classes=num_classes, config=config)

    # ------------------------------------------------------------------
    # BaseModel interface
    # ------------------------------------------------------------------

    def build(self) -> None:
        factory, weights = _RESNET_VARIANTS[self.variant]
        model = factory(weights=weights if self.pretrained else None)

        if self.freeze_backbone:
            for param in model.parameters():
                param.requires_grad = False

        # Replace the FC head.
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, self.num_classes)

        self.backbone = nn.Sequential(*list(model.children())[:-1])  # strip FC
        self.head = model.fc
        self._model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._model(x)

    def load_checkpoint(self, checkpoint_path: str | Path) -> None:
        payload = torch.load(checkpoint_path, map_location="cpu")
        self._model.load_state_dict(payload["state_dict"])
