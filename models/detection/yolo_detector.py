"""
YOLO-style single-stage object detector for underwater inspection.

This placeholder wires up a lightweight CSPDarkNet backbone with a
decoupled detection head. Replace or extend to integrate an official
YOLO implementation (e.g. ultralytics) as needed.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from aquascope.models.base_model import BaseModel
from aquascope.models.model_registry import ModelRegistry


@ModelRegistry.register("detection", "yolo")
class YOLODetector(BaseModel):
    """Lightweight YOLO-style detector.

    Args:
        num_classes: Number of target object categories.
        anchors_per_scale: Number of anchors per detection scale.
        config: Optional config object with extra hyper-parameters.
    """

    def __init__(
        self,
        num_classes: int,
        anchors_per_scale: int = 3,
        config: Any | None = None,
    ) -> None:
        self.anchors_per_scale = anchors_per_scale
        super().__init__(num_classes=num_classes, config=config)

    # ------------------------------------------------------------------
    # BaseModel interface
    # ------------------------------------------------------------------

    def build(self) -> None:
        # Backbone — placeholder CSPDarkNet stub
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.SiLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.SiLU(),
        )
        # Detection head — outputs (batch, anchors*(5+classes), H, W)
        out_channels = self.anchors_per_scale * (5 + self.num_classes)
        self.head = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        return self.head(features)

    def load_checkpoint(self, checkpoint_path: str | Path) -> None:
        payload = torch.load(checkpoint_path, map_location="cpu")
        self.load_state_dict(payload["state_dict"])
