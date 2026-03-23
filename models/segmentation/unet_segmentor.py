"""
U-Net segmentation model for underwater inspection.

Implements a standard encoder-decoder with skip connections.
Swap the ``_EncoderBlock`` or ``_DecoderBlock`` internals to use a
pretrained backbone (e.g. EfficientNet, ResNet) as the encoder.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from aquascope.models.base_model import BaseModel
from aquascope.models.model_registry import ModelRegistry


class _ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


@ModelRegistry.register("segmentation", "unet")
class UNetSegmentor(BaseModel):
    """U-Net encoder-decoder segmentor.

    Args:
        num_classes: Number of segmentation classes.
        base_channels: Feature channels at the first encoder level (doubles each level).
        config: Optional config object.
    """

    def __init__(
        self,
        num_classes: int,
        base_channels: int = 64,
        config: Any | None = None,
    ) -> None:
        self.base_channels = base_channels
        super().__init__(num_classes=num_classes, config=config)

    # ------------------------------------------------------------------
    # BaseModel interface
    # ------------------------------------------------------------------

    def build(self) -> None:
        c = self.base_channels
        self.backbone = nn.ModuleDict(
            {
                "enc1": _ConvBlock(3, c),
                "enc2": _ConvBlock(c, c * 2),
                "enc3": _ConvBlock(c * 2, c * 4),
                "bottleneck": _ConvBlock(c * 4, c * 8),
            }
        )
        self.pool = nn.MaxPool2d(2)
        self.head = nn.ModuleDict(
            {
                "up3": nn.ConvTranspose2d(c * 8, c * 4, 2, stride=2),
                "dec3": _ConvBlock(c * 8, c * 4),
                "up2": nn.ConvTranspose2d(c * 4, c * 2, 2, stride=2),
                "dec2": _ConvBlock(c * 4, c * 2),
                "up1": nn.ConvTranspose2d(c * 2, c, 2, stride=2),
                "dec1": _ConvBlock(c * 2, c),
                "out": nn.Conv2d(c, self.num_classes, 1),
            }
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        enc = self.backbone
        head = self.head

        e1 = enc["enc1"](x)
        e2 = enc["enc2"](self.pool(e1))
        e3 = enc["enc3"](self.pool(e2))
        b = enc["bottleneck"](self.pool(e3))

        d3 = head["dec3"](torch.cat([head["up3"](b), e3], dim=1))
        d2 = head["dec2"](torch.cat([head["up2"](d3), e2], dim=1))
        d1 = head["dec1"](torch.cat([head["up1"](d2), e1], dim=1))
        return head["out"](d1)

    def load_checkpoint(self, checkpoint_path: str | Path) -> None:
        payload = torch.load(checkpoint_path, map_location="cpu")
        self.load_state_dict(payload["state_dict"])
