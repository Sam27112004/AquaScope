"""
BaseModel — abstract base class for all Aquascope models.

Every concrete model (detectors, segmentors, classifiers) must subclass
:class:`BaseModel` and implement the three abstract methods.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn


class BaseModel(ABC, nn.Module):
    """Abstract model interface.

    Subclasses must implement:

    - :meth:`build` — construct and return the underlying ``nn.Module``.
    - :meth:`forward` — define the forward pass.
    - :meth:`load_checkpoint` — restore weights from a checkpoint file.
    """

    def __init__(self, num_classes: int, config: Any | None = None) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.config = config
        self.backbone: nn.Module | None = None
        self.head: nn.Module | None = None
        self.build()

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @abstractmethod
    def build(self) -> None:
        """Instantiate backbone and head layers and assign them to ``self``."""

    @abstractmethod
    def forward(self, x: torch.Tensor) -> Any:
        """Run the forward pass and return model-specific outputs."""

    @abstractmethod
    def load_checkpoint(self, checkpoint_path: str | Path) -> None:
        """Load model weights from *checkpoint_path*."""

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

    def save_checkpoint(self, path: str | Path, metadata: dict | None = None) -> None:
        """Save model state-dict and optional *metadata* to *path*."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload: dict = {
            "state_dict": self.state_dict(),
            "num_classes": self.num_classes,
            "model_class": self.__class__.__name__,
        }
        if metadata:
            payload.update(metadata)
        torch.save(payload, path)

    @property
    def device(self) -> torch.device:
        try:
            return next(self.parameters()).device
        except StopIteration:
            return torch.device("cpu")

    def param_count(self) -> dict[str, int]:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"total": total, "trainable": trainable}
