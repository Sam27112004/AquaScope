"""
Trainer — task-agnostic training loop for Aquascope models.

Handles:
- Mixed-precision training via ``torch.amp``
- Learning-rate scheduling
- Periodic checkpoint saving
- MLflow / TensorBoard experiment logging (delegated to callbacks)
- Early stopping
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from aquascope.models.base_model import BaseModel
from aquascope.training.callbacks import CallbackList, EarlyStopping, CheckpointSaver
from aquascope.utils.logging_utils import get_logger

logger = get_logger(__name__)


class Trainer:
    """Generic training loop.

    Args:
        model: An instance of :class:`~aquascope.models.base_model.BaseModel`.
        optimizer: PyTorch optimiser already configured with model parameters.
        loss_fn: Callable ``(predictions, targets) -> scalar tensor``.
        scheduler: Optional LR scheduler (called once per epoch).
        config: Config object (from :class:`~aquascope.config.ConfigManager`).
        callbacks: Optional list of :class:`~aquascope.training.callbacks.Callback` instances.
    """

    def __init__(
        self,
        model: BaseModel,
        optimizer: torch.optim.Optimizer,
        loss_fn: Any,
        scheduler: Any | None = None,
        config: Any | None = None,
        callbacks: list | None = None,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.scheduler = scheduler
        self.config = config

        device_str = getattr(config, "device", "cpu") if config else "cpu"
        self.device = torch.device(device_str if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        mixed = getattr(config, "mixed_precision", False) if config else False
        self.scaler = torch.amp.GradScaler(enabled=mixed and self.device.type == "cuda")
        self.mixed_precision = mixed

        self.callbacks = CallbackList(callbacks or [])

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader | None = None,
        epochs: int = 10,
        max_train_batches: int | None = None,
        max_val_batches: int | None = None,
    ) -> dict[str, list[float]]:
        """Run the full training loop.

        Returns:
            History dict with ``train_loss`` and optionally ``val_loss`` lists.
        """
        history: dict[str, list[float]] = {"train_loss": [], "val_loss": []}
        self.callbacks.on_train_begin({"epochs": epochs, "model": self.model})

        for epoch in range(1, epochs + 1):
            self.callbacks.on_epoch_begin(epoch, {})
            train_loss = self._train_epoch(
                train_loader,
                epoch,
                epochs,
                max_batches=max_train_batches,
            )
            history["train_loss"].append(train_loss)

            logs: dict = {"epoch": epoch, "train_loss": train_loss}

            if val_loader is not None:
                val_loss = self._val_epoch(val_loader, max_batches=max_val_batches)
                history["val_loss"].append(val_loss)
                logs["val_loss"] = val_loss

            if self.scheduler is not None:
                self.scheduler.step()

            self.callbacks.on_epoch_end(epoch, logs)
            val_suffix = f"  val_loss: {logs.get('val_loss', 0):.4f}" if val_loader else ""
            logger.info(f"Epoch {epoch}/{epochs} — train_loss: {train_loss:.4f}{val_suffix}")

            if self.callbacks.should_stop_early():
                logger.info(f"Early stopping triggered at epoch {epoch}.")
                break

        self.callbacks.on_train_end(history)
        return history

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _train_epoch(
        self,
        loader: DataLoader,
        epoch: int,
        total_epochs: int,
        max_batches: int | None = None,
    ) -> float:
        self.model.train()
        running_loss = 0.0
        processed_batches = 0

        for batch_idx, (images, targets) in enumerate(loader):
            if max_batches is not None and batch_idx >= max_batches:
                break

            images = self._prepare_images(images)
            targets = self._to_device(targets)

            self.optimizer.zero_grad()
            with torch.amp.autocast(
                device_type=self.device.type, enabled=self.mixed_precision
            ):
                predictions = self.model(images)
                loss = self.loss_fn(predictions, targets)

            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            running_loss += loss.item()
            self.callbacks.on_batch_end(
                batch_idx, {"loss": loss.item(), "epoch": epoch, "total_epochs": total_epochs}
            )
            processed_batches += 1

        return running_loss / max(processed_batches, 1)

    @torch.no_grad()
    def _val_epoch(self, loader: DataLoader, max_batches: int | None = None) -> float:
        self.model.eval()
        running_loss = 0.0
        processed_batches = 0

        for batch_idx, (images, targets) in enumerate(loader):
            if max_batches is not None and batch_idx >= max_batches:
                break

            images = self._prepare_images(images)
            targets = self._to_device(targets)
            with torch.amp.autocast(
                device_type=self.device.type, enabled=self.mixed_precision
            ):
                predictions = self.model(images)
                loss = self.loss_fn(predictions, targets)
            running_loss += loss.item()
            processed_batches += 1

        return running_loss / max(processed_batches, 1)

    def _to_device(self, data: Any) -> Any:
        if isinstance(data, torch.Tensor):
            return data.to(self.device, non_blocking=True)
        if isinstance(data, list):
            return [self._to_device(d) for d in data]
        if isinstance(data, dict):
            return {k: self._to_device(v) for k, v in data.items()}
        return data

    def _prepare_images(self, images: Any) -> torch.Tensor:
        """Convert collated image batches to a tensor [B, C, H, W]."""
        if isinstance(images, torch.Tensor):
            return images.to(self.device, non_blocking=True)

        if isinstance(images, list):
            if not images:
                raise ValueError("Received empty image batch")

            first = images[0]
            if isinstance(first, torch.Tensor):
                return torch.stack([img.to(self.device, non_blocking=True) for img in images], dim=0)

            if isinstance(first, np.ndarray):
                tensors = [torch.from_numpy(img).permute(2, 0, 1).float() for img in images]
                return torch.stack(tensors, dim=0).to(self.device, non_blocking=True)

        raise TypeError(f"Unsupported image batch type: {type(images)}")
