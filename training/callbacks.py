"""
Training callbacks — modular hooks called by :class:`~aquascope.training.trainer.Trainer`.

Implement the :class:`Callback` base class to add custom behaviour such as
experiment logging, model checkpointing, or early stopping.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from aquascope.utils.logging_utils import get_logger

logger = get_logger(__name__)


class Callback:
    """Base callback — all methods are no-ops by default."""

    def on_train_begin(self, logs: dict) -> None: ...
    def on_train_end(self, logs: dict) -> None: ...
    def on_epoch_begin(self, epoch: int, logs: dict) -> None: ...
    def on_epoch_end(self, epoch: int, logs: dict) -> None: ...
    def on_batch_end(self, batch: int, logs: dict) -> None: ...
    def should_stop(self) -> bool:
        return False


class CallbackList:
    """Container that dispatches events to a list of :class:`Callback` objects."""

    def __init__(self, callbacks: list[Callback]) -> None:
        self._callbacks = callbacks

    def on_train_begin(self, logs: dict) -> None:
        for cb in self._callbacks:
            cb.on_train_begin(logs)

    def on_train_end(self, logs: dict) -> None:
        for cb in self._callbacks:
            cb.on_train_end(logs)

    def on_epoch_begin(self, epoch: int, logs: dict) -> None:
        for cb in self._callbacks:
            cb.on_epoch_begin(epoch, logs)

    def on_epoch_end(self, epoch: int, logs: dict) -> None:
        for cb in self._callbacks:
            cb.on_epoch_end(epoch, logs)

    def on_batch_end(self, batch: int, logs: dict) -> None:
        for cb in self._callbacks:
            cb.on_batch_end(batch, logs)

    def should_stop_early(self) -> bool:
        return any(cb.should_stop() for cb in self._callbacks)


class EarlyStopping(Callback):
    """Stop training when *monitor* metric stops improving.

    Args:
        monitor: Key to watch in the epoch logs (e.g. ``"val_loss"``).
        patience: Number of epochs to wait without improvement before stopping.
        min_delta: Minimum change to qualify as an improvement.
        mode: ``"min"`` to stop when metric stops decreasing; ``"max"`` for increasing.
    """

    def __init__(
        self,
        monitor: str = "val_loss",
        patience: int = 10,
        min_delta: float = 1e-4,
        mode: str = "min",
    ) -> None:
        self.monitor = monitor
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self._best: float = float("inf") if mode == "min" else float("-inf")
        self._wait = 0
        self._stop = False

    def on_epoch_end(self, epoch: int, logs: dict) -> None:
        value = logs.get(self.monitor)
        if value is None:
            return

        improved = (
            value < self._best - self.min_delta
            if self.mode == "min"
            else value > self._best + self.min_delta
        )
        if improved:
            self._best = value
            self._wait = 0
        else:
            self._wait += 1
            if self._wait >= self.patience:
                self._stop = True
                logger.info(
                    f"EarlyStopping: '{self.monitor}' has not improved for {self.patience} epochs. "
                    f"Best: {self._best:.6f}"
                )

    def should_stop(self) -> bool:
        return self._stop


class CheckpointSaver(Callback):
    """Save model checkpoints to disk during training.

    Args:
        checkpoint_dir: Directory to write checkpoint files into.
        monitor: Metric to track for "best" checkpoint selection.
        mode: ``"min"`` or ``"max"`` (direction of improvement for *monitor*).
        save_best_only: If True, only overwrite the checkpoint when the metric improves.
        filename_template: f-string template with ``epoch`` and metric key available.
    """

    def __init__(
        self,
        checkpoint_dir: str | Path,
        monitor: str = "val_loss",
        mode: str = "min",
        save_best_only: bool = True,
        filename_template: str = "epoch_{epoch:03d}.pt",
    ) -> None:
        self.checkpoint_dir = Path(checkpoint_dir)
        self.monitor = monitor
        self.mode = mode
        self.save_best_only = save_best_only
        self.filename_template = filename_template
        self._best: float = float("inf") if mode == "min" else float("-inf")
        self._model: Any = None

    def on_train_begin(self, logs: dict) -> None:
        self._model = logs.get("model")
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def on_epoch_end(self, epoch: int, logs: dict) -> None:
        if self._model is None:
            return

        value = logs.get(self.monitor)
        improved = value is not None and (
            value < self._best if self.mode == "min" else value > self._best
        )

        if not self.save_best_only or improved:
            if improved and value is not None:
                self._best = value
                best_path = self.checkpoint_dir / "best.pt"
                self._model.save_checkpoint(best_path, metadata={"epoch": epoch, self.monitor: value})
                logger.info(f"Saved best checkpoint -> {best_path} ({value:.6f})")

            fmt_logs = dict(logs)
            fmt_logs.pop("epoch", None)
            fname = self.filename_template.format(epoch=epoch, **fmt_logs)
            path = self.checkpoint_dir / fname
            self._model.save_checkpoint(path, metadata={"epoch": epoch})
