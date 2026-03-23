"""
ImageInferencePipeline — run a model on one or more static images.

Supports single-image, batch, and directory inputs.  Results can be
returned as dicts or written to disk as annotated images.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

from aquascope.datasets.preprocessing import preprocess_frame
from aquascope.inference.postprocessing import apply_nms, scale_boxes
from aquascope.models.base_model import BaseModel
from aquascope.utils.logging_utils import get_logger
from aquascope.utils.visualization import draw_detections

logger = get_logger(__name__)

_IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}


class ImageInferencePipeline:
    """Run model inference on static images.

    Args:
        model: Loaded :class:`~aquascope.models.base_model.BaseModel` instance.
        config: Config object with ``inference.*`` keys.
        class_names: Optional list of class name strings for annotation.
    """

    def __init__(
        self,
        model: BaseModel,
        config: Any | None = None,
        class_names: list[str] | None = None,
    ) -> None:
        self.model = model
        self.config = config
        self.class_names = class_names or []

        device_str = getattr(getattr(config, "device", None), "__str__", lambda: "cpu")()
        self.device = torch.device(device_str if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()

        inf = getattr(config, "inference", None) if config else None
        self.conf_threshold: float = getattr(inf, "confidence_threshold", 0.35) if inf else 0.35
        self.iou_threshold: float = getattr(inf, "iou_threshold", 0.45) if inf else 0.45
        self.input_size: tuple[int, int] = tuple(getattr(getattr(config, "datasets", None), "image_size", [640, 640]))[:2]  # type: ignore[assignment]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def predict(self, source: str | Path | np.ndarray) -> dict:
        """Run inference on a single image path or numpy array.

        Returns:
            Dict with keys ``"boxes"``, ``"scores"``, ``"labels"``.
        """
        if isinstance(source, (str, Path)):
            frame = cv2.imread(str(source))
            if frame is None:
                raise FileNotFoundError(f"Cannot read image: {source}")
            orig_h, orig_w = frame.shape[:2]
        else:
            frame = source
            orig_h, orig_w = frame.shape[:2]

        tensor = self._preprocess(frame)
        with torch.no_grad():
            raw = self.model(tensor)

        result = self._postprocess(raw, orig_w=orig_w, orig_h=orig_h)
        return result

    def predict_directory(
        self,
        directory: str | Path,
        output_dir: str | Path | None = None,
    ) -> list[dict]:
        """Run inference on all images in *directory*.

        Optionally saves annotated images to *output_dir*.
        """
        directory = Path(directory)
        results = []
        image_paths = [p for p in directory.iterdir() if p.suffix.lower() in _IMG_EXTS]

        if output_dir:
            Path(output_dir).mkdir(parents=True, exist_ok=True)

        for img_path in image_paths:
            result = self.predict(img_path)
            result["image_path"] = str(img_path)
            results.append(result)

            if output_dir:
                frame = cv2.imread(str(img_path))
                annotated = draw_detections(frame, result, self.class_names)
                out_path = Path(output_dir) / img_path.name
                cv2.imwrite(str(out_path), annotated)
                logger.debug("Saved annotated image → %s", out_path)

        return results

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _preprocess(self, frame: np.ndarray) -> torch.Tensor:
        w, h = self.input_size[1], self.input_size[0]
        processed = preprocess_frame(frame, target_size=(w, h), normalize=True)
        tensor = torch.from_numpy(processed).permute(2, 0, 1).unsqueeze(0)
        return tensor.to(self.device, non_blocking=True)

    def _postprocess(self, raw: Any, orig_w: int, orig_h: int) -> dict:
        if not isinstance(raw, torch.Tensor):
            return raw  # Faster R-CNN already returns dicts
        boxes = torch.zeros((0, 4))
        scores = torch.zeros(0)
        labels = torch.zeros(0, dtype=torch.long)
        boxes = scale_boxes(boxes, self.input_size, (orig_h, orig_w))
        boxes, scores, labels = apply_nms(boxes, scores, labels, iou_threshold=self.iou_threshold)
        return {
            "boxes": boxes,
            "scores": scores,
            "labels": labels,
        }
