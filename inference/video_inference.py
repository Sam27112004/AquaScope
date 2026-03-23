"""
VideoInferencePipeline — frame-by-frame inference on underwater video footage.

Features:
- Frame skipping for faster processing.
- Optional per-frame annotation and video file export.
- Per-frame result accumulation for downstream analysis.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Generator

import cv2
import numpy as np
import torch

from aquascope.datasets.preprocessing import preprocess_frame
from aquascope.inference.postprocessing import apply_nms, scale_boxes
from aquascope.models.base_model import BaseModel
from aquascope.utils.logging_utils import get_logger
from aquascope.utils.visualization import draw_detections

logger = get_logger(__name__)


class VideoInferencePipeline:
    """Process a video file or camera stream through an Aquascope model.

    Args:
        model: Loaded :class:`~aquascope.models.base_model.BaseModel` instance.
        config: Config object with ``inference.*`` and ``video.*`` keys.
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

        device_str = str(getattr(config, "device", "cpu")) if config else "cpu"
        self.device = torch.device(device_str if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()

        inf = getattr(config, "inference", None) if config else None
        self.conf_threshold: float = getattr(inf, "confidence_threshold", 0.35) if inf else 0.35
        self.iou_threshold: float = getattr(inf, "iou_threshold", 0.45) if inf else 0.45

        vid = getattr(config, "video", None) if config else None
        self.frame_skip: int = getattr(vid, "frame_skip", 0) if vid else 0
        self.save_annotated: bool = getattr(vid, "save_annotated", True) if vid else True
        self.show_live: bool = getattr(vid, "show_live", False) if vid else False

        ds = getattr(config, "datasets", None) if config else None
        self.input_size: tuple[int, int] = tuple(getattr(ds, "image_size", [640, 640])[:2])  # type: ignore[assignment]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        source: str | Path | int,
        output_path: str | Path | None = None,
    ) -> list[dict]:
        """Process *source* and return a list of per-frame result dicts.

        Args:
            source: Path to a video file, or integer device index for webcam.
            output_path: Optional path to write an annotated output video.

        Returns:
            List of dicts — one per processed frame — with keys
            ``"frame_idx"``, ``"boxes"``, ``"scores"``, ``"labels"``.
        """
        cap = cv2.VideoCapture(str(source) if not isinstance(source, int) else source)
        if not cap.isOpened():
            raise FileNotFoundError(f"Cannot open video source: {source}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        writer: cv2.VideoWriter | None = None
        if output_path and self.save_annotated:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

        all_results: list[dict] = []
        frame_idx = 0

        try:
            for frame, idx in self._iter_frames(cap):
                result = self._process_frame(frame, idx, orig_w=width, orig_h=height)
                all_results.append(result)

                if writer or self.show_live:
                    annotated = draw_detections(frame, result, self.class_names)
                    if writer:
                        writer.write(annotated)
                    if self.show_live:
                        cv2.imshow("Aquascope — Live Inference", annotated)
                        if cv2.waitKey(1) & 0xFF == ord("q"):
                            break

                frame_idx = idx
        finally:
            cap.release()
            if writer:
                writer.release()
            if self.show_live:
                cv2.destroyAllWindows()

        logger.info("Processed %d frames from '%s'.", frame_idx + 1, source)
        return all_results

    def stream(self, source: str | Path | int) -> Generator[dict, None, None]:
        """Yield per-frame result dicts one at a time — useful for real-time pipelines."""
        cap = cv2.VideoCapture(str(source) if not isinstance(source, int) else source)
        if not cap.isOpened():
            raise FileNotFoundError(f"Cannot open video source: {source}")
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        try:
            for frame, idx in self._iter_frames(cap):
                yield self._process_frame(frame, idx, orig_w=width, orig_h=height)
        finally:
            cap.release()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _iter_frames(self, cap: cv2.VideoCapture) -> Generator[tuple[np.ndarray, int], None, None]:
        idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if idx % (self.frame_skip + 1) == 0:
                yield frame, idx
            idx += 1

    @torch.no_grad()
    def _process_frame(self, frame: np.ndarray, frame_idx: int, orig_w: int, orig_h: int) -> dict:
        w, h = self.input_size[1], self.input_size[0]
        processed = preprocess_frame(frame, target_size=(w, h), normalize=True)
        tensor = torch.from_numpy(processed).permute(2, 0, 1).unsqueeze(0).to(self.device)

        raw = self.model(tensor)

        # Placeholder postprocessing — extend for task-specific decoding.
        if isinstance(raw, torch.Tensor):
            boxes = torch.zeros((0, 4))
            scores = torch.zeros(0)
            labels = torch.zeros(0, dtype=torch.long)
            boxes = scale_boxes(boxes, self.input_size, (orig_h, orig_w))
            boxes, scores, labels = apply_nms(boxes, scores, labels, self.iou_threshold)
            return {"frame_idx": frame_idx, "boxes": boxes, "scores": scores, "labels": labels}

        # Faster R-CNN / two-stage models return list of dicts directly.
        result = raw[0] if isinstance(raw, list) else raw
        result["frame_idx"] = frame_idx
        return result
