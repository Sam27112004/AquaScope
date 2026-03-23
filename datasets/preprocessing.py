"""
Preprocessing utilities for raw underwater inspection footage.

Handles frame extraction from video, image resizing, colour correction,
and conversion between common array and tensor formats.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def preprocess_frame(
    frame: np.ndarray,
    target_size: tuple[int, int] = (640, 640),
    normalize: bool = True,
) -> np.ndarray:
    """Resize and optionally normalise a single BGR frame from OpenCV.

    Args:
        frame: HxWxC uint8 BGR array as returned by ``cv2.VideoCapture``.
        target_size: ``(width, height)`` to resize to.
        normalize: If True, convert to float32 in [0, 1].

    Returns:
        Preprocessed HxWxC array.
    """
    resized = cv2.resize(frame, target_size, interpolation=cv2.INTER_LINEAR)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    if normalize:
        return rgb.astype(np.float32) / 255.0
    return rgb


def extract_frames(
    video_path: str | Path,
    output_dir: str | Path,
    frame_skip: int = 0,
    target_size: tuple[int, int] | None = None,
) -> list[Path]:
    """Extract frames from a video file and save them as JPEG images.

    Args:
        video_path: Path to the source video.
        output_dir: Directory to write extracted frames into.
        frame_skip: Save every ``frame_skip + 1``-th frame. 0 = every frame.
        target_size: Optional ``(width, height)`` resize before saving.

    Returns:
        List of paths to the saved frame images.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    saved: list[Path] = []
    frame_idx = 0
    save_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % (frame_skip + 1) == 0:
            if target_size:
                frame = cv2.resize(frame, target_size, interpolation=cv2.INTER_LINEAR)
            out_path = output_dir / f"frame_{save_idx:06d}.jpg"
            cv2.imwrite(str(out_path), frame)
            saved.append(out_path)
            save_idx += 1
        frame_idx += 1

    cap.release()
    return saved


def apply_clahe(frame: np.ndarray, clip_limit: float = 2.0, tile_size: int = 8) -> np.ndarray:
    """Apply CLAHE (Contrast Limited Adaptive Histogram Equalisation) to a BGR frame.

    Useful for improving visibility in dark or turbid underwater footage.
    """
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l_ch, a_ch, b_ch = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_size, tile_size))
    l_ch = clahe.apply(l_ch)
    enhanced = cv2.merge([l_ch, a_ch, b_ch])
    return cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
