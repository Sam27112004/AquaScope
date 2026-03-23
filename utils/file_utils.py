"""
File and path utilities for Aquascope.
"""

from __future__ import annotations

import hashlib
import os
import shutil
from pathlib import Path

_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
_VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".wmv"}


def ensure_dir(path: str | Path) -> Path:
    """Create *path* and all parents if they do not exist. Returns the Path."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def list_images(directory: str | Path, recursive: bool = True) -> list[Path]:
    """Return all image files under *directory*, sorted by name.

    Args:
        directory: Root directory to search.
        recursive: If True, search all subdirectories.
    """
    directory = Path(directory)
    glob = directory.rglob if recursive else directory.glob
    return sorted(p for p in glob("*") if p.suffix.lower() in _IMAGE_EXTS)


def list_videos(directory: str | Path, recursive: bool = True) -> list[Path]:
    """Return all video files under *directory*, sorted by name."""
    directory = Path(directory)
    glob = directory.rglob if recursive else directory.glob
    return sorted(p for p in glob("*") if p.suffix.lower() in _VIDEO_EXTS)


def file_md5(path: str | Path, chunk_size: int = 65536) -> str:
    """Compute the MD5 hex digest of a file without loading it fully into memory."""
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def safe_copy(src: str | Path, dst: str | Path, overwrite: bool = False) -> Path:
    """Copy *src* to *dst*, optionally refusing to overwrite existing files.

    Returns the destination Path.
    """
    src, dst = Path(src), Path(dst)
    if dst.exists() and not overwrite:
        raise FileExistsError(f"Destination already exists: {dst}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return dst


def clean_dir(directory: str | Path) -> None:
    """Delete all contents of *directory* without removing the directory itself."""
    directory = Path(directory)
    for item in directory.iterdir():
        if item.is_dir():
            shutil.rmtree(item)
        else:
            item.unlink()
