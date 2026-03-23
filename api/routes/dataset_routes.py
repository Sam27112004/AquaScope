"""
Dataset management routes — list, upload, and inspect datasets.
"""

from __future__ import annotations

import shutil
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from aquascope.utils.logging_utils import get_logger

logger = get_logger(__name__)
router = APIRouter()

_UPLOAD_DIR = Path("datasets/raw/uploads")


@router.get("/", summary="List available processed datasets")
async def list_datasets() -> dict:
    """Return folder names found under ``datasets/processed/``."""
    processed = Path("datasets/processed")
    if not processed.exists():
        return {"datasets": []}
    entries = [p.name for p in processed.iterdir() if p.is_dir()]
    return {"datasets": entries}


@router.post("/upload", summary="Upload a raw image or video file")
async def upload_file(file: UploadFile = File(...)) -> JSONResponse:
    """Save an uploaded file to the raw uploads staging directory.

    Accepted file types: images and video files only.
    """
    _ALLOWED_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".mp4", ".avi", ".mov", ".mkv"}
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in _ALLOWED_SUFFIXES:
        raise HTTPException(
            status_code=415,
            detail=f"File type '{suffix}' is not supported. Allowed: {sorted(_ALLOWED_SUFFIXES)}",
        )

    _UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    dest = _UPLOAD_DIR / (file.filename or "upload")

    with dest.open("wb") as fh:
        shutil.copyfileobj(file.file, fh)

    logger.info("Uploaded file saved → %s", dest)
    return JSONResponse(
        status_code=201,
        content={"status": "uploaded", "path": str(dest), "filename": dest.name},
    )


@router.get("/{dataset_name}/info", summary="Get metadata for a specific dataset")
async def dataset_info(dataset_name: str) -> dict:
    """Return basic statistics for a named dataset under ``datasets/processed/``."""
    dataset_dir = Path("datasets/processed") / dataset_name
    if not dataset_dir.exists():
        raise HTTPException(status_code=404, detail=f"Dataset '{dataset_name}' not found.")

    image_exts = {".jpg", ".jpeg", ".png", ".bmp"}
    image_count = sum(1 for p in dataset_dir.rglob("*") if p.suffix.lower() in image_exts)
    ann_file = dataset_dir / "annotations.json"

    return {
        "name": dataset_name,
        "path": str(dataset_dir),
        "image_count": image_count,
        "has_annotations": ann_file.exists(),
    }
