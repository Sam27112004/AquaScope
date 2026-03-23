"""
Inference routes — POST endpoints for image and video analysis.
"""

from __future__ import annotations

import io
from pathlib import Path

import cv2
import numpy as np
from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from aquascope.api.schemas.request_schemas import InferenceConfig
from aquascope.utils.logging_utils import get_logger

logger = get_logger(__name__)
router = APIRouter()

# ---------------------------------------------------------------------------
# In-memory model store (populated at startup or via /load endpoint).
# ---------------------------------------------------------------------------
_LOADED_MODELS: dict[str, object] = {}


@router.post("/image", summary="Run detection/segmentation on a single image")
async def infer_image(
    file: UploadFile = File(..., description="Image file (JPEG, PNG, etc.)"),
    model_slug: str = "yolo",
    task: str = "detection",
) -> JSONResponse:
    """Accept an image upload and return model predictions.

    Returns a JSON response with detected bounding boxes, scores, and labels.
    """
    if model_slug not in _LOADED_MODELS:
        raise HTTPException(
            status_code=400,
            detail=f"Model '{model_slug}' is not loaded. POST to /inference/load first.",
        )

    raw_bytes = await file.read()
    nparr = np.frombuffer(raw_bytes, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if frame is None:
        raise HTTPException(status_code=422, detail="Cannot decode uploaded image.")

    pipeline = _LOADED_MODELS[model_slug]
    result = pipeline.predict(frame)  # type: ignore[attr-defined]

    return JSONResponse(
        content={
            "model": model_slug,
            "task": task,
            "boxes": result.get("boxes", []),
            "scores": result.get("scores", []),
            "labels": result.get("labels", []),
        }
    )


@router.post("/load", summary="Load a model checkpoint into memory")
async def load_model(cfg: InferenceConfig) -> dict:
    """Load a model specified by task, slug, and checkpoint path.

    The loaded pipeline is cached in ``_LOADED_MODELS`` for subsequent calls.
    """
    from aquascope.config.config_manager import ConfigManager
    from aquascope.inference.image_inference import ImageInferencePipeline
    from aquascope.models.model_registry import ModelRegistry

    try:
        config = ConfigManager.load("config/inference_config.yaml")
        model = ModelRegistry.build(cfg.task, cfg.model_slug, num_classes=cfg.num_classes)
        if cfg.checkpoint_path and Path(cfg.checkpoint_path).exists():
            model.load_checkpoint(cfg.checkpoint_path)
        pipeline = ImageInferencePipeline(model=model, config=config)
        _LOADED_MODELS[cfg.model_slug] = pipeline
        logger.info("Loaded model '%s' (%s) from '%s'.", cfg.model_slug, cfg.task, cfg.checkpoint_path)
        return {"status": "loaded", "model": cfg.model_slug}
    except Exception as exc:
        logger.exception("Failed to load model: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.get("/models", summary="List currently loaded models")
async def list_loaded_models() -> dict:
    return {"loaded_models": list(_LOADED_MODELS.keys())}
