"""
Pydantic request/response schemas for the Aquascope API.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator


class InferenceConfig(BaseModel):
    """Body schema for the ``POST /inference/load`` endpoint."""

    task: Literal["detection", "segmentation", "classification"] = Field(
        default="detection",
        description="Inference task type.",
    )
    model_slug: str = Field(
        default="yolo",
        description="Model identifier as registered in ModelRegistry.",
        examples=["yolo", "faster_rcnn", "unet", "resnet"],
    )
    num_classes: int = Field(
        default=10,
        ge=1,
        description="Number of output classes the model was trained with.",
    )
    checkpoint_path: str | None = Field(
        default=None,
        description="Absolute or relative path to a .pt checkpoint file.",
    )
    confidence_threshold: float = Field(
        default=0.35,
        ge=0.0,
        le=1.0,
        description="Minimum score to retain a detection.",
    )
    iou_threshold: float = Field(
        default=0.45,
        ge=0.0,
        le=1.0,
        description="IoU threshold for NMS.",
    )


class TrainingRequest(BaseModel):
    """Body schema for triggering a training run via the API."""

    task: Literal["detection", "segmentation", "classification"] = "detection"
    model_slug: str = "yolo"
    dataset_name: str = Field(..., description="Name of a processed dataset under datasets/processed/.")
    epochs: int = Field(default=50, ge=1, le=1000)
    batch_size: int = Field(default=16, ge=1)
    learning_rate: float = Field(default=1e-3, gt=0.0)
    experiment_name: str | None = None

    @field_validator("dataset_name")
    @classmethod
    def _no_traversal(cls, v: str) -> str:
        if ".." in v or "/" in v or "\\" in v:
            raise ValueError("dataset_name must be a plain folder name, not a path.")
        return v


class DetectionResult(BaseModel):
    """Response schema for a single-image detection."""

    model: str
    task: str
    boxes: list[list[float]] = Field(default_factory=list, description="[[x1,y1,x2,y2], ...]")
    scores: list[float] = Field(default_factory=list)
    labels: list[int] = Field(default_factory=list)
