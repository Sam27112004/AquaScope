from aquascope.models.base_model import BaseModel
from aquascope.models.model_registry import ModelRegistry
from aquascope.models import detection as _detection_models  # noqa: F401
from aquascope.models import segmentation as _segmentation_models  # noqa: F401
from aquascope.models import classification as _classification_models  # noqa: F401

__all__ = ["BaseModel", "ModelRegistry"]
