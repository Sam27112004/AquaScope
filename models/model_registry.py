"""
ModelRegistry — maps slug strings to concrete model classes.

Models self-register via the :func:`register` decorator so the training
and inference pipelines can instantiate them by name without hard-coded
imports.

Example::

    @ModelRegistry.register("detection", "my_detector")
    class MyDetector(BaseModel):
        ...

    model = ModelRegistry.build("detection", "my_detector", num_classes=5)
"""

from __future__ import annotations

import importlib
from typing import Type

from aquascope.models.base_model import BaseModel


class ModelRegistry:
    _registry: dict[str, dict[str, Type[BaseModel]]] = {}

    @classmethod
    def register(cls, task: str, slug: str):
        """Class decorator that registers *model_cls* under (*task*, *slug*)."""

        def decorator(model_cls: Type[BaseModel]) -> Type[BaseModel]:
            cls._registry.setdefault(task, {})[slug] = model_cls
            return model_cls

        return decorator

    @classmethod
    def build(cls, task: str, slug: str, num_classes: int, **kwargs) -> BaseModel:
        """Instantiate a registered model by *task* and *slug*.

        Raises:
            KeyError: If the (*task*, *slug*) combination is not registered.
        """
        cls._lazy_discover_models()
        try:
            model_cls = cls._registry[task][slug]
        except KeyError:
            available = {t: list(m.keys()) for t, m in cls._registry.items()}
            raise KeyError(
                f"Model '{slug}' for task '{task}' not found. "
                f"Available models: {available}"
            ) from None
        return model_cls(num_classes=num_classes, **kwargs)

    @classmethod
    def _lazy_discover_models(cls) -> None:
        """Import model modules so decorator-based registration runs."""
        importlib.import_module("aquascope.models.detection")
        importlib.import_module("aquascope.models.segmentation")
        importlib.import_module("aquascope.models.classification")

    @classmethod
    def list_models(cls) -> dict[str, list[str]]:
        """Return a dict mapping each task to its registered model slugs."""
        return {task: list(models.keys()) for task, models in cls._registry.items()}
