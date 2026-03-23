"""
ConfigManager — loads and deep-merges YAML configuration files.

Usage::

    cfg = ConfigManager.load("config/training_config.yaml")
    print(cfg.training.epochs)
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml


class _AttrDict(dict):
    """Dictionary subclass that allows attribute-style access."""

    def __getattr__(self, key: str) -> Any:
        try:
            value = self[key]
        except KeyError:
            raise AttributeError(f"Config has no key '{key}'") from None
        return _AttrDict(value) if isinstance(value, dict) else value

    def __setattr__(self, key: str, value: Any) -> None:
        self[key] = value


class ConfigManager:
    """Load, merge, and expose YAML configuration files."""

    BASE_CONFIG = Path(__file__).parent / "base_config.yaml"

    @classmethod
    def load(cls, config_path: str | Path, overrides: dict | None = None) -> _AttrDict:
        """Load *config_path* merged on top of base_config.yaml.

        Args:
            config_path: Path to the task-specific YAML file.
            overrides: Optional flat or nested dict of key=value overrides
                       applied after merging (highest priority).

        Returns:
            An :class:`_AttrDict` with attribute-style access.
        """
        base = cls._read_yaml(cls.BASE_CONFIG)
        task = cls._read_yaml(Path(config_path))
        merged = cls._deep_merge(base, task)
        if overrides:
            merged = cls._deep_merge(merged, overrides)
        return _AttrDict(merged)

    @classmethod
    def _read_yaml(cls, path: Path) -> dict:
        with path.open("r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
        # Drop the Hydra-style 'defaults' key — we handle merging manually.
        data.pop("defaults", None)
        return data

    @classmethod
    def _deep_merge(cls, base: dict, override: dict) -> dict:
        merged = copy.deepcopy(base)
        for key, value in override.items():
            if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
                merged[key] = cls._deep_merge(merged[key], value)
            else:
                merged[key] = copy.deepcopy(value)
        return merged
