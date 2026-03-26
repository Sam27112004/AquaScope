"""AquaScope namespace package shim.

This package maps the top-level project modules (config, datasets, models, etc.)
under the ``aquascope`` namespace so imports like ``aquascope.config...`` work
when running from source.
"""

from __future__ import annotations

from pathlib import Path

# Expose the repository root as this package's search path.
# This allows subpackages such as aquascope.config to resolve to ./config.
__path__ = [str(Path(__file__).resolve().parent.parent)]
