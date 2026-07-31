"""Public NeuroVLM package interface.

Public objects are imported lazily so importing :mod:`neurovlm` does not
initialize the retrieval, model-loading, or language-model stacks.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

__all__ = [
    "NeuroVLM",
    "AtlasFreeCNNDataset",
    "AtlasFreeCNNDataProvider",
    "atlas_free_cnn_splits",
    "NeuroVLMRuntime",
    "RuntimeMetadata",
    "load_pipeline",
]

if TYPE_CHECKING:
    from .data.atlas_free_dataset import (
        AtlasFreeCNNDataProvider,
        AtlasFreeCNNDataset,
        atlas_free_cnn_splits,
    )
    from .core import NeuroVLM
    from .core.runtime import NeuroVLMRuntime, RuntimeMetadata, load_pipeline


def __getattr__(name: str):
    if name == "NeuroVLM":
        from .core import NeuroVLM  # noqa: PLC0415

        return NeuroVLM
    if name in {"AtlasFreeCNNDataset", "AtlasFreeCNNDataProvider", "atlas_free_cnn_splits"}:
        from .data import atlas_free_dataset  # noqa: PLC0415

        return getattr(atlas_free_dataset, name)
    if name in {"NeuroVLMRuntime", "RuntimeMetadata", "load_pipeline"}:
        from .core import runtime  # noqa: PLC0415

        return getattr(runtime, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
