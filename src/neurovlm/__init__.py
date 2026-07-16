"""Public NeuroVLM package interface.

``NeuroVLM`` is imported lazily so lightweight modules such as
``neurovlm.models`` and ``neurovlm.ale_cnn`` do not initialize the full
retrieval and language-model stack. ``from neurovlm import NeuroVLM`` keeps
the same public behavior as a direct import.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

__all__ = ["NeuroVLM"]

if TYPE_CHECKING:
    from .core import NeuroVLM


def __getattr__(name: str):
    if name == "NeuroVLM":
        from .core import NeuroVLM  # noqa: PLC0415
        return NeuroVLM
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
