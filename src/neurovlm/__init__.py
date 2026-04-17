from __future__ import annotations


def __getattr__(name: str):
    if name == "NeuroVLM":
        from .core import NeuroVLM  # noqa: PLC0415
        return NeuroVLM
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")