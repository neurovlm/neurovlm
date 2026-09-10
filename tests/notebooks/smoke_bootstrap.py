"""Runtime patches applied before executing a notebook's jupytext .py mirror in smoke mode.

Import side effects only: capping training to a handful of batches for a single
epoch lets a notebook's real code path (data loading, model construction,
training loop, checkpointing) run against real data without paying for a full
training run. Only active when NEUROVLM_SMOKE=1, so importing this module in a
non-smoke context is a no-op beyond the Agg backend switch.
"""

from __future__ import annotations

import dataclasses
import functools
import os

SMOKE = os.environ.get("NEUROVLM_SMOKE") == "1"
MAX_BATCHES = int(os.environ.get("NEUROVLM_SMOKE_MAX_BATCHES", "2"))

import matplotlib

matplotlib.use("Agg")

_TRAIN_FUNCS = [
    ("neurovlm.training.autoencoder", "train_autoencoder"),
    ("neurovlm.training.contrastive", "train_contrastive"),
    ("neurovlm.training.text_to_brain", "train_text_to_brain"),
    ("neurovlm.training.brain_to_text", "train_brain_to_text_generation"),
    ("neurovlm.training.mlp", "train_mlp_autoencoder"),
    ("neurovlm.training.mlp", "train_mlp_text_to_brain"),
    ("neurovlm.training.mlp", "train_mlp_contrastive"),
    ("neurovlm.training.mlp", "train_mlp_brain_to_text_retrieval"),
]


def _cap_epochs(fn):
    @functools.wraps(fn)
    def wrapped(config, *args, **kwargs):
        if getattr(config, "epochs", None) and config.epochs > 1:
            config = dataclasses.replace(config, epochs=1)
        return fn(config, *args, **kwargs)

    return wrapped


def _patch_training() -> None:
    import importlib

    import neurovlm.training as training_pkg

    for module_name, func_name in _TRAIN_FUNCS:
        module = importlib.import_module(module_name)
        wrapped = _cap_epochs(getattr(module, func_name))
        setattr(module, func_name, wrapped)
        setattr(training_pkg, func_name, wrapped)


def _patch_dataloader() -> None:
    from torch.utils.data import DataLoader

    original_iter = DataLoader.__iter__

    def capped_iter(self):
        for index, batch in enumerate(original_iter(self)):
            if index >= MAX_BATCHES:
                break
            yield batch

    DataLoader.__iter__ = capped_iter


if SMOKE:
    _patch_training()
    _patch_dataloader()
