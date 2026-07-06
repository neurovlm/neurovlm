"""Shared metric helpers used by multiple evaluation modules."""

from __future__ import annotations

import numpy as np
import torch


def as_latent_batch(latents) -> torch.Tensor:
    """Convert a latent tensor, array, or list of tensors/arrays into a 2D tensor batch."""

    if isinstance(latents, torch.Tensor):
        batch = latents.detach().cpu()
    elif isinstance(latents, np.ndarray):
        batch = torch.as_tensor(latents)
    else:
        batch = torch.stack([torch.as_tensor(x) for x in latents])
    if batch.dim() == 1:
        batch = batch.unsqueeze(0)
    return batch
