"""Shared metric helpers used by multiple evaluation modules."""

from __future__ import annotations

import numpy as np
import torch
from torch.nn import functional as F


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


def project_brain_latents_to_shared(
    nvlm,
    brain_latents,
    batch_size: int = 4096,
) -> torch.Tensor:
    """Project autoencoder latents into the normalized shared image space."""

    nvlm._ensure_projection_heads()
    batch = as_latent_batch(brain_latents).float()
    chunks = []
    with torch.no_grad():
        for start in range(0, len(batch), batch_size):
            z = nvlm._proj_head_image_infonce(
                batch[start : start + batch_size].to(nvlm.device)
            )
            chunks.append(F.normalize(z.float(), dim=1, eps=1e-8).detach().cpu())
    return torch.cat(chunks, dim=0)


def project_text_latents_to_shared(
    nvlm,
    text_latents,
    batch_size: int = 4096,
) -> torch.Tensor:
    """Project SPECTER latents into the normalized shared text space."""

    nvlm._ensure_projection_heads()
    batch = as_latent_batch(text_latents).float()
    chunks = []
    with torch.no_grad():
        for start in range(0, len(batch), batch_size):
            text_batch = F.normalize(
                batch[start : start + batch_size].to(nvlm.device),
                dim=1,
                eps=1e-8,
            )
            z = nvlm._proj_head_text_infonce(text_batch)
            chunks.append(F.normalize(z.float(), dim=1, eps=1e-8).detach().cpu())
    return torch.cat(chunks, dim=0)


__all__ = [
    "as_latent_batch",
    "project_brain_latents_to_shared",
    "project_text_latents_to_shared",
]
