"""Shared encode/decode adapters for MLP vs. atlas-free CNN model comparison.

Each adapter exposes the same minimal interface for its family so evaluation
scripts do not need to know whether they are talking to the packaged MLP
NeuroVLM baseline or an atlas-free CNN variant:

- autoencoder adapters: ``encode(x)`` / ``decode(z)``
- contrastive adapters: ``encode_brain_to_shared(x)`` / ``encode_text_to_shared(x)``
- text-to-brain adapters: ``generate(...)``

CNN contrastive (Stage 3) and CNN text-to-brain (Stage 4) checkpoints come in
six variants, not four: three ``mixed_to_{domain}`` baseline branches (the
Stage 1A mixed AE evaluated on that domain) and three domain-specialized
branches (``pubmed``, ``nilearn``, ``neurovault``, each paired with its own
Stage 1B domain-finetuned AE). They are not uploaded yet for any variant, so
constructing ``CNNContrastiveAdapter`` or ``CNNTextToBrainAdapter`` will raise
until the checkpoints land in the ``neurovlm/3d_cnn`` model repo.
"""

from __future__ import annotations

from typing import Any

import torch

import neurovlm.retrieval_resources as rr
from neurovlm.core import NeuroVLM

from atlas_free_cnn.evaluation.stage4_semantic import load_stage3_evaluator, load_stage4_projector

CNN_AE_DOMAINS = ("mixed", "pubmed", "nilearn", "neurovault")
CNN_STAGE_VARIANTS = (
    "mixed_to_pubmed",
    "mixed_to_nilearn",
    "mixed_to_neurovault",
    "pubmed",
    "nilearn",
    "neurovault",
)

_CNN_AE_LOADER_NAMES = {
    "mixed": "_load_mixed_ae",
    "pubmed": "_load_pubmed_finetuned_ae",
    "nilearn": "_load_nilearn_finetuned_ae",
    "neurovault": "_load_neurovault_finetuned_ae",
}


def _check_ae_domain(domain: str) -> None:
    if domain not in CNN_AE_DOMAINS:
        raise ValueError(f"Unknown CNN AE domain {domain!r}; expected one of {CNN_AE_DOMAINS}")


def _check_stage_variant(variant: str) -> None:
    if variant not in CNN_STAGE_VARIANTS:
        raise ValueError(f"Unknown CNN stage variant {variant!r}; expected one of {CNN_STAGE_VARIANTS}")


def _stage_variant_ae_loader_name(variant: str) -> str:
    """Return the retrieval_resources AE loader name backing a Stage 3/4 variant."""

    if variant.startswith("mixed_to_"):
        return _CNN_AE_LOADER_NAMES["mixed"]
    return _CNN_AE_LOADER_NAMES[variant]


def _l2_normalize(x: torch.Tensor) -> torch.Tensor:
    return x / x.norm(dim=-1, keepdim=True).clamp_min(1e-12)


def _as_2d(x: torch.Tensor) -> torch.Tensor:
    return x.unsqueeze(0) if x.ndim == 1 else x


class MLPAutoencoderAdapter:
    """Encode/decode adapter for the packaged MLP ``NeuroAutoEncoder``."""

    def __init__(self, *, device: str | torch.device = "cpu") -> None:
        self.device = torch.device(device)
        self.autoencoder = rr._load_autoencoder().to(self.device).eval()
        self.masker = rr._load_masker()

    @torch.no_grad()
    def encode(self, flat_batch: torch.Tensor) -> torch.Tensor:
        return self.autoencoder.encoder(_as_2d(flat_batch).to(self.device))

    @torch.no_grad()
    def decode(self, latent_batch: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.autoencoder.decoder(_as_2d(latent_batch).to(self.device)))


class CNNAutoencoderAdapter:
    """Encode/decode adapter for a Stage 1 atlas-free CNN autoencoder variant."""

    def __init__(self, domain: str, *, device: str | torch.device = "cpu") -> None:
        _check_ae_domain(domain)
        self.domain = domain
        self.device = torch.device(device)
        loader = getattr(rr, _CNN_AE_LOADER_NAMES[domain])
        self.autoencoder = loader().to(self.device).eval()

    @torch.no_grad()
    def encode(self, volume_batch: torch.Tensor) -> torch.Tensor:
        return self.autoencoder.encoder(volume_batch.to(self.device))

    @torch.no_grad()
    def decode(self, latent_batch: torch.Tensor) -> torch.Tensor:
        return self.autoencoder.decoder(_as_2d(latent_batch).to(self.device))


class MLPContrastiveAdapter:
    """Shared-space adapter using the MLP autoencoder encoder plus InfoNCE proj heads."""

    def __init__(self, *, device: str | torch.device = "cpu") -> None:
        self.device = torch.device(device)
        self.autoencoder = rr._load_autoencoder().to(self.device).eval()
        self.image_head = rr._proj_head_image_infonce().to(self.device).eval()
        self.text_head = rr._proj_head_text_infonce().to(self.device).eval()

    @torch.no_grad()
    def encode_brain_to_shared(self, flat_batch: torch.Tensor) -> torch.Tensor:
        latent = self.autoencoder.encoder(_as_2d(flat_batch).to(self.device))
        return _l2_normalize(self.image_head(latent))

    @torch.no_grad()
    def encode_text_to_shared(self, text_embedding_batch: torch.Tensor) -> torch.Tensor:
        text = _l2_normalize(_as_2d(text_embedding_batch).to(self.device))
        return _l2_normalize(self.text_head(text))


class CNNContrastiveAdapter:
    """Stage 3 CNN contrastive adapter: brain encoder + text projection in shared space."""

    def __init__(self, variant: str, *, device: str | torch.device = "cpu") -> None:
        _check_stage_variant(variant)
        self.variant = variant
        self.device = torch.device(device)
        checkpoint_path = rr._load_cnn_contrastive_checkpoint_path(variant)
        self.brain_encoder, self.text_projection = load_stage3_evaluator(checkpoint_path, self.device)

    @torch.no_grad()
    def encode_brain_to_shared(self, volume_batch: torch.Tensor) -> torch.Tensor:
        return self.brain_encoder(volume_batch.to(self.device))

    @torch.no_grad()
    def encode_text_to_shared(self, text_embedding_batch: torch.Tensor) -> torch.Tensor:
        return self.text_projection(_as_2d(text_embedding_batch).to(self.device).float())


class MLPTextToBrainAdapter:
    """Text-to-brain generation adapter using the packaged NeuroVLM MSE head."""

    def __init__(self, *, device: str = "cpu") -> None:
        self.vlm = NeuroVLM(device=device)

    def generate(self, text: Any) -> torch.Tensor:
        result = self.vlm.text(text).to_brain(head="mse")
        return result.generated_flatmaps


class CNNTextToBrainAdapter:
    """Stage 4 CNN text-to-brain adapter: SPECTER2 embedding -> generated volume."""

    def __init__(self, variant: str, *, device: str | torch.device = "cpu") -> None:
        _check_stage_variant(variant)
        self.variant = variant
        self.device = torch.device(device)
        ae_loader = getattr(rr, _stage_variant_ae_loader_name(variant))
        self.autoencoder = ae_loader().to(self.device).eval()
        checkpoint_path = rr._load_cnn_t2b_checkpoint_path(variant)
        latent_dim = int(getattr(self.autoencoder.decoder, "latent_dim", 384))
        self.projector, _ = load_stage4_projector(checkpoint_path, self.device, latent_dim=latent_dim)

    @torch.no_grad()
    def generate(self, text_embedding_batch: torch.Tensor) -> torch.Tensor:
        latent = self.projector(_as_2d(text_embedding_batch).to(self.device).float())
        return self.autoencoder.decoder(latent)
