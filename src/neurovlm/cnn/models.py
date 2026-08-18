"""Packaged runtime wrappers for pretrained atlas-free 3D CNN models."""

from __future__ import annotations

from functools import lru_cache
from typing import Any

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from neurovlm.cnn.architectures import (
    ALE3DCNNAutoEncoder,
    ALE3DCNNEncoder,
    ALEResNet3DEncoder,
)

CNN_AE_DOMAINS = ("mixed", "pubmed", "nilearn", "neurovault")
CNN_STAGE_VARIANTS = (
    "mixed_to_pubmed",
    "mixed_to_nilearn",
    "mixed_to_neurovault",
    "pubmed",
    "nilearn",
    "neurovault",
)
ATLAS_FREE_VOLUME_SHAPE = (36, 45, 38)
MLP_MASKER_VOXEL_COUNT = 28_542


def _freeze(module: nn.Module) -> nn.Module:
    module.eval()
    for parameter in module.parameters():
        parameter.requires_grad_(False)
    return module


def _as_batch(x: torch.Tensor, *, expected_ndim: int) -> torch.Tensor:
    return x.unsqueeze(0) if x.ndim == expected_ndim - 1 else x


def _autoencoder_arch(payload: dict[str, Any]) -> dict[str, Any]:
    recorded_arch = payload.get("architecture", {}) if isinstance(payload, dict) else {}
    config = payload.get("config", {}) if isinstance(payload, dict) else {}
    model_config = config.get("model", {}) if isinstance(config, dict) else {}
    target_shape = (
        recorded_arch.get("output_shape")
        or recorded_arch.get("target_shape")
        or payload.get("target_shape")
        or config.get("target_shape")
        or ATLAS_FREE_VOLUME_SHAPE
    )
    return {
        "latent_dim": int(
            recorded_arch.get(
                "latent_dim", model_config.get("latent_dim", payload.get("latent_dim", 384))
            )
        ),
        "base_channels": int(recorded_arch.get("base_channels", model_config.get("base_channels", 64))),
        "num_blocks": int(recorded_arch.get("num_blocks", model_config.get("num_blocks", 4))),
        "encoder_arch": str(recorded_arch.get("encoder_arch", model_config.get("encoder_arch", "plain"))),
        "dropout": float(recorded_arch.get("dropout", model_config.get("dropout", 0.1))),
        "norm": str(recorded_arch.get("norm", model_config.get("norm", "group"))),
        "pooling": str(recorded_arch.get("pooling", model_config.get("pooling", "max"))),
        "blocks_per_stage": int(recorded_arch.get("blocks_per_stage", model_config.get("blocks_per_stage", 2))),
        "multi_scale": bool(recorded_arch.get("multi_scale", model_config.get("multi_scale", False))),
        "global_context": str(recorded_arch.get("global_context", model_config.get("global_context", "none"))),
        "target_shape": tuple(int(value) for value in target_shape),
    }


def autoencoder_from_payload(payload: dict[str, Any]) -> ALE3DCNNAutoEncoder:
    """Construct a frozen CNN autoencoder from a trusted checkpoint payload."""

    architecture = _autoencoder_arch(payload)
    model = ALE3DCNNAutoEncoder(
        output_shape=architecture["target_shape"],
        latent_dim=architecture["latent_dim"],
        base_channels=architecture["base_channels"],
        num_blocks=architecture["num_blocks"],
        dropout=architecture["dropout"],
        norm=architecture["norm"],
        pooling=architecture["pooling"],
        encoder_arch=architecture["encoder_arch"],
        blocks_per_stage=architecture["blocks_per_stage"],
        multi_scale=architecture["multi_scale"],
        global_context=architecture["global_context"],
    )
    state = (
        payload.get("model_state_dict")
        or payload.get("model")
        or payload.get("autoencoder")
        or payload.get("state_dict")
    )
    if not isinstance(state, dict):
        raise KeyError(
            "CNN autoencoder checkpoint must contain 'model_state_dict', 'model', "
            "'autoencoder', or 'state_dict'"
        )
    model.load_state_dict(state, strict=True)
    return _freeze(model)


class CNNContrastiveModel(nn.Module):
    """Brain and text encoders aligned in a shared normalized space."""

    def __init__(self, brain_encoder: nn.Module, text_projection: nn.Module) -> None:
        super().__init__()
        self.brain_encoder = brain_encoder
        self.text_projection = text_projection

    def encode_brain(self, volume: torch.Tensor) -> torch.Tensor:
        volume = _as_batch(volume, expected_ndim=5)
        return F.normalize(self.brain_encoder(volume), dim=-1)

    def encode_text(self, text_embedding: torch.Tensor) -> torch.Tensor:
        text_embedding = _as_batch(text_embedding, expected_ndim=2)
        return F.normalize(self.text_projection(text_embedding), dim=-1)

    def forward(
        self,
        volume: torch.Tensor,
        text_embedding: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.encode_brain(volume), self.encode_text(text_embedding)


def _stage3_encoder(payload: dict[str, Any]) -> nn.Module:
    config = payload.get("config", {}) if isinstance(payload, dict) else {}
    architecture = payload.get("architecture", {}) if isinstance(payload, dict) else {}
    if not isinstance(config, dict):
        config = {}
    if not isinstance(architecture, dict):
        architecture = {}
    # Standardized checkpoints keep reload-critical fields in architecture;
    # historical Stage 3 payloads kept the same values directly in config.
    merged = {**config, **architecture}
    model_name = merged.get("model", "ale_3dcnn")
    if isinstance(model_name, dict):
        model_name = model_name.get("name", "ale_3dcnn")
    common = {
        "out_dim": int(merged.get("out_dim", merged.get("latent_dim", 384))),
        "base_channels": int(merged.get("base_channels", 64)),
        "dropout": float(merged.get("dropout", 0.1)),
        "norm": str(merged.get("norm", "group")),
    }
    if model_name in {None, "", "ale_3dcnn"}:
        return ALE3DCNNEncoder(
            in_channels=int(merged.get("in_channels", 1)),
            num_blocks=int(merged.get("num_blocks", 4)),
            pooling=str(merged.get("pooling", "max")),
            **common,
        )
    if model_name == "ale_3dcnn_resnet":
        return ALEResNet3DEncoder(
            in_channels=int(merged.get("in_channels", 1)),
            num_stages=int(merged.get("num_blocks", 4)),
            blocks_per_stage=int(merged.get("blocks_per_stage", 2)),
            multi_scale=bool(merged.get("multi_scale", False)),
            global_context=str(merged.get("global_context", "none")),
            **common,
        )
    raise ValueError(f"Unknown Stage 3 CNN architecture {model_name!r}")


def contrastive_from_payload(payload: dict[str, Any]) -> CNNContrastiveModel:
    """Construct a frozen Stage 3 contrastive model from a trusted payload."""

    from neurovlm.models.base import ProjHead

    brain_encoder = _stage3_encoder(payload)
    architecture = payload.get("architecture") or {}
    text_projection = ProjHead(
        latent_in_dim=int(architecture.get("text_in_dim", 768)),
        hidden_dim=int(architecture.get("text_hidden_dim", 512)),
        latent_out_dim=int(architecture.get("text_out_dim", 384)),
    )
    composite_state = payload.get("model_state_dict")
    if isinstance(composite_state, dict):
        model = CNNContrastiveModel(brain_encoder, text_projection)
        try:
            model.load_state_dict(composite_state, strict=True)
        except RuntimeError as error:
            raise ValueError(
                f"Contrastive checkpoint weights do not match recorded architecture: {error}"
            ) from error
        return _freeze(model)

    brain_state = payload.get("brain_encoder")
    if not isinstance(brain_state, dict):
        raise KeyError(
            "CNN contrastive checkpoint must contain standardized 'model_state_dict' "
            "or legacy 'brain_encoder' weights"
        )
    brain_encoder.load_state_dict(brain_state, strict=True)
    text_state = payload.get("text_proj") or payload.get("text_projection")
    if not isinstance(text_state, dict):
        raise KeyError("CNN contrastive checkpoint must contain 'text_proj' or 'text_projection'")
    text_projection.load_state_dict(text_state, strict=True)
    return _freeze(CNNContrastiveModel(brain_encoder, text_projection))


class GenerativeTextToAELatent(nn.Module):
    """Map a SPECTER2 embedding into the CNN autoencoder latent space."""

    def __init__(self, in_dim: int = 768, hidden_dim: int = 512, latent_dim: int = 384) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
        )

    def forward(self, text_embedding: torch.Tensor) -> torch.Tensor:
        return self.net(text_embedding)


class CNNTextToBrainModel(nn.Module):
    """Generate a dense brain volume from a precomputed SPECTER2 embedding."""

    def __init__(self, text_projection: nn.Module, autoencoder: ALE3DCNNAutoEncoder) -> None:
        super().__init__()
        self.text_projection = text_projection
        self.autoencoder = autoencoder

    def forward(self, text_embedding: torch.Tensor) -> torch.Tensor:
        text_embedding = _as_batch(text_embedding, expected_ndim=2)
        return self.autoencoder.decoder(self.text_projection(text_embedding))


def _stage4_state(payload: dict[str, Any]) -> dict[str, torch.Tensor]:
    keys = (
        "model_state_dict",
        "generative_text_to_ae_latent",
        "text_projector",
        "text_projection",
        "model",
        "state_dict",
    )
    for key in keys:
        state = payload.get(key)
        if isinstance(state, dict) and state and all(torch.is_tensor(value) for value in state.values()):
            for prefix in ("text_projection.", "text_projector."):
                selected = {
                    name.removeprefix(prefix): value
                    for name, value in state.items()
                    if name.startswith(prefix)
                }
                if selected:
                    return selected
            return state
    if payload and all(torch.is_tensor(value) for value in payload.values()):
        return payload
    raise KeyError("CNN text-to-brain checkpoint does not contain a recognized projector state dict")


def text_to_brain_from_payload(
    payload: dict[str, Any],
    autoencoder: ALE3DCNNAutoEncoder,
) -> CNNTextToBrainModel:
    """Construct a frozen Stage 4 generator from trusted checkpoint payloads."""

    config = payload.get("config", {}) if isinstance(payload, dict) else {}
    architecture = payload.get("architecture", {}) if isinstance(payload, dict) else {}
    projection_config = config.get("generative_text_to_ae_latent", {}) if isinstance(config, dict) else {}
    if not projection_config and isinstance(architecture, dict):
        projection_config = architecture.get("text_projection", architecture)
    state = _stage4_state(payload)
    if not any(key.startswith("net.") for key in state):
        raise ValueError("Stage 4 checkpoint is not the retained GenerativeTextToAELatent architecture")
    projector = GenerativeTextToAELatent(
        in_dim=int(projection_config.get("in_dim", 768)),
        hidden_dim=int(projection_config.get("hidden_dim", config.get("hidden_dim", 512))),
        latent_dim=autoencoder.decoder.latent_dim,
    )
    projector.load_state_dict(state, strict=True)
    return _freeze(CNNTextToBrainModel(projector, autoencoder))


@lru_cache(maxsize=1)
def _crop_mask() -> torch.Tensor:
    """Return the package mask cropped to the atlas-free CNN volume bounds."""

    from neurovlm.resources.loaders import _load_masker

    mask = np.asarray(_load_masker().mask_img_.get_fdata() > 0)
    occupied = np.where(mask)
    crop = tuple(slice(int(axis.min()), int(axis.max()) + 1) for axis in occupied)
    crop_mask = mask[crop]
    if tuple(crop_mask.shape) != ATLAS_FREE_VOLUME_SHAPE:
        raise ValueError(
            f"MLP mask crop shape {tuple(crop_mask.shape)} does not match {ATLAS_FREE_VOLUME_SHAPE}"
        )
    if int(crop_mask.sum()) != MLP_MASKER_VOXEL_COUNT:
        raise ValueError(
            f"MLP mask crop contains {int(crop_mask.sum())} voxels; expected {MLP_MASKER_VOXEL_COUNT}"
        )
    return torch.from_numpy(crop_mask)


def atlas_free_volume_to_mlp_flat(volume: torch.Tensor, *, binarize: bool = True) -> torch.Tensor:
    """Convert cropped CNN volumes to the packaged MLP masker's flat space."""

    volume = _as_batch(volume, expected_ndim=5).detach().cpu().float()
    if volume.shape[1] != 1 or tuple(volume.shape[-3:]) != ATLAS_FREE_VOLUME_SHAPE:
        raise ValueError(f"Expected (B, 1, {ATLAS_FREE_VOLUME_SHAPE}), got {tuple(volume.shape)}")
    volume = volume[:, 0]
    if binarize:
        volume = (volume > 0).float()
    return volume[:, _crop_mask()]


def mlp_flat_to_atlas_free_volume(flat: torch.Tensor) -> torch.Tensor:
    """Scatter packaged MLP flat vectors into cropped CNN volume space."""

    flat = _as_batch(flat, expected_ndim=2).detach().cpu().float()
    if flat.shape[1] != MLP_MASKER_VOXEL_COUNT:
        raise ValueError(f"Expected {MLP_MASKER_VOXEL_COUNT} features, got {flat.shape[1]}")
    volume = torch.zeros((flat.shape[0], *ATLAS_FREE_VOLUME_SHAPE), dtype=flat.dtype)
    volume[:, _crop_mask()] = flat
    return volume.unsqueeze(1)


__all__ = [
    "ATLAS_FREE_VOLUME_SHAPE",
    "CNN_AE_DOMAINS",
    "CNN_STAGE_VARIANTS",
    "CNNContrastiveModel",
    "CNNTextToBrainModel",
    "GenerativeTextToAELatent",
    "atlas_free_volume_to_mlp_flat",
    "mlp_flat_to_atlas_free_volume",
]
