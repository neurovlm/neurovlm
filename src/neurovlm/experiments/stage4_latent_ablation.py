"""Conditioning-only ablations for the frozen Stage 4 CNN latent target.

This module is deliberately separate from :mod:`neurovlm.training`.  It does
not alter the retained Stage 4 model, data path, or defaults.  Every projector
still emits 384 values and every decoder call receives a raw 384-dimensional
Stage 1 autoencoder latent.
"""

from __future__ import annotations

import copy
import csv
import json
import math
import os
import pickle
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Literal

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset

from neurovlm.atlas_free_text import AtlasFreeContrastiveCollator
from neurovlm.atlas_free_text import (
    AtlasFreeTextEmbeddingLookup,
    primary_positive_text_id,
)
from neurovlm.cnn import GenerativeTextToAELatent
from neurovlm.evaluation.spatial import reconstruction_metrics
from neurovlm.evaluation.text_to_brain_audit import tensor_sha256
from neurovlm.pipelines import (
    atomic_write_csv,
    atomic_write_json,
    json_safe,
    sha256_file,
    sha256_state_dict,
    sha256_value,
)


LatentTransformKind = Literal["raw", "standardized", "full_whitening", "pca_99_5"]
Stage4AblationVariant = Literal[
    "baseline_raw",
    "standardized_mse",
    "standardized_cosine",
    "standardized_cosine_norm",
    "full_whitening",
    "pca_99_5",
]

STAGE4_ABLATION_VARIANTS: dict[str, LatentTransformKind] = {
    "baseline_raw": "raw",
    "standardized_mse": "standardized",
    "standardized_cosine": "standardized",
    "standardized_cosine_norm": "standardized",
    "full_whitening": "full_whitening",
    "pca_99_5": "pca_99_5",
}


@dataclass(frozen=True)
class Stage4AblationTrainConfig:
    """Baseline-matched controls for one branch/variant experiment run."""

    variant: Stage4AblationVariant
    seed: int = 42
    projector_seed: int = 42
    epochs: int = 100
    batch_size: int = 64
    eval_batch_size: int | None = None
    num_workers: int = 0
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    gradient_clip: float | None = 1.0
    early_stopping_patience: int | None = 10
    early_stopping_min_delta: float = 0.0
    amp: bool = True
    amp_dtype: Literal["auto", "bfloat16", "float16", "float32"] = "auto"
    cosine_weight: float = 0.10
    norm_weight: float = 0.10
    scheduler: Literal["none"] = "none"
    max_train_batches: int | None = None
    max_eval_batches: int | None = None
    reconstruction_examples: int = 6

    def __post_init__(self) -> None:
        if self.variant not in STAGE4_ABLATION_VARIANTS:
            raise ValueError(f"Unknown ablation variant {self.variant!r}")
        if self.seed < 0 or self.projector_seed < 0:
            raise ValueError("seeds must be non-negative")
        if self.epochs < 1 or self.batch_size < 1:
            raise ValueError("epochs and batch_size must be positive")
        if self.eval_batch_size is not None and self.eval_batch_size < 1:
            raise ValueError("eval_batch_size must be positive")
        if self.num_workers < 0:
            raise ValueError("num_workers must be non-negative")
        if self.learning_rate <= 0 or self.weight_decay < 0:
            raise ValueError("learning_rate must be positive and weight_decay non-negative")
        if self.gradient_clip is not None and self.gradient_clip <= 0:
            raise ValueError("gradient_clip must be positive or None")
        if self.early_stopping_patience is not None and self.early_stopping_patience < 1:
            raise ValueError("early_stopping_patience must be positive or None")
        if self.early_stopping_min_delta < 0:
            raise ValueError("early_stopping_min_delta must be non-negative")
        if self.cosine_weight < 0 or self.norm_weight < 0:
            raise ValueError("auxiliary loss weights must be non-negative")
        if self.max_train_batches is not None and self.max_train_batches < 1:
            raise ValueError("max_train_batches must be positive or None")
        if self.max_eval_batches is not None and self.max_eval_batches < 1:
            raise ValueError("max_eval_batches must be positive or None")
        if self.reconstruction_examples < 0:
            raise ValueError("reconstruction_examples must be non-negative")

    def effective_dict(self) -> dict[str, Any]:
        values = asdict(self)
        values.update(
            {
                "architecture": {
                    "name": "GenerativeTextToAELatent",
                    "layers": [768, 512, "ReLU", 384],
                },
                "optimizer": "AdamW",
                "decoder_input": "raw_384d_stage1_ae_latent",
                "autoencoder_frozen": True,
                "test_used_for_selection": False,
            }
        )
        return values


@dataclass(frozen=True)
class Stage4AblationEvaluation:
    summary: Mapping[str, float]
    per_dimension: tuple[Mapping[str, float], ...]
    examples: tuple[Mapping[str, Any], ...]
    n: int


class LatentTransform(nn.Module):
    """A training-only fitted transform with an exact raw-latent inverse path.

    The representation always has ``latent_dim`` columns.  For ``pca_99_5``,
    only the leading ``active_dim`` whitened principal components are active;
    trailing projector outputs are masked before both the loss and inverse.
    This keeps the production 384-output projector architecture unchanged.
    """

    FORMAT_VERSION = 1

    def __init__(
        self,
        *,
        kind: LatentTransformKind,
        mean: Tensor,
        scale: Tensor,
        components: Tensor,
        active_dim: int,
        epsilon: float,
        retained_variance: float,
        eigenvalues: Tensor,
    ) -> None:
        super().__init__()
        mean = torch.as_tensor(mean, dtype=torch.float32).flatten()
        scale = torch.as_tensor(scale, dtype=torch.float32).flatten()
        components = torch.as_tensor(components, dtype=torch.float32)
        eigenvalues = torch.as_tensor(eigenvalues, dtype=torch.float32).flatten()
        latent_dim = int(mean.numel())
        if kind not in {"raw", "standardized", "full_whitening", "pca_99_5"}:
            raise ValueError(f"Unknown latent transform kind: {kind!r}")
        if latent_dim < 1 or scale.shape != mean.shape:
            raise ValueError("mean and scale must be matching non-empty vectors")
        if components.shape != (latent_dim, latent_dim):
            raise ValueError("components must be a square latent_dim x latent_dim matrix")
        if eigenvalues.shape != mean.shape:
            raise ValueError("eigenvalues must have latent_dim entries")
        if not 1 <= int(active_dim) <= latent_dim:
            raise ValueError("active_dim must be between 1 and latent_dim")
        if not math.isfinite(epsilon) or epsilon <= 0:
            raise ValueError("epsilon must be finite and positive")
        if not 0 < retained_variance <= 1:
            raise ValueError("retained_variance must be in (0, 1]")
        if not all(torch.isfinite(value).all() for value in (mean, scale, components, eigenvalues)):
            raise ValueError("latent transform statistics must be finite")
        if bool((scale <= 0).any()):
            raise ValueError("latent transform scale must be positive")
        self.kind = kind
        self.active_dim = int(active_dim)
        self.epsilon = float(epsilon)
        self.retained_variance = float(retained_variance)
        self.register_buffer("mean", mean)
        self.register_buffer("scale", scale)
        self.register_buffer("components", components)
        self.register_buffer("eigenvalues", eigenvalues)
        mask = torch.zeros(latent_dim, dtype=torch.float32)
        mask[: self.active_dim] = 1.0
        self.register_buffer("active_mask", mask)

    @property
    def latent_dim(self) -> int:
        return int(self.mean.numel())

    @property
    def is_lossless(self) -> bool:
        return self.active_dim == self.latent_dim

    @classmethod
    def fit(
        cls,
        training_latents: Tensor,
        kind: LatentTransformKind,
        *,
        epsilon: float = 1e-4,
        retained_variance: float = 0.995,
    ) -> "LatentTransform":
        """Fit statistics using one explicitly supplied training tensor only."""

        values = torch.as_tensor(training_latents).detach().to(dtype=torch.float64, device="cpu")
        if values.ndim != 2 or values.shape[0] < 2 or values.shape[1] < 1:
            raise ValueError("training_latents must have shape N x D with N >= 2")
        if not bool(torch.isfinite(values).all()):
            raise ValueError("training_latents contain NaNs or infinities")
        if not math.isfinite(epsilon) or epsilon <= 0:
            raise ValueError("epsilon must be finite and positive")
        if not 0 < retained_variance <= 1:
            raise ValueError("retained_variance must be in (0, 1]")

        n, latent_dim = values.shape
        identity = torch.eye(latent_dim, dtype=torch.float64)
        zeros = torch.zeros(latent_dim, dtype=torch.float64)
        ones = torch.ones(latent_dim, dtype=torch.float64)
        if kind == "raw":
            return cls(
                kind=kind,
                mean=zeros,
                scale=ones,
                components=identity,
                active_dim=latent_dim,
                epsilon=epsilon,
                retained_variance=1.0,
                eigenvalues=ones,
            )

        mean = values.mean(dim=0)
        centered = values - mean
        per_dimension_variance = centered.square().mean(dim=0)
        if kind == "standardized":
            scale = per_dimension_variance.sqrt().clamp_min(epsilon)
            return cls(
                kind=kind,
                mean=mean,
                scale=scale,
                components=identity,
                active_dim=latent_dim,
                epsilon=epsilon,
                retained_variance=1.0,
                eigenvalues=per_dimension_variance,
            )

        if kind not in {"full_whitening", "pca_99_5"}:
            raise ValueError(f"Unknown latent transform kind: {kind!r}")
        covariance = centered.T @ centered / max(n - 1, 1)
        eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
        order = eigenvalues.argsort(descending=True)
        eigenvalues = eigenvalues[order].clamp_min(0)
        components = eigenvectors[:, order]
        total_variance = eigenvalues.sum()
        if not bool(total_variance > 0):
            raise ValueError("Cannot fit PCA whitening to a zero-variance latent tensor")
        if kind == "pca_99_5":
            cumulative = eigenvalues.cumsum(0) / total_variance
            active_dim = int(
                torch.searchsorted(
                    cumulative,
                    torch.as_tensor(retained_variance, dtype=cumulative.dtype),
                ).item()
                + 1
            )
        else:
            active_dim = latent_dim
        actual_retained = float(eigenvalues[:active_dim].sum() / total_variance)
        scale = eigenvalues.sqrt().clamp_min(epsilon)
        return cls(
            kind=kind,
            mean=mean,
            scale=scale,
            components=components,
            active_dim=active_dim,
            epsilon=epsilon,
            retained_variance=actual_retained,
            eigenvalues=eigenvalues,
        )

    def transform(self, raw_latent: Tensor) -> Tensor:
        """Map raw Stage 1 latents to the configured 384-value convention."""

        raw_latent = torch.as_tensor(raw_latent)
        if raw_latent.shape[-1] != self.latent_dim:
            raise ValueError(
                f"Expected raw latent dimension {self.latent_dim}, got {raw_latent.shape[-1]}"
            )
        if self.kind == "raw":
            represented = raw_latent
        elif self.kind == "standardized":
            represented = (raw_latent - self.mean) / self.scale
        else:
            represented = ((raw_latent - self.mean) @ self.components) / self.scale
        return represented * self.active_mask

    def inverse(self, representation: Tensor) -> Tensor:
        """Return raw 384-dimensional Stage 1 latents for the frozen decoder."""

        representation = torch.as_tensor(representation)
        if representation.shape[-1] != self.latent_dim:
            raise ValueError(
                f"Expected representation dimension {self.latent_dim}, "
                f"got {representation.shape[-1]}"
            )
        represented = representation * self.active_mask
        if self.kind == "raw":
            return represented
        if self.kind == "standardized":
            return represented * self.scale + self.mean
        return (represented * self.scale) @ self.components.T + self.mean

    def forward(self, raw_latent: Tensor) -> Tensor:
        return self.transform(raw_latent)

    def metadata(self) -> dict[str, Any]:
        total = float(self.eigenvalues.sum())
        retained = float(self.eigenvalues[: self.active_dim].sum())
        return {
            "format_version": self.FORMAT_VERSION,
            "kind": self.kind,
            "latent_dim": self.latent_dim,
            "projector_output_dim": self.latent_dim,
            "active_dim": self.active_dim,
            "epsilon": self.epsilon,
            "retained_variance": retained / total if total > 0 else self.retained_variance,
            "is_lossless": self.is_lossless,
            "projector_output_convention": (
                "raw_stage1_ae_latent"
                if self.kind == "raw"
                else (
                    "leading_active_whitened_pca_components_with_trailing_outputs_masked"
                    if self.kind == "pca_99_5"
                    else f"{self.kind}_latent_coordinates"
                )
            ),
            "decoder_input_convention": "inverse_transformed_raw_384d_stage1_ae_latent",
            "state_sha256": sha256_state_dict(self),
        }

    def to_payload(self) -> dict[str, Any]:
        return {"metadata": self.metadata(), "state_dict": self.state_dict()}

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "LatentTransform":
        metadata = payload.get("metadata")
        state = payload.get("state_dict")
        if not isinstance(metadata, Mapping) or not isinstance(state, Mapping):
            raise TypeError("Latent transform payload requires metadata and state_dict mappings")
        transform = cls(
            kind=str(metadata["kind"]),  # type: ignore[arg-type]
            mean=state["mean"],
            scale=state["scale"],
            components=state["components"],
            active_dim=int(metadata["active_dim"]),
            epsilon=float(metadata["epsilon"]),
            retained_variance=float(metadata["retained_variance"]),
            eigenvalues=state["eigenvalues"],
        )
        transform.load_state_dict(state, strict=True)
        expected = metadata.get("state_sha256")
        if expected and sha256_state_dict(transform) != expected:
            raise ValueError("Latent transform state SHA256 mismatch")
        return transform


@dataclass(frozen=True)
class Stage4AblationLoss:
    """Loss output with the raw decoder input made inspectable."""

    total: Tensor
    raw_prediction_latent: Tensor
    target_representation: Tensor
    prediction_volume: Tensor
    parts: Mapping[str, Tensor]


def compute_stage4_ablation_loss(
    variant: Stage4AblationVariant,
    projector_output: Tensor,
    target_raw_latent: Tensor,
    target_volume: Tensor,
    *,
    transform: LatentTransform,
    decoder: nn.Module,
    cosine_weight: float = 0.10,
    norm_weight: float = 0.10,
) -> Stage4AblationLoss:
    """Compute one conditioning-only loss and decode raw Stage 1 latents."""

    expected_kind = STAGE4_ABLATION_VARIANTS.get(variant)
    if expected_kind is None:
        raise ValueError(f"Unknown Stage 4 ablation variant: {variant!r}")
    if transform.kind != expected_kind:
        raise ValueError(
            f"Variant {variant!r} requires transform {expected_kind!r}, "
            f"got {transform.kind!r}"
        )
    if projector_output.shape != target_raw_latent.shape:
        raise ValueError("projector output and raw target latent must have identical N x 384 shapes")
    target_representation = transform.transform(target_raw_latent.detach())
    prediction_representation = projector_output * transform.active_mask
    raw_prediction = transform.inverse(prediction_representation)
    prediction_volume = decoder(raw_prediction)
    active = slice(0, transform.active_dim)
    transformed_mse = F.mse_loss(
        prediction_representation[:, active],
        target_representation[:, active],
    )
    decoded_mse = F.mse_loss(prediction_volume, target_volume)
    latent_cosine_loss = 1 - F.cosine_similarity(
        prediction_representation[:, active],
        target_representation[:, active],
        dim=1,
        eps=1e-8,
    ).mean()
    raw_norm_mse = F.mse_loss(
        raw_prediction.norm(dim=1),
        target_raw_latent.detach().norm(dim=1),
    )
    total = transformed_mse + decoded_mse
    if variant in {"standardized_cosine", "standardized_cosine_norm", "full_whitening"}:
        total = total + cosine_weight * latent_cosine_loss
    if variant == "standardized_cosine_norm":
        total = total + norm_weight * raw_norm_mse
    parts = {
        "loss": total,
        "transformed_latent_mse": transformed_mse,
        "decoded_volume_mse": decoded_mse,
        "latent_cosine_loss": latent_cosine_loss,
        "raw_latent_norm_mse": raw_norm_mse,
        "raw_latent_mse": F.mse_loss(raw_prediction, target_raw_latent.detach()),
    }
    return Stage4AblationLoss(
        total=total,
        raw_prediction_latent=raw_prediction,
        target_representation=target_representation,
        prediction_volume=prediction_volume,
        parts=parts,
    )


def _safe_correlation(left: Tensor, right: Tensor) -> Tensor:
    left = left - left.mean()
    right = right - right.mean()
    return (left * right).sum() / (left.norm() * right.norm()).clamp_min(1e-12)


def _nearest_distances(
    prediction: Tensor,
    reference: Tensor,
    *,
    prediction_chunk_size: int = 512,
    reference_chunk_size: int = 4096,
    device: str | torch.device | None = None,
) -> Tensor:
    if prediction_chunk_size < 1 or reference_chunk_size < 1:
        raise ValueError("distance chunk sizes must be positive")
    resolved = torch.device(
        device
        if device is not None
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    reference = reference.to(resolved)
    nearest: list[Tensor] = []
    for start in range(0, len(prediction), prediction_chunk_size):
        query = prediction[start : start + prediction_chunk_size].to(resolved)
        minimum = torch.full(
            (len(query),),
            float("inf"),
            dtype=query.dtype,
            device=resolved,
        )
        for ref_start in range(0, len(reference), reference_chunk_size):
            distances = torch.cdist(
                query,
                reference[ref_start : ref_start + reference_chunk_size],
            )
            minimum = torch.minimum(minimum, distances.min(dim=1).values)
        nearest.append(minimum.cpu())
    return torch.cat(nearest)


def latent_ablation_metrics(
    target_raw: Tensor,
    prediction_raw: Tensor,
    *,
    transform: LatentTransform,
    nearest_reference: Tensor | None = None,
    prediction_chunk_size: int = 512,
    reference_chunk_size: int = 4096,
    distance_device: str | torch.device | None = None,
) -> tuple[dict[str, float], list[dict[str, float]]]:
    """Compute checkpoint metrics and per-dimension raw-latent diagnostics."""

    target = torch.as_tensor(target_raw).detach().float().cpu()
    prediction = torch.as_tensor(prediction_raw).detach().float().cpu()
    if target.ndim != 2 or target.shape != prediction.shape:
        raise ValueError("target_raw and prediction_raw must have identical N x D shapes")
    if len(target) < 2:
        raise ValueError("latent metrics require at least two examples")
    cpu_transform = copy.deepcopy(transform).to("cpu")
    target_transformed = cpu_transform.transform(target)
    prediction_transformed = cpu_transform.transform(prediction)
    active = slice(0, cpu_transform.active_dim)
    target_mean = target.mean(dim=0)
    prediction_mean = prediction.mean(dim=0)
    target_variance = target.var(dim=0, unbiased=False)
    prediction_variance = prediction.var(dim=0, unbiased=False)
    residual = target - prediction
    ss_res = residual.square().sum(dim=0)
    ss_tot = (target - target_mean).square().sum(dim=0)
    r_squared = torch.where(ss_tot > 0, 1 - ss_res / ss_tot, torch.zeros_like(ss_tot))
    order = target_variance.argsort(descending=True)
    quartile = order[: max(1, target.shape[1] // 4)]
    total_target_variance = target_variance.sum().clamp_min(1e-12)
    reference = target if nearest_reference is None else torch.as_tensor(
        nearest_reference
    ).detach().float().cpu()
    if reference.ndim != 2 or reference.shape[1] != target.shape[1] or not len(reference):
        raise ValueError("nearest_reference must be a non-empty M x D tensor")
    nearest = _nearest_distances(
        prediction,
        reference,
        prediction_chunk_size=prediction_chunk_size,
        reference_chunk_size=reference_chunk_size,
        device=distance_device,
    )
    target_norm = target.norm(dim=1)
    prediction_norm = prediction.norm(dim=1)
    metrics = {
        "raw_latent_mse": float(F.mse_loss(prediction, target)),
        "transformed_latent_mse": float(
            F.mse_loss(
                prediction_transformed[:, active],
                target_transformed[:, active],
            )
        ),
        "latent_cosine_similarity": float(
            F.cosine_similarity(prediction, target, dim=1, eps=1e-8).mean()
        ),
        "predicted_target_latent_variance_ratio": float(
            prediction_variance.sum() / total_target_variance
        ),
        "predicted_target_latent_norm_ratio": float(
            prediction_norm.mean() / target_norm.mean().clamp_min(1e-12)
        ),
        "global_explained_variance": float(
            1
            - residual.var(dim=0, unbiased=False).sum()
            / total_target_variance
        ),
        "mean_per_dimension_r_squared": float(r_squared.mean()),
        "highest_target_variance_quartile_mean_r_squared": float(r_squared[quartile].mean()),
        "target_prediction_per_dimension_variance_correlation": float(
            _safe_correlation(target_variance, prediction_variance)
        ),
        "distance_to_nearest_real_ae_latent": float(nearest.mean()),
        "distance_to_mean_target_latent": float(
            (prediction - target_mean).norm(dim=1).mean()
        ),
        "target_latent_norm_mean": float(target_norm.mean()),
        "predicted_latent_norm_mean": float(prediction_norm.mean()),
    }
    rows = [
        {
            "dimension": int(index),
            "target_mean": float(target_mean[index]),
            "prediction_mean": float(prediction_mean[index]),
            "target_variance": float(target_variance[index]),
            "prediction_variance": float(prediction_variance[index]),
            "variance_ratio": float(
                prediction_variance[index] / target_variance[index].clamp_min(1e-12)
            ),
            "r_squared": float(r_squared[index]),
        }
        for index in range(target.shape[1])
    ]
    return metrics, rows


def _seed_everything(seed: int) -> None:
    import random

    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def resolve_amp_dtype(
    device: str | torch.device,
    requested: Literal["auto", "bfloat16", "float16", "float32"] = "auto",
) -> torch.dtype:
    """Choose BF16 on supported CUDA devices, otherwise FP16, or FP32 on CPU."""

    resolved = torch.device(device)
    if requested == "float32" or resolved.type != "cuda":
        return torch.float32
    if requested == "bfloat16":
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError("BF16 was requested but this CUDA device does not support it")
        return torch.bfloat16
    if requested == "float16":
        return torch.float16
    return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16


def _loader(
    dataset: Dataset,
    lookup: AtlasFreeTextEmbeddingLookup,
    *,
    batch_size: int,
    shuffle: bool,
    seed: int,
    num_workers: int,
    target_shape: tuple[int, int, int],
) -> DataLoader:
    rows = getattr(dataset, "rows", None)
    if rows is not None:
        lookup.validate_dataset(rows)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=AtlasFreeContrastiveCollator(lookup, target_shape),
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
        generator=torch.Generator().manual_seed(seed),
    )


@torch.no_grad()
def encode_stage1_latents(
    autoencoder: nn.Module,
    dataset: Dataset,
    lookup: AtlasFreeTextEmbeddingLookup,
    *,
    device: str | torch.device,
    batch_size: int = 64,
    num_workers: int = 0,
    target_shape: tuple[int, int, int] = (36, 45, 38),
) -> Tensor:
    """Encode an ordered split without changing the frozen AE mode."""

    resolved = torch.device(device)
    autoencoder.to(resolved).eval()
    if any(parameter.requires_grad for parameter in autoencoder.parameters()):
        raise RuntimeError("Stage 1 autoencoder must be frozen before latent extraction")
    encoded: list[Tensor] = []
    for batch in _loader(
        dataset,
        lookup,
        batch_size=batch_size,
        shuffle=False,
        seed=0,
        num_workers=num_workers,
        target_shape=target_shape,
    ):
        encoded.append(autoencoder.encoder(batch["volume"].to(resolved)).float().cpu())
    if not encoded:
        raise RuntimeError("Cannot encode an empty split")
    output = torch.cat(encoded)
    if len(output) != len(dataset):
        raise RuntimeError("Encoded latent count does not match the ordered dataset")
    return output


@torch.no_grad()
def evaluate_stage4_ablation(
    projector: nn.Module,
    autoencoder: nn.Module,
    transform: LatentTransform,
    dataset: Dataset,
    lookup: AtlasFreeTextEmbeddingLookup,
    *,
    training_reference_latents: Tensor,
    device: str | torch.device,
    batch_size: int = 64,
    num_workers: int = 0,
    target_shape: tuple[int, int, int] = (36, 45, 38),
    max_batches: int | None = None,
    reconstruction_examples: int = 6,
    semantic_evaluator: Callable[..., Mapping[str, float]] | None = None,
) -> Stage4AblationEvaluation:
    """Evaluate a split; callers control whether the split is validation or test."""

    resolved = torch.device(device)
    projector.to(resolved).eval()
    autoencoder.to(resolved).eval()
    transform.to(resolved).eval()
    spatial_totals: dict[str, float] = {}
    target_latents: list[Tensor] = []
    prediction_latents: list[Tensor] = []
    examples: list[Mapping[str, Any]] = []
    semantic_predictions: list[Tensor] = []
    semantic_targets: list[Tensor] = []
    semantic_metadata: list[dict[str, Any]] = []
    n = 0
    loader = _loader(
        dataset,
        lookup,
        batch_size=batch_size,
        shuffle=False,
        seed=0,
        num_workers=num_workers,
        target_shape=target_shape,
    )
    for batch_index, batch in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        target_volume = batch["volume"].to(resolved, non_blocking=True)
        text = batch["text_embedding"].to(resolved, non_blocking=True)
        target_raw = autoencoder.encoder(target_volume)
        representation = projector(text)
        prediction_raw = transform.inverse(representation)
        prediction_volume = autoencoder.decoder(prediction_raw)
        spatial = reconstruction_metrics(prediction_volume, target_volume)
        batch_n = len(target_volume)
        for name, value in spatial.items():
            spatial_totals[name] = spatial_totals.get(name, 0.0) + float(value) * batch_n
        target_latents.append(target_raw.float().cpu())
        prediction_latents.append(prediction_raw.float().cpu())
        n += batch_n
        remaining = max(0, reconstruction_examples - len(examples))
        for index in range(min(remaining, batch_n)):
            examples.append(
                {
                    "map_id": str(batch["map_id"][index]),
                    "text_id": str(batch["text_id"][index]),
                    "source": str(batch["source"][index]),
                    "prediction": prediction_volume[index].float().cpu(),
                    "target": target_volume[index].float().cpu(),
                }
            )
        if semantic_evaluator is not None:
            semantic_predictions.append(prediction_volume.float().cpu())
            semantic_targets.append(target_volume.float().cpu())
            semantic_metadata.extend(
                {
                    "map_id": str(batch["map_id"][index]),
                    "text_id": str(batch["text_id"][index]),
                    "text": str(batch["primary_text"][index]),
                    "source": str(batch["source"][index]),
                }
                for index in range(batch_n)
            )
    if not n:
        raise RuntimeError("Evaluation dataset produced no batches")
    latent_metrics, per_dimension = latent_ablation_metrics(
        torch.cat(target_latents),
        torch.cat(prediction_latents),
        transform=transform,
        nearest_reference=training_reference_latents,
    )
    summary = {
        **latent_metrics,
        **{name: value / n for name, value in spatial_totals.items()},
    }
    summary["decoded_mse"] = summary["mse"]
    summary["prediction_negative_voxel_fraction_before_clamping"] = summary[
        "raw_pred_fraction_below_zero"
    ]
    if semantic_evaluator is not None:
        semantic = semantic_evaluator(
            predictions=torch.cat(semantic_predictions),
            targets=torch.cat(semantic_targets),
            metadata=tuple(semantic_metadata),
        )
        for name, value in semantic.items():
            summary[f"semantic_{name}"] = float(value)
    return Stage4AblationEvaluation(
        summary=summary,
        per_dimension=tuple(per_dimension),
        examples=tuple(examples),
        n=n,
    )


def _read_csv(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def _train_epoch(
    projector: nn.Module,
    autoencoder: nn.Module,
    transform: LatentTransform,
    dataset: Dataset,
    lookup: AtlasFreeTextEmbeddingLookup,
    training_latents: Tensor,
    optimizer: torch.optim.Optimizer,
    config: Stage4AblationTrainConfig,
    *,
    device: torch.device,
    epoch: int,
    target_shape: tuple[int, int, int],
) -> tuple[dict[str, float], int]:
    projector.train()
    autoencoder.eval()
    transform.eval()
    for parameter in autoencoder.parameters():
        parameter.requires_grad_(False)
    amp_dtype = resolve_amp_dtype(device, config.amp_dtype)
    autocast_enabled = config.amp and device.type == "cuda" and amp_dtype != torch.float32
    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=autocast_enabled and amp_dtype == torch.float16,
    )
    totals: dict[str, float] = {}
    n = 0
    loader = _loader(
        dataset,
        lookup,
        batch_size=config.batch_size,
        shuffle=True,
        seed=config.seed + epoch,
        num_workers=config.num_workers,
        target_shape=target_shape,
    )
    parameters = tuple(projector.parameters())
    for batch_index, batch in enumerate(loader):
        if config.max_train_batches is not None and batch_index >= config.max_train_batches:
            break
        target_volume = batch["volume"].to(device, non_blocking=True)
        text = batch["text_embedding"].to(device, non_blocking=True)
        indices = torch.as_tensor(batch["dataset_index"], dtype=torch.long)
        if bool((indices < 0).any()):
            raise RuntimeError("Training batches require stable dataset_index values")
        target_raw = training_latents.index_select(0, indices).to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type=device.type,
            dtype=amp_dtype,
            enabled=autocast_enabled,
        ):
            output = compute_stage4_ablation_loss(
                config.variant,
                projector(text),
                target_raw,
                target_volume,
                transform=transform,
                decoder=autoencoder.decoder,
                cosine_weight=config.cosine_weight,
                norm_weight=config.norm_weight,
            )
        if scaler.is_enabled():
            scaler.scale(output.total).backward()
            scaler.unscale_(optimizer)
            if config.gradient_clip is not None:
                torch.nn.utils.clip_grad_norm_(parameters, config.gradient_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            output.total.backward()
            if config.gradient_clip is not None:
                torch.nn.utils.clip_grad_norm_(parameters, config.gradient_clip)
            optimizer.step()
        batch_n = len(target_volume)
        for name, value in output.parts.items():
            totals[name] = totals.get(name, 0.0) + float(value.detach()) * batch_n
        n += batch_n
    if not n:
        raise RuntimeError("Training dataset produced no batches")
    return {name: value / n for name, value in totals.items()}, n


def train_stage4_ablation(
    config: Stage4AblationTrainConfig,
    *,
    run_dir: str | Path,
    autoencoder: nn.Module,
    transform: LatentTransform,
    training_latents: Tensor,
    train_dataset: Dataset,
    validation_dataset: Dataset,
    lookup: AtlasFreeTextEmbeddingLookup,
    binding: Mapping[str, Any],
    device: str | torch.device,
    target_shape: tuple[int, int, int] = (36, 45, 38),
    semantic_evaluator: Callable[..., Mapping[str, float]] | None = None,
) -> dict[str, Any]:
    """Train/resume one variant; validation is the only selection split."""

    run_path = Path(run_dir)
    run_path.mkdir(parents=True, exist_ok=True)
    resolved = torch.device(device)
    _seed_everything(config.projector_seed)
    projector = GenerativeTextToAELatent(768, 512, 384).to(resolved)
    autoencoder.to(resolved).eval()
    transform.to(resolved).eval()
    for parameter in autoencoder.parameters():
        parameter.requires_grad_(False)
    optimizer = torch.optim.AdamW(
        projector.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scheduler = None
    effective = config.effective_dict()
    architecture = effective["architecture"]
    manager = AblationCheckpointManager(
        run_path,
        binding=binding,
        architecture=architecture,
        config=effective,
    )
    resumed = manager.resume(projector, optimizer, scheduler=scheduler, map_location=resolved)
    start_epoch = 1 if resumed is None else int(resumed["epoch"]) + 1
    early = dict((resumed or {}).get("early_stopping") or {})
    early_best = float(early.get("best", -float("inf")))
    stale_epochs = int(early.get("stale_epochs", 0))
    history = [
        row
        for row in _read_csv(run_path / "training_history.csv")
        if int(row.get("epoch", 0)) < start_epoch
    ]
    validation_rows = [
        row
        for row in _read_csv(run_path / "validation_metrics.csv")
        if int(row.get("epoch", 0)) < start_epoch
    ]
    per_dimension_rows: list[dict[str, Any]] = []
    last_examples: tuple[Mapping[str, Any], ...] = ()
    epochs_completed = start_epoch - 1
    for epoch in range(start_epoch, config.epochs + 1):
        train_metrics, train_n = _train_epoch(
            projector,
            autoencoder,
            transform,
            train_dataset,
            lookup,
            training_latents,
            optimizer,
            config,
            device=resolved,
            epoch=epoch,
            target_shape=target_shape,
        )
        validation = evaluate_stage4_ablation(
            projector,
            autoencoder,
            transform,
            validation_dataset,
            lookup,
            training_reference_latents=training_latents,
            device=resolved,
            batch_size=config.eval_batch_size or config.batch_size,
            num_workers=config.num_workers,
            target_shape=target_shape,
            max_batches=config.max_eval_batches,
            reconstruction_examples=config.reconstruction_examples,
            semantic_evaluator=semantic_evaluator,
        )
        row = {"epoch": epoch, "n": train_n, **{f"train_{k}": v for k, v in train_metrics.items()}}
        history.append(row)
        validation_row = {
            "epoch": epoch,
            "n": validation.n,
            **{f"val_{key}": value for key, value in validation.summary.items()},
        }
        validation_rows.append(validation_row)
        atomic_write_csv(run_path / "training_history.csv", history)
        atomic_write_csv(run_path / "validation_metrics.csv", validation_rows)
        per_dimension_rows = [
            {"epoch": epoch, **dict(item)} for item in validation.per_dimension
        ]
        atomic_write_csv(
            run_path / "per_dimension_latent_diagnostics.csv",
            per_dimension_rows,
        )
        selected = float(validation.summary["top5_dice"])
        if selected > early_best + config.early_stopping_min_delta:
            early_best = selected
            stale_epochs = 0
        else:
            stale_epochs += 1
        early_state = {"best": early_best, "stale_epochs": stale_epochs}
        metrics = {**row, **validation_row}
        objectives = {
            "val_top5_dice": validation.summary.get("top5_dice"),
            "val_spatial_corr": validation.summary.get("spatial_corr"),
            "val_global_explained_variance": validation.summary.get(
                "global_explained_variance"
            ),
        }
        semantic_auc = validation.summary.get("semantic_normalized_auc")
        if semantic_auc is None:
            semantic_auc = validation.summary.get(
                "semantic_normalized_k_recall_curve_auc"
            )
        if semantic_auc is not None:
            objectives["val_semantic_normalized_auc"] = semantic_auc
        for metric, value in objectives.items():
            if value is not None:
                manager.update_best(
                    metric,
                    float(value),
                    projector,
                    optimizer,
                    epoch=epoch,
                    metrics=metrics,
                    scheduler=scheduler,
                    early_stopping=early_state,
                )
        manager.save_last(
            projector,
            optimizer,
            epoch=epoch,
            metrics=metrics,
            scheduler=scheduler,
            early_stopping=early_state,
        )
        epochs_completed = epoch
        last_examples = validation.examples
        if (
            config.early_stopping_patience is not None
            and stale_epochs >= config.early_stopping_patience
        ):
            break
    return {
        "projector": projector,
        "checkpoint_manager": manager,
        "epochs_completed": epochs_completed,
        "last_validation_examples": last_examples,
        "last_per_dimension": per_dimension_rows,
    }


def split_fingerprint(dataset: Any) -> dict[str, Any]:
    """Fingerprint the immutable ordered row identity used by one split."""

    rows = getattr(dataset, "rows", None)
    indices = getattr(dataset, "_tensor_indices", None)
    if not isinstance(rows, Sequence) or not isinstance(indices, Sequence):
        raise TypeError("dataset must expose ordered rows and tensor indices")
    if len(rows) != len(indices):
        raise ValueError("dataset rows and tensor indices are misaligned")
    ordered = [
        {
            "position": position,
            "map_id": str(row.get("map_id") or ""),
            "text_id": primary_positive_text_id(row),
            "tensor_index": int(indices[position]),
            "split": str(row.get("split") or getattr(dataset, "split", "")),
            "source": str(row.get("source") or ""),
        }
        for position, row in enumerate(rows)
    ]
    if any(not item["map_id"] for item in ordered):
        raise ValueError("split contains an empty map_id")
    return {
        "split": str(getattr(dataset, "split", "")),
        "n": len(ordered),
        "ordered_rows_sha256": sha256_value(ordered),
        "ordered_map_ids_sha256": sha256_value([item["map_id"] for item in ordered]),
        "ordered_text_ids_sha256": sha256_value([item["text_id"] for item in ordered]),
        "ordered_tensor_indices_sha256": sha256_value(
            [item["tensor_index"] for item in ordered]
        ),
    }


def text_cache_identity(lookup: AtlasFreeTextEmbeddingLookup) -> dict[str, Any]:
    """Return exact tensor and ordered-ID hashes for the Stage 4 cache."""

    return {
        "tensor_sha256": tensor_sha256(lookup.embeddings),
        "state_sha256": sha256_state_dict({"embeddings": lookup.embeddings}),
        "ordered_text_ids_sha256": sha256_value(list(lookup.text_ids)),
        "n": len(lookup),
        "dimension": int(lookup.embeddings.shape[1]),
        "metadata": dict(lookup.metadata),
    }


def validate_checkpoint_binding(
    recorded: Mapping[str, Any],
    expected: Mapping[str, Any],
    *,
    label: str = "checkpoint",
) -> None:
    """Fail loudly unless two machine-readable identity bindings are exact."""

    recorded_hash = sha256_value(recorded)
    expected_hash = sha256_value(expected)
    if recorded_hash != expected_hash:
        recorded_keys = set(recorded)
        expected_keys = set(expected)
        mismatched = sorted(
            key
            for key in recorded_keys & expected_keys
            if sha256_value(recorded[key]) != sha256_value(expected[key])
        )
        missing = sorted(expected_keys - recorded_keys)
        extra = sorted(recorded_keys - expected_keys)
        raise ValueError(
            f"{label} provenance binding mismatch "
            f"(mismatched={mismatched}, missing={missing}, extra={extra})"
        )


def _atomic_torch_save(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        torch.save(dict(payload), temporary)
        with temporary.open("rb") as stream:
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


class AblationCheckpointManager:
    """Atomic multi-objective checkpoints with exact experiment bindings."""

    FORMAT_VERSION = 1
    OBJECTIVES = {
        "val_top5_dice": "best_validation_top5_dice.pt",
        "val_spatial_corr": "best_validation_spatial_correlation.pt",
        "val_global_explained_variance": "best_validation_latent_explained_variance.pt",
        "val_semantic_normalized_auc": "best_validation_semantic_normalized_auc.pt",
    }

    def __init__(
        self,
        run_dir: str | Path,
        *,
        binding: Mapping[str, Any],
        architecture: Mapping[str, Any],
        config: Mapping[str, Any],
    ) -> None:
        self.run_dir = Path(run_dir)
        self.checkpoint_dir = self.run_dir / "checkpoints"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.manifest_path = self.run_dir / "checkpoint_manifest.json"
        # Checkpoint metadata must contain exact built-in JSON primitives.
        # Recent PyTorch exposes ``torch.__version__`` as a TorchVersion
        # subclass, which the restricted weights-only unpickler rejects unless
        # it is canonicalized first.
        self.binding = json_safe(binding)
        self.architecture = json_safe(architecture)
        self.config_sha256 = sha256_value(json_safe(config))
        if self.manifest_path.exists():
            manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
            validate_checkpoint_binding(
                manifest.get("binding") or {},
                self.binding,
                label="checkpoint manifest",
            )
            if manifest.get("architecture_sha256") != sha256_value(self.architecture):
                raise ValueError("checkpoint manifest architecture mismatch")
            if manifest.get("config_sha256") != self.config_sha256:
                raise ValueError("checkpoint manifest effective configuration mismatch")
        else:
            atomic_write_json(
                self.manifest_path,
                {
                    "format_version": self.FORMAT_VERSION,
                    "binding": self.binding,
                    "binding_sha256": sha256_value(self.binding),
                    "architecture_sha256": sha256_value(self.architecture),
                    "config_sha256": self.config_sha256,
                    "checkpoints": {},
                },
            )

    def _manifest(self) -> dict[str, Any]:
        return json.loads(self.manifest_path.read_text(encoding="utf-8"))

    def _payload(
        self,
        projector: nn.Module,
        optimizer: torch.optim.Optimizer,
        *,
        epoch: int,
        metrics: Mapping[str, float],
        scheduler: Any | None,
        early_stopping: Mapping[str, Any],
    ) -> dict[str, Any]:
        payload = {
            "format_version": self.FORMAT_VERSION,
            "epoch": int(epoch),
            "model_state_dict": projector.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "metrics": json_safe(metrics),
            "binding": self.binding,
            "binding_sha256": sha256_value(self.binding),
            "architecture": self.architecture,
            "config_sha256": self.config_sha256,
            "early_stopping": json_safe(early_stopping),
        }
        if scheduler is not None:
            payload["scheduler_state_dict"] = scheduler.state_dict()
        return payload

    def _save(
        self,
        filename: str,
        role: str,
        projector: nn.Module,
        optimizer: torch.optim.Optimizer,
        *,
        epoch: int,
        metrics: Mapping[str, float],
        metric: str | None,
        value: float | None,
        scheduler: Any | None,
        early_stopping: Mapping[str, Any],
    ) -> Path:
        path = self.checkpoint_dir / filename
        _atomic_torch_save(
            path,
            self._payload(
                projector,
                optimizer,
                epoch=epoch,
                metrics=metrics,
                scheduler=scheduler,
                early_stopping=early_stopping,
            ),
        )
        manifest = self._manifest()
        manifest["checkpoints"][role] = {
            "path": path.relative_to(self.run_dir).as_posix(),
            "epoch": int(epoch),
            "metric": metric,
            "value": value,
            "sha256": sha256_file(path),
            "size": path.stat().st_size,
        }
        atomic_write_json(self.manifest_path, manifest)
        return path

    def save_last(
        self,
        projector: nn.Module,
        optimizer: torch.optim.Optimizer,
        *,
        epoch: int,
        metrics: Mapping[str, float],
        scheduler: Any | None = None,
        early_stopping: Mapping[str, Any] | None = None,
    ) -> Path:
        return self._save(
            "last.pt",
            "last",
            projector,
            optimizer,
            epoch=epoch,
            metrics=metrics,
            metric=None,
            value=None,
            scheduler=scheduler,
            early_stopping=early_stopping or {},
        )

    def update_best(
        self,
        metric: str,
        value: float,
        projector: nn.Module,
        optimizer: torch.optim.Optimizer,
        *,
        epoch: int,
        metrics: Mapping[str, float],
        scheduler: Any | None = None,
        early_stopping: Mapping[str, Any] | None = None,
    ) -> Path | None:
        if metric not in self.OBJECTIVES:
            raise ValueError(f"Unknown checkpoint objective {metric!r}")
        value = float(value)
        if not math.isfinite(value):
            return None
        role = metric.removeprefix("val_")
        current = (self._manifest().get("checkpoints", {}).get(role) or {}).get("value")
        if current is not None and value <= float(current):
            return None
        return self._save(
            self.OBJECTIVES[metric],
            role,
            projector,
            optimizer,
            epoch=epoch,
            metrics=metrics,
            metric=metric,
            value=value,
            scheduler=scheduler,
            early_stopping=early_stopping or {},
        )

    def resume(
        self,
        projector: nn.Module,
        optimizer: torch.optim.Optimizer | None = None,
        *,
        scheduler: Any | None = None,
        path: str | Path | None = None,
        map_location: str | torch.device = "cpu",
    ) -> Mapping[str, Any] | None:
        resolved = self.checkpoint_dir / "last.pt" if path is None else Path(path)
        if not resolved.is_absolute():
            resolved = self.run_dir / resolved
        if not resolved.exists():
            return None
        manifest = self._manifest()
        relative = resolved.relative_to(self.run_dir).as_posix()
        record = next(
            (
                item
                for item in manifest.get("checkpoints", {}).values()
                if item.get("path") == relative
            ),
            None,
        )
        if record and record.get("sha256") != sha256_file(resolved):
            raise ValueError(f"Checkpoint file SHA256 mismatch: {relative}")
        try:
            payload = torch.load(
                resolved,
                map_location=map_location,
                weights_only=True,
            )
        except pickle.UnpicklingError as error:
            # Compatibility for checkpoints created by this experiment before
            # metadata canonicalization. Their manifest checksum has already
            # been verified above. Keep the restricted loader enabled and
            # allowlist only PyTorch's inert string-like version class; never
            # fall back to arbitrary pickle execution.
            torch_version_type = type(torch.__version__)
            legacy_global = "torch.torch_version.TorchVersion"
            if (
                legacy_global not in str(error)
                or torch_version_type.__module__ != "torch.torch_version"
                or torch_version_type.__name__ != "TorchVersion"
            ):
                raise
            with torch.serialization.safe_globals([torch_version_type]):
                payload = torch.load(
                    resolved,
                    map_location=map_location,
                    weights_only=True,
                )
        if not isinstance(payload, Mapping):
            raise TypeError("Ablation checkpoint payload must be a mapping")
        validate_checkpoint_binding(
            payload.get("binding") or {},
            self.binding,
            label="checkpoint",
        )
        if payload.get("binding_sha256") != sha256_value(self.binding):
            raise ValueError("Checkpoint binding SHA256 mismatch")
        if sha256_value(payload.get("architecture") or {}) != sha256_value(self.architecture):
            raise ValueError("Checkpoint architecture mismatch")
        if payload.get("config_sha256") != self.config_sha256:
            raise ValueError("Checkpoint effective configuration mismatch")
        projector.load_state_dict(payload["model_state_dict"], strict=True)
        if optimizer is not None and "optimizer_state_dict" in payload:
            optimizer.load_state_dict(payload["optimizer_state_dict"])
        if scheduler is not None and "scheduler_state_dict" in payload:
            scheduler.load_state_dict(payload["scheduler_state_dict"])
        return payload


__all__ = [
    "STAGE4_ABLATION_VARIANTS",
    "AblationCheckpointManager",
    "LatentTransform",
    "LatentTransformKind",
    "Stage4AblationEvaluation",
    "Stage4AblationLoss",
    "Stage4AblationTrainConfig",
    "Stage4AblationVariant",
    "compute_stage4_ablation_loss",
    "encode_stage1_latents",
    "evaluate_stage4_ablation",
    "latent_ablation_metrics",
    "resolve_amp_dtype",
    "split_fingerprint",
    "text_cache_identity",
    "train_stage4_ablation",
    "validate_checkpoint_binding",
]
