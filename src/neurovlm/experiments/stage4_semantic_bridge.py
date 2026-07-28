"""Opt-in Stage 3 semantic-to-Stage 1 AE bridge experiments.

The production Stage 3 and Stage 4 modules are deliberately not changed here.
Models in this module consume already-normalized Stage 3 semantic embeddings
and emit unconstrained raw Stage 1 autoencoder latents.
"""

from __future__ import annotations

import json
import math
import os
import tempfile
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from neurovlm.cnn import GenerativeTextToAELatent
from neurovlm.pipelines import (
    atomic_write_json,
    sha256_file,
    sha256_state_dict,
    sha256_value,
)


RAW_TEXT_DIM = 768
SEMANTIC_DIM = 384
AE_LATENT_DIM = 384
BRIDGE_FORMAT_VERSION = 1

BRIDGE_ARCHITECTURES = (
    "mlp_512",
    "deep_mlp_1024",
    "residual_mlp_1024",
)
BRIDGE_PATHS = (
    "direct_baseline",
    "stage3_text_bridge",
    "stage3_brain_bridge_oracle",
    "shared_bridge_dual_supervision",
    "concatenated_text_semantic",
    "residual_direct_plus_semantic",
)
BRIDGE_LOSS_VARIANTS = (
    "primary_raw_decoded",
    "standardized_decoded",
    "standardized_cosine_decoded",
    "standardized_cosine_norm_decoded",
)

BridgeArchitecture = Literal[
    "mlp_512",
    "deep_mlp_1024",
    "residual_mlp_1024",
]
BridgePath = Literal[
    "direct_baseline",
    "stage3_text_bridge",
    "stage3_brain_bridge_oracle",
    "shared_bridge_dual_supervision",
    "concatenated_text_semantic",
    "residual_direct_plus_semantic",
]
BridgeLossVariant = Literal[
    "primary_raw_decoded",
    "standardized_decoded",
    "standardized_cosine_decoded",
    "standardized_cosine_norm_decoded",
]


def freeze_module(module: nn.Module) -> nn.Module:
    """Put a module in evaluation mode and disable all parameter gradients."""

    module.eval()
    for parameter in module.parameters():
        parameter.requires_grad_(False)
    return module


class _ResidualBlock(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.fc1 = nn.Linear(width, width)
        self.activation = nn.GELU()
        self.fc2 = nn.Linear(width, width)

    def forward(self, value: Tensor) -> Tensor:
        return value + self.fc2(self.activation(self.fc1(self.norm(value))))


class SemanticToAELatentBridge(nn.Module):
    """Map a normalized 384-d semantic vector to a raw 384-d AE latent."""

    def __init__(self, architecture: BridgeArchitecture = "mlp_512") -> None:
        super().__init__()
        if architecture not in BRIDGE_ARCHITECTURES:
            raise ValueError(
                f"Unknown bridge architecture {architecture!r}; "
                f"expected one of {BRIDGE_ARCHITECTURES}"
            )
        self.architecture = architecture
        if architecture == "mlp_512":
            self.net = nn.Sequential(
                nn.Linear(SEMANTIC_DIM, 512),
                nn.GELU(),
                nn.Linear(512, AE_LATENT_DIM),
            )
        elif architecture == "deep_mlp_1024":
            self.net = nn.Sequential(
                nn.Linear(SEMANTIC_DIM, 1024),
                nn.GELU(),
                nn.Linear(1024, 1024),
                nn.GELU(),
                nn.Linear(1024, AE_LATENT_DIM),
            )
        else:
            self.net = nn.Sequential(
                nn.Linear(SEMANTIC_DIM, 1024),
                _ResidualBlock(1024),
                nn.LayerNorm(1024),
                nn.Linear(1024, AE_LATENT_DIM),
            )

    def forward(self, semantic: Tensor) -> Tensor:
        if semantic.shape[-1] != SEMANTIC_DIM:
            raise ValueError(
                f"Expected semantic dimension {SEMANTIC_DIM}, got {semantic.shape[-1]}"
            )
        # Deliberately no normalization, clipping, bounded activation, or
        # other transform after the final Linear.
        return self.net(semantic)


class SharedDualSemanticBridge(nn.Module):
    """Apply one shared bridge to text and brain semantic embeddings."""

    def __init__(self, architecture: BridgeArchitecture = "mlp_512") -> None:
        super().__init__()
        self.bridge = SemanticToAELatentBridge(architecture)

    def forward(
        self,
        text_semantic: Tensor,
        brain_semantic: Tensor,
    ) -> tuple[Tensor, Tensor]:
        return self.bridge(text_semantic), self.bridge(brain_semantic)


class ConcatenatedTextSemanticProjector(nn.Module):
    """Map normalized raw text plus normalized Stage 3 semantics to a raw latent."""

    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(RAW_TEXT_DIM + SEMANTIC_DIM, 1024),
            nn.GELU(),
            nn.Linear(1024, 1024),
            nn.GELU(),
            nn.Linear(1024, AE_LATENT_DIM),
        )

    def forward(self, raw_text: Tensor, text_semantic: Tensor) -> Tensor:
        if raw_text.shape[-1] != RAW_TEXT_DIM:
            raise ValueError(f"Expected raw text dimension {RAW_TEXT_DIM}")
        if text_semantic.shape[-1] != SEMANTIC_DIM:
            raise ValueError(f"Expected semantic dimension {SEMANTIC_DIM}")
        return self.net(torch.cat((raw_text, text_semantic), dim=-1))


class ResidualDirectPlusSemantic(nn.Module):
    """Add a semantic residual to the retained direct Stage 4 projector."""

    def __init__(self, architecture: BridgeArchitecture = "mlp_512") -> None:
        super().__init__()
        self.direct = GenerativeTextToAELatent(
            RAW_TEXT_DIM,
            512,
            AE_LATENT_DIM,
        )
        self.semantic_residual = SemanticToAELatentBridge(architecture)

    def forward(self, raw_text: Tensor, text_semantic: Tensor) -> Tensor:
        return self.direct(raw_text) + self.semantic_residual(text_semantic)


def build_bridge_model(
    path: BridgePath,
    *,
    architecture: BridgeArchitecture = "mlp_512",
) -> nn.Module:
    """Build one of the six diagnostic conditioning paths."""

    if path not in BRIDGE_PATHS:
        raise ValueError(f"Unknown path {path!r}; expected one of {BRIDGE_PATHS}")
    if path == "direct_baseline":
        return GenerativeTextToAELatent(RAW_TEXT_DIM, 512, AE_LATENT_DIM)
    if path in {"stage3_text_bridge", "stage3_brain_bridge_oracle"}:
        return SemanticToAELatentBridge(architecture)
    if path == "shared_bridge_dual_supervision":
        return SharedDualSemanticBridge(architecture)
    if path == "concatenated_text_semantic":
        return ConcatenatedTextSemanticProjector()
    if path == "residual_direct_plus_semantic":
        return ResidualDirectPlusSemantic(architecture)
    raise AssertionError("Unhandled validated bridge path")


def bridge_architecture_record(
    path: BridgePath,
    architecture: BridgeArchitecture,
    module: nn.Module,
) -> dict[str, Any]:
    """Return stable architecture metadata for provenance and checkpoints."""

    if path == "direct_baseline":
        layers: list[Any] = [768, 512, "ReLU", 384]
        effective_architecture = "retained_direct_768_512_384"
    elif path == "concatenated_text_semantic":
        layers = [1152, 1024, "GELU", 1024, "GELU", 384]
        effective_architecture = "concatenated_1152_deep_mlp"
    elif architecture == "mlp_512":
        layers = [384, 512, "GELU", 384]
        effective_architecture = architecture
    elif architecture == "deep_mlp_1024":
        layers = [384, 1024, "GELU", 1024, "GELU", 384]
        effective_architecture = architecture
    else:
        layers = [
            384,
            1024,
            {
                "block": "pre_norm_residual_mlp",
                "width": 1024,
                "activation": "GELU",
            },
            "LayerNorm(1024)",
            384,
        ]
        effective_architecture = architecture
    return {
        "format_version": BRIDGE_FORMAT_VERSION,
        "namespace": "neurovlm.experiments.stage4_semantic_bridge",
        "path": path,
        "architecture": effective_architecture,
        "layers": layers,
        "raw_text_dim": RAW_TEXT_DIM,
        "semantic_dim": SEMANTIC_DIM,
        "ae_latent_dim": AE_LATENT_DIM,
        "semantic_input_convention": "l2_unit_normalized",
        "decoder_input_convention": "raw_stage1_ae_latent",
        "final_output_transform": None,
        "parameter_count": sum(parameter.numel() for parameter in module.parameters()),
    }


@dataclass(frozen=True)
class BridgeLossConfig:
    """One explicitly labeled optimization objective."""

    variant: BridgeLossVariant = "primary_raw_decoded"
    latent_weight: float = 1.0
    decoded_weight: float = 1.0
    cosine_weight: float = 0.1
    norm_weight: float = 0.1
    epsilon: float = 1e-6

    def __post_init__(self) -> None:
        if self.variant not in BRIDGE_LOSS_VARIANTS:
            raise ValueError(
                f"Unknown loss variant {self.variant!r}; "
                f"expected one of {BRIDGE_LOSS_VARIANTS}"
            )
        for name in (
            "latent_weight",
            "decoded_weight",
            "cosine_weight",
            "norm_weight",
            "epsilon",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative")
        if self.epsilon == 0:
            raise ValueError("epsilon must be positive")

    def effective_dict(self) -> dict[str, Any]:
        output = asdict(self)
        output["architecture_loss_axis"] = (
            "primary_architecture_comparison"
            if self.variant == "primary_raw_decoded"
            else "separate_loss_sensitivity"
        )
        return output


@dataclass(frozen=True)
class BridgeLoss:
    total: Tensor
    raw_latent_mse: Tensor
    standardized_latent_mse: Tensor
    decoded_volume_mse: Tensor
    latent_cosine_loss: Tensor
    latent_norm_loss: Tensor

    def detached(self) -> dict[str, float]:
        return {
            name: float(getattr(self, name).detach())
            for name in (
                "total",
                "raw_latent_mse",
                "standardized_latent_mse",
                "decoded_volume_mse",
                "latent_cosine_loss",
                "latent_norm_loss",
            )
        }


def compute_bridge_loss(
    prediction_raw: Tensor,
    target_raw: Tensor,
    prediction_volume: Tensor,
    target_volume: Tensor,
    *,
    training_latent_mean: Tensor,
    training_latent_std: Tensor,
    config: BridgeLossConfig,
) -> BridgeLoss:
    """Compute one loss without silently mixing objective variants."""

    if prediction_raw.shape != target_raw.shape:
        raise ValueError("prediction_raw and target_raw must have matching shapes")
    if prediction_volume.shape != target_volume.shape:
        raise ValueError("prediction_volume and target_volume must have matching shapes")
    mean = training_latent_mean.to(
        device=prediction_raw.device,
        dtype=prediction_raw.dtype,
    )
    std = training_latent_std.to(
        device=prediction_raw.device,
        dtype=prediction_raw.dtype,
    ).clamp_min(config.epsilon)
    raw_mse = F.mse_loss(prediction_raw, target_raw)
    standardized_mse = F.mse_loss(
        (prediction_raw - mean) / std,
        (target_raw - mean) / std,
    )
    decoded_mse = F.mse_loss(prediction_volume.float(), target_volume.float())
    cosine = (
        1
        - F.cosine_similarity(
            prediction_raw.float(),
            target_raw.float(),
            dim=-1,
            eps=config.epsilon,
        ).mean()
    )
    target_norm = target_raw.float().norm(dim=-1).clamp_min(config.epsilon)
    prediction_norm = prediction_raw.float().norm(dim=-1)
    norm = ((prediction_norm / target_norm) - 1).square().mean()

    if config.variant == "primary_raw_decoded":
        latent_term = raw_mse
        cosine_term = raw_mse.new_zeros(())
        norm_term = raw_mse.new_zeros(())
    else:
        latent_term = standardized_mse
        cosine_term = (
            cosine
            if config.variant
            in {"standardized_cosine_decoded", "standardized_cosine_norm_decoded"}
            else cosine.new_zeros(())
        )
        norm_term = (
            norm
            if config.variant == "standardized_cosine_norm_decoded"
            else norm.new_zeros(())
        )
    total = (
        config.latent_weight * latent_term
        + config.decoded_weight * decoded_mse
        + config.cosine_weight * cosine_term
        + config.norm_weight * norm_term
    )
    return BridgeLoss(
        total=total,
        raw_latent_mse=raw_mse,
        standardized_latent_mse=standardized_mse,
        decoded_volume_mse=decoded_mse,
        latent_cosine_loss=cosine,
        latent_norm_loss=norm,
    )


def fixed_derangement(length: int, seed: int) -> Tensor:
    """Return a deterministic permutation with no fixed points."""

    if length < 2:
        raise ValueError("A shuffled control requires at least two examples")
    generator = torch.Generator().manual_seed(int(seed))
    order = torch.randperm(length, generator=generator)
    deranged = torch.empty_like(order)
    deranged[order] = order.roll(1)
    if bool((deranged == torch.arange(length)).any()):
        raise AssertionError("Internal derangement construction produced a fixed point")
    return deranged


def validate_semantic_embeddings(
    value: Tensor,
    *,
    label: str,
    dimension: int = SEMANTIC_DIM,
    atol: float = 1e-4,
) -> dict[str, float | int | str]:
    """Fail unless a Stage 3 tensor is finite, 384-d, and unit normalized."""

    semantic = torch.as_tensor(value).detach().float()
    if semantic.ndim != 2 or semantic.shape[1] != dimension:
        raise ValueError(
            f"{label} must have shape N x {dimension}, got {tuple(semantic.shape)}"
        )
    if not bool(torch.isfinite(semantic).all()):
        raise ValueError(f"{label} contains NaNs or infinities")
    norms = semantic.norm(dim=1)
    maximum_error = float((norms - 1).abs().max()) if len(norms) else 0.0
    if maximum_error > atol:
        raise ValueError(
            f"{label} violates L2-unit normalization: max norm error={maximum_error}"
        )
    return {
        "label": label,
        "n": int(len(semantic)),
        "dimension": int(dimension),
        "normalization": "l2_unit_normalized",
        "mean_norm": float(norms.mean()) if len(norms) else float("nan"),
        "max_norm_error": maximum_error,
        "atol": float(atol),
    }


def semantic_alignment_metrics(
    text_semantic: Tensor,
    brain_semantic: Tensor,
    *,
    shuffled_indices: Tensor | None = None,
) -> dict[str, float]:
    """Measure matched and mismatched Stage 3 text/brain alignment."""

    text = F.normalize(torch.as_tensor(text_semantic).float(), dim=1, eps=1e-8)
    brain = F.normalize(torch.as_tensor(brain_semantic).float(), dim=1, eps=1e-8)
    if text.shape != brain.shape or text.ndim != 2:
        raise ValueError("text_semantic and brain_semantic must have matching N x D shapes")
    if shuffled_indices is None:
        shuffled_indices = fixed_derangement(len(text), 0)
    shuffled_indices = torch.as_tensor(shuffled_indices, dtype=torch.long)
    if shuffled_indices.shape != (len(text),):
        raise ValueError("shuffled_indices must have one index per example")
    matched = (text * brain).sum(dim=1)
    shuffled = (text * brain[shuffled_indices]).sum(dim=1)
    return {
        "stage3_text_brain_matched_cosine": float(matched.mean()),
        "stage3_text_brain_shuffled_cosine": float(shuffled.mean()),
        "stage3_matched_minus_shuffled_cosine": float(
            matched.mean() - shuffled.mean()
        ),
    }


def _nearest_distances(
    prediction: Tensor,
    reference: Tensor,
    *,
    prediction_chunk_size: int,
    reference_chunk_size: int,
    device: str | torch.device | None,
) -> Tensor:
    if prediction_chunk_size < 1 or reference_chunk_size < 1:
        raise ValueError("distance chunk sizes must be positive")
    resolved = torch.device(
        device
        if device is not None
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    reference = reference.to(resolved)
    chunks: list[Tensor] = []
    for start in range(0, len(prediction), prediction_chunk_size):
        query = prediction[start : start + prediction_chunk_size].to(resolved)
        nearest = torch.full(
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
            nearest = torch.minimum(nearest, distances.min(dim=1).values)
        chunks.append(nearest.cpu())
    return torch.cat(chunks)


def bridge_latent_metrics(
    target_raw: Tensor,
    prediction_raw: Tensor,
    *,
    training_mean: Tensor,
    training_std: Tensor,
    nearest_reference: Tensor,
    prediction_chunk_size: int = 512,
    reference_chunk_size: int = 4096,
    distance_device: str | torch.device | None = None,
) -> tuple[dict[str, float], list[dict[str, float]]]:
    """Compute all requested raw-AE-latent diagnostics and per-dimension R²."""

    target = torch.as_tensor(target_raw).detach().float().cpu()
    prediction = torch.as_tensor(prediction_raw).detach().float().cpu()
    reference = torch.as_tensor(nearest_reference).detach().float().cpu()
    mean = torch.as_tensor(training_mean).detach().float().cpu().flatten()
    std = torch.as_tensor(training_std).detach().float().cpu().flatten().clamp_min(1e-6)
    if target.ndim != 2 or target.shape != prediction.shape or len(target) < 2:
        raise ValueError("target_raw and prediction_raw must match as N x D with N >= 2")
    if mean.shape != (target.shape[1],) or std.shape != mean.shape:
        raise ValueError("training_mean and training_std must have D values")
    if reference.ndim != 2 or reference.shape[1] != target.shape[1] or not len(reference):
        raise ValueError("nearest_reference must be a non-empty M x D tensor")

    target_mean = target.mean(dim=0)
    prediction_mean = prediction.mean(dim=0)
    target_variance = target.var(dim=0, unbiased=False)
    prediction_variance = prediction.var(dim=0, unbiased=False)
    residual = target - prediction
    residual_variance = residual.var(dim=0, unbiased=False)
    ss_res = residual.square().sum(dim=0)
    ss_tot = (target - target_mean).square().sum(dim=0)
    r_squared = torch.where(
        ss_tot > 0,
        1 - ss_res / ss_tot,
        torch.zeros_like(ss_tot),
    )
    total_target_variance = target_variance.sum().clamp_min(1e-12)
    nearest = _nearest_distances(
        prediction,
        reference,
        prediction_chunk_size=prediction_chunk_size,
        reference_chunk_size=reference_chunk_size,
        device=distance_device,
    )
    target_norm = target.norm(dim=1)
    prediction_norm = prediction.norm(dim=1)
    standardized_error = (prediction - target) / std
    metrics = {
        "raw_latent_mse": float(F.mse_loss(prediction, target)),
        "standardized_latent_mse": float(standardized_error.square().mean()),
        "latent_cosine": float(
            F.cosine_similarity(prediction, target, dim=1, eps=1e-8).mean()
        ),
        "latent_variance_ratio": float(
            prediction_variance.sum() / total_target_variance
        ),
        "latent_norm_ratio": float(
            prediction_norm.mean() / target_norm.mean().clamp_min(1e-12)
        ),
        "global_explained_variance": float(
            1 - residual_variance.sum() / total_target_variance
        ),
        "mean_per_dimension_r_squared": float(r_squared.mean()),
        "nearest_real_latent_distance": float(nearest.mean()),
        "target_latent_norm_mean": float(target_norm.mean()),
        "predicted_latent_norm_mean": float(prediction_norm.mean()),
    }
    rows = [
        {
            "dimension": int(index),
            "training_mean": float(mean[index]),
            "training_std": float(std[index]),
            "target_mean": float(target_mean[index]),
            "prediction_mean": float(prediction_mean[index]),
            "target_variance": float(target_variance[index]),
            "prediction_variance": float(prediction_variance[index]),
            "variance_ratio": float(
                prediction_variance[index]
                / target_variance[index].clamp_min(1e-12)
            ),
            "r_squared": float(r_squared[index]),
        }
        for index in range(target.shape[1])
    ]
    return metrics, rows


def stage3_identity(
    stage3_model: nn.Module,
    *,
    checkpoint: str | Path,
    branch: str,
) -> dict[str, Any]:
    """Fingerprint the composite Stage 3 file and its two frozen components."""

    brain_encoder = getattr(stage3_model, "brain_encoder", None)
    text_projection = getattr(stage3_model, "text_projection", None)
    if not isinstance(brain_encoder, nn.Module) or not isinstance(
        text_projection, nn.Module
    ):
        raise TypeError("Stage 3 model must expose brain_encoder and text_projection")
    path = Path(checkpoint)
    if not path.is_file():
        raise FileNotFoundError(path)
    checkpoint_sha256 = sha256_file(path)
    return {
        "branch": branch,
        "composite_checkpoint_path": str(path.absolute()),
        "composite_checkpoint_sha256": checkpoint_sha256,
        "composite_state_sha256": sha256_state_dict(stage3_model),
        "brain_encoder": {
            "checkpoint_path": str(path.absolute()),
            "checkpoint_sha256": checkpoint_sha256,
            "state_sha256": sha256_state_dict(brain_encoder),
        },
        "text_projection": {
            "checkpoint_path": str(path.absolute()),
            "checkpoint_sha256": checkpoint_sha256,
            "state_sha256": sha256_state_dict(text_projection),
        },
        "semantic_dimension": SEMANTIC_DIM,
        "normalization_convention": "F.normalize(..., dim=-1), l2_unit_normalized",
    }


_OBJECTIVES = {
    "top5_dice": ("top5_dice", "max", "best_top5_dice.pt"),
    "spatial_correlation": ("spatial_corr", "max", "best_spatial_correlation.pt"),
    "latent_explained_variance": (
        "global_explained_variance",
        "max",
        "best_latent_explained_variance.pt",
    ),
    "semantic_normalized_auc": (
        "semantic_normalized_auc",
        "max",
        "best_semantic_normalized_auc.pt",
    ),
}


def _atomic_torch_save(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        torch.save(dict(payload), temporary)
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


class BridgeCheckpointManager:
    """Multi-objective, provenance-bound checkpoints with strict resume."""

    def __init__(
        self,
        run_dir: str | Path,
        *,
        binding: Mapping[str, Any],
        effective_config: Mapping[str, Any],
        architecture: Mapping[str, Any],
    ) -> None:
        self.run_dir = Path(run_dir)
        self.checkpoint_dir = self.run_dir / "checkpoints"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.manifest_path = self.checkpoint_dir / "checkpoint_manifest.json"
        self.binding = dict(binding)
        self.effective_config = dict(effective_config)
        self.architecture = dict(architecture)
        self.binding_sha256 = sha256_value(self.binding)
        self.config_sha256 = sha256_value(self.effective_config)
        self.architecture_sha256 = sha256_value(self.architecture)

    def _manifest(self) -> dict[str, Any]:
        if self.manifest_path.exists():
            manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
            expected = {
                "binding_sha256": self.binding_sha256,
                "config_sha256": self.config_sha256,
                "architecture_sha256": self.architecture_sha256,
            }
            mismatched = [
                key for key, value in expected.items() if manifest.get(key) != value
            ]
            if mismatched:
                raise ValueError(
                    "Checkpoint manifest identity mismatch: "
                    + ", ".join(mismatched)
                )
            return manifest
        return {
            "format_version": BRIDGE_FORMAT_VERSION,
            "binding_sha256": self.binding_sha256,
            "config_sha256": self.config_sha256,
            "architecture_sha256": self.architecture_sha256,
            "objectives": {},
            "checkpoints": {},
        }

    def _payload(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer | None,
        *,
        epoch: int,
        metrics: Mapping[str, Any],
        role: str,
        extra: Mapping[str, Any] | None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "format_version": BRIDGE_FORMAT_VERSION,
            "role": role,
            "epoch": int(epoch),
            "metrics": dict(metrics),
            "binding": self.binding,
            "binding_sha256": self.binding_sha256,
            "effective_config": self.effective_config,
            "config_sha256": self.config_sha256,
            "architecture": self.architecture,
            "architecture_sha256": self.architecture_sha256,
            "model_state_dict": model.state_dict(),
        }
        if optimizer is not None:
            payload["optimizer_state_dict"] = optimizer.state_dict()
        if extra:
            payload["extra"] = dict(extra)
        return payload

    def _save(
        self,
        filename: str,
        role: str,
        model: nn.Module,
        optimizer: torch.optim.Optimizer | None,
        *,
        epoch: int,
        metrics: Mapping[str, Any],
        metric_name: str | None,
        metric_value: float | None,
        extra: Mapping[str, Any] | None,
    ) -> Path:
        path = self.checkpoint_dir / filename
        _atomic_torch_save(
            path,
            self._payload(
                model,
                optimizer,
                epoch=epoch,
                metrics=metrics,
                role=role,
                extra=extra,
            ),
        )
        manifest = self._manifest()
        manifest["checkpoints"][role] = {
            "path": path.relative_to(self.run_dir).as_posix(),
            "epoch": int(epoch),
            "metric": metric_name,
            "value": metric_value,
            "sha256": sha256_file(path),
        }
        atomic_write_json(self.manifest_path, manifest)
        return path

    def save_epoch(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        *,
        epoch: int,
        metrics: Mapping[str, Any],
        extra: Mapping[str, Any] | None = None,
    ) -> dict[str, Path]:
        """Save every improved validation objective plus the last state."""

        manifest = self._manifest()
        saved: dict[str, Path] = {}
        for role, (metric_name, direction, filename) in _OBJECTIVES.items():
            value = float(metrics.get(metric_name, float("nan")))
            if not math.isfinite(value):
                continue
            previous = (manifest.get("objectives", {}).get(role) or {}).get("value")
            improves = previous is None or (
                value > float(previous) if direction == "max" else value < float(previous)
            )
            if not improves:
                continue
            saved[role] = self._save(
                filename,
                role,
                model,
                optimizer,
                epoch=epoch,
                metrics=metrics,
                metric_name=metric_name,
                metric_value=value,
                extra=extra,
            )
            manifest = self._manifest()
            manifest["objectives"][role] = {
                "metric": metric_name,
                "direction": direction,
                "value": value,
                "epoch": int(epoch),
            }
            atomic_write_json(self.manifest_path, manifest)
        saved["last"] = self._save(
            "last.pt",
            "last",
            model,
            optimizer,
            epoch=epoch,
            metrics=metrics,
            metric_name=None,
            metric_value=None,
            extra=extra,
        )
        return saved

    def path_for(self, role: str) -> Path:
        record = (self._manifest().get("checkpoints") or {}).get(role)
        if not record:
            raise FileNotFoundError(f"No checkpoint recorded for role {role!r}")
        return self.run_dir / record["path"]

    def load(
        self,
        role_or_path: str | Path,
        *,
        model: nn.Module,
        optimizer: torch.optim.Optimizer | None = None,
        map_location: str | torch.device = "cpu",
    ) -> Mapping[str, Any]:
        """Load only after file, binding, config, and architecture validation."""

        candidate = Path(role_or_path)
        role = str(role_or_path)
        path = (
            candidate
            if candidate.is_file()
            else self.path_for(role)
        )
        manifest = self._manifest()
        records = manifest.get("checkpoints") or {}
        matching = [
            record
            for record in records.values()
            if self.run_dir / record["path"] == path
        ]
        if matching and sha256_file(path) != matching[-1]["sha256"]:
            raise ValueError(f"Checkpoint file SHA256 mismatch: {path}")
        payload = torch.load(path, map_location=map_location, weights_only=True)
        if not isinstance(payload, Mapping):
            raise TypeError("Checkpoint payload must be a mapping")
        expected = {
            "binding_sha256": self.binding_sha256,
            "config_sha256": self.config_sha256,
            "architecture_sha256": self.architecture_sha256,
        }
        mismatched = [
            key for key, value in expected.items() if payload.get(key) != value
        ]
        if mismatched:
            raise ValueError(
                "Checkpoint provenance mismatch: " + ", ".join(mismatched)
            )
        model.load_state_dict(payload["model_state_dict"], strict=True)
        if optimizer is not None and "optimizer_state_dict" in payload:
            optimizer.load_state_dict(payload["optimizer_state_dict"])
        return payload


__all__ = [
    "AE_LATENT_DIM",
    "BRIDGE_ARCHITECTURES",
    "BRIDGE_FORMAT_VERSION",
    "BRIDGE_LOSS_VARIANTS",
    "BRIDGE_PATHS",
    "RAW_TEXT_DIM",
    "SEMANTIC_DIM",
    "BridgeCheckpointManager",
    "BridgeLoss",
    "BridgeLossConfig",
    "ConcatenatedTextSemanticProjector",
    "ResidualDirectPlusSemantic",
    "SemanticToAELatentBridge",
    "SharedDualSemanticBridge",
    "bridge_architecture_record",
    "bridge_latent_metrics",
    "build_bridge_model",
    "compute_bridge_loss",
    "fixed_derangement",
    "freeze_module",
    "semantic_alignment_metrics",
    "stage3_identity",
    "validate_semantic_embeddings",
]
