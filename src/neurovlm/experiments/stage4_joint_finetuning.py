"""Safety-critical helpers for Stage 4 projector/autoencoder joint adaptation.

The experiment notebook owns orchestration, data loading, evaluation, and
artifact generation.  This module keeps the layer-selection, loss, identity,
and retention rules importable so they can be regression tested without a
GPU or released checkpoints.
"""

from __future__ import annotations

import copy
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from neurovlm.pipelines import sha256_state_dict, sha256_value


JOINT_FINETUNING_VARIANTS = (
    "projector_only_baseline",
    "projector_plus_decoder_output",
    "projector_plus_last_decoder_block",
    "projector_plus_decoder_seed",
    "projector_plus_seed_and_last_block",
    "projector_plus_encoder_head_and_decoder",
    "latent_noise_decoder_adaptation",
)

DEFAULT_NOISE_DECODER_COMPONENTS = (
    "decoder_seed",
    "last_decoder_block",
    "decoder_output",
)

VARIANT_COMPONENTS: dict[str, tuple[str, ...]] = {
    "projector_only_baseline": (),
    "projector_plus_decoder_output": ("decoder_output",),
    "projector_plus_last_decoder_block": (
        "last_decoder_block",
        "decoder_output",
    ),
    "projector_plus_decoder_seed": ("decoder_seed",),
    "projector_plus_seed_and_last_block": DEFAULT_NOISE_DECODER_COMPONENTS,
    "projector_plus_encoder_head_and_decoder": (
        "encoder_head",
        *DEFAULT_NOISE_DECODER_COMPONENTS,
    ),
}


@dataclass(frozen=True)
class JointLossWeights:
    """Explicit weights for the joint generation/replay objective."""

    generation_latent: float = 1.0
    generation_image: float = 1.0
    replay: float = 4.0
    distill: float = 2.0
    parameter: float = 0.0
    generation_foreground: float = 1.0
    replay_foreground: float = 0.0

    def __post_init__(self) -> None:
        for name, value in self.__dict__.items():
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative")


@dataclass
class JointLossResult:
    """Differentiable loss parts and path outputs for one paired batch."""

    total: Tensor
    components: dict[str, Tensor]
    weighted: dict[str, Tensor]
    original_latent: Tensor
    adapted_latent: Tensor
    predicted_latent: Tensor
    generated_volume: Tensor
    clean_replay_volume: Tensor
    replay_volume: Tensor
    original_reconstruction: Tensor


def untouched_autoencoder(autoencoder: nn.Module) -> nn.Module:
    """Return an evaluation-only deep copy that cannot accidentally be trained."""

    original = copy.deepcopy(autoencoder).eval()
    for parameter in original.parameters():
        parameter.requires_grad_(False)
    return original


def _final_linear_parameter_names(module: nn.Module, prefix: str) -> tuple[str, ...]:
    candidates = [
        (name, child)
        for name, child in module.named_modules()
        if isinstance(child, nn.Linear)
    ]
    if not candidates:
        raise ValueError(f"{prefix} does not contain a Linear layer")
    name, layer = candidates[-1]
    parameter_ids = {id(parameter) for parameter in layer.parameters()}
    return tuple(
        full_name
        for full_name, parameter in module.named_parameters()
        if id(parameter) in parameter_ids
    )


def component_parameter_names(autoencoder: nn.Module) -> dict[str, tuple[str, ...]]:
    """Resolve the retained Stage 1 layer roles without hard-coding depth."""

    decoder = getattr(autoencoder, "decoder", None)
    encoder = getattr(autoencoder, "encoder", None)
    if decoder is None or encoder is None:
        raise TypeError("autoencoder must expose encoder and decoder modules")
    up = getattr(decoder, "up", None)
    if not isinstance(up, nn.Sequential) or len(up) < 1:
        raise TypeError("decoder.up must be a non-empty Sequential")

    by_component = {
        "decoder_seed": tuple(
            name
            for name, _ in autoencoder.named_parameters()
            if name.startswith("decoder.fc.")
        ),
        "last_decoder_block": tuple(
            name
            for name, _ in autoencoder.named_parameters()
            if name.startswith(f"decoder.up.{len(up) - 1}.")
        ),
        "decoder_output": tuple(
            name
            for name, _ in autoencoder.named_parameters()
            if name.startswith("decoder.out.")
        ),
    }
    encoder_local = _final_linear_parameter_names(encoder, "encoder")
    by_component["encoder_head"] = tuple(f"encoder.{name}" for name in encoder_local)
    empty = [name for name, values in by_component.items() if not values]
    if empty:
        raise RuntimeError(f"Could not resolve parameters for components: {empty}")
    return by_component


def configure_trainable_variant(
    autoencoder: nn.Module,
    projector: nn.Module,
    variant: str,
    *,
    noise_decoder_components: Sequence[str] = DEFAULT_NOISE_DECODER_COMPONENTS,
) -> dict[str, tuple[tuple[str, nn.Parameter], ...]]:
    """Freeze everything, then expose exact optimizer groups for one variant."""

    if variant not in JOINT_FINETUNING_VARIANTS:
        raise ValueError(f"Unknown joint fine-tuning variant: {variant!r}")
    autoencoder.eval()
    projector.train()
    for parameter in autoencoder.parameters():
        parameter.requires_grad_(False)
    for parameter in projector.parameters():
        parameter.requires_grad_(True)

    resolved = component_parameter_names(autoencoder)
    components = (
        tuple(noise_decoder_components)
        if variant == "latent_noise_decoder_adaptation"
        else VARIANT_COMPONENTS[variant]
    )
    unknown = sorted(set(components) - set(resolved))
    if unknown:
        raise ValueError(f"Unknown AE component selectors: {unknown}")
    selected = {name for component in components for name in resolved[component]}
    for name, parameter in autoencoder.named_parameters():
        parameter.requires_grad_(name in selected)

    projector_group = tuple(
        (f"projector.{name}", parameter)
        for name, parameter in projector.named_parameters()
        if parameter.requires_grad
    )
    decoder_group = tuple(
        (name, parameter)
        for name, parameter in autoencoder.named_parameters()
        if parameter.requires_grad and name.startswith("decoder.")
    )
    encoder_group = tuple(
        (name, parameter)
        for name, parameter in autoencoder.named_parameters()
        if parameter.requires_grad and name.startswith("encoder.")
    )
    groups = {
        "projector": projector_group,
        "decoder": decoder_group,
        "encoder_head": encoder_group,
    }
    flattened = [name for values in groups.values() for name, _ in values]
    if len(flattened) != len(set(flattened)):
        raise RuntimeError("A trainable parameter was assigned to multiple optimizer groups")
    actual = {
        name for name, parameter in autoencoder.named_parameters() if parameter.requires_grad
    }
    if actual != selected:
        raise RuntimeError("AE trainable manifest does not match the variant selector")
    return groups


def trainable_parameter_manifest(
    autoencoder: nn.Module,
    projector: nn.Module,
    groups: Mapping[str, Sequence[tuple[str, nn.Parameter]]],
) -> dict[str, Any]:
    """Return the exact names, shapes, counts, and initial identities."""

    rows = []
    for group_name, values in groups.items():
        for name, parameter in values:
            rows.append(
                {
                    "group": group_name,
                    "name": name,
                    "shape": list(parameter.shape),
                    "numel": parameter.numel(),
                    "dtype": str(parameter.dtype),
                }
            )
    frozen_ae = [
        name
        for name, parameter in autoencoder.named_parameters()
        if not parameter.requires_grad
    ]
    return {
        "trainable": rows,
        "unfrozen_parameter_names": [row["name"] for row in rows],
        "frozen_ae_parameter_names": frozen_ae,
        "trainable_numel_by_group": {
            group: sum(parameter.numel() for _, parameter in values)
            for group, values in groups.items()
        },
        "autoencoder_state_sha256": sha256_state_dict(autoencoder),
        "projector_state_sha256": sha256_state_dict(projector),
    }


def optimizer_group_settings(
    groups: Mapping[str, Sequence[tuple[str, nn.Parameter]]],
    *,
    projector_learning_rate: float = 1e-4,
    decoder_learning_rate: float = 1e-5,
    encoder_head_learning_rate: float = 5e-6,
    weight_decay: float = 1e-4,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Build non-empty AdamW groups plus serializable settings."""

    learning_rates = {
        "projector": projector_learning_rate,
        "decoder": decoder_learning_rate,
        "encoder_head": encoder_head_learning_rate,
    }
    optimizer_groups = []
    settings = []
    for name in ("projector", "decoder", "encoder_head"):
        values = tuple(groups.get(name, ()))
        if not values:
            continue
        lr = float(learning_rates[name])
        optimizer_groups.append(
            {
                "params": [parameter for _, parameter in values],
                "lr": lr,
                "weight_decay": float(weight_decay),
                "name": name,
            }
        )
        settings.append(
            {
                "name": name,
                "learning_rate": lr,
                "weight_decay": float(weight_decay),
                "parameter_names": [parameter_name for parameter_name, _ in values],
                "parameter_count": sum(parameter.numel() for _, parameter in values),
            }
        )
    return optimizer_groups, settings


def parameter_snapshot(
    groups: Mapping[str, Sequence[tuple[str, nn.Parameter]]],
) -> dict[str, Tensor]:
    """Clone trainable parameters by their globally unique manifest names."""

    return {
        name: parameter.detach().cpu().clone()
        for values in groups.values()
        for name, parameter in values
    }


def parameter_distance(
    autoencoder: nn.Module,
    initial_parameters: Mapping[str, Tensor],
) -> Tensor:
    """Return RMS L2 drift for selected decoder parameters only."""

    current = dict(autoencoder.named_parameters())
    terms = []
    for name, initial in initial_parameters.items():
        if not name.startswith("decoder."):
            continue
        if name not in current:
            raise KeyError(f"Initial parameter {name!r} is absent from the autoencoder")
        terms.append(
            (current[name].float() - initial.to(current[name].device).float())
            .square()
            .sum()
        )
    if not terms:
        return torch.zeros((), device=next(autoencoder.parameters()).device)
    denominator = sum(
        current[name].numel()
        for name in initial_parameters
        if name.startswith("decoder.")
    )
    mean_square = torch.stack(terms).sum().div(max(denominator, 1))
    # Preserve a true RMS-L2 scale while avoiding sqrt's infinite derivative
    # at the exact initialization point.
    epsilon = torch.as_tensor(1e-24, device=mean_square.device)
    return (mean_square + epsilon).sqrt() - epsilon.sqrt()


def foreground_mse(prediction: Tensor, target: Tensor) -> Tensor:
    """Mean per-example MSE over positive target voxels, with dense fallback."""

    error = (prediction.float() - target.float()).square().flatten(1)
    foreground = target.float().flatten(1) > 0
    dense = error.mean(dim=1)
    selected = (error * foreground).sum(dim=1) / foreground.sum(dim=1).clamp_min(1)
    return torch.where(foreground.any(dim=1), selected, dense).mean()


def fit_latent_standardization(latents: Tensor, epsilon: float = 1e-6) -> dict[str, Tensor]:
    """Fit train-only per-dimension statistics for optional alignment loss."""

    values = torch.as_tensor(latents).double()
    if values.ndim != 2 or len(values) < 2:
        raise ValueError("latents must have shape N x D with N >= 2")
    return {
        "mean": values.mean(dim=0).float(),
        "scale": values.std(dim=0, unbiased=False).clamp_min(epsilon).float(),
    }


def _alignment_loss(
    prediction: Tensor,
    target: Tensor,
    standardization: Mapping[str, Tensor] | None,
) -> Tensor:
    if standardization is None:
        return F.mse_loss(prediction.float(), target.float())
    mean = standardization["mean"].to(prediction.device).float()
    scale = standardization["scale"].to(prediction.device).float()
    return F.mse_loss(
        (prediction.float() - mean) / scale,
        (target.float() - mean) / scale,
    )


def compute_joint_loss(
    projector: nn.Module,
    adapted_autoencoder: nn.Module,
    original_autoencoder: nn.Module,
    text_embedding: Tensor,
    target_volume: Tensor,
    *,
    weights: JointLossWeights,
    initial_parameters: Mapping[str, Tensor],
    alignment_target: Literal["original", "adapted"] = "original",
    latent_standardization: Mapping[str, Tensor] | None = None,
    replay_noise_std: Tensor | None = None,
    replay_noise_scale: float = 0.0,
    noise_generator: torch.Generator | None = None,
) -> JointLossResult:
    """Compute both required paths and every explicit weighted loss component."""

    if alignment_target not in {"original", "adapted"}:
        raise ValueError("alignment_target must be 'original' or 'adapted'")
    with torch.no_grad():
        original_latent = original_autoencoder.encoder(target_volume)
        original_reconstruction = original_autoencoder.decoder(original_latent)
    adapted_latent = adapted_autoencoder.encoder(target_volume)
    predicted_latent = projector(text_embedding)
    generated_volume = adapted_autoencoder.decoder(predicted_latent)
    clean_replay_volume = adapted_autoencoder.decoder(adapted_latent)

    replay_input = adapted_latent
    if replay_noise_scale:
        if replay_noise_std is None:
            raise ValueError("replay_noise_std is required when replay_noise_scale is non-zero")
        scale = replay_noise_std.to(replay_input.device, replay_input.dtype)
        noise = torch.randn(
            replay_input.shape,
            device=replay_input.device,
            dtype=replay_input.dtype,
            generator=noise_generator,
        )
        replay_input = replay_input + float(replay_noise_scale) * noise * scale
    replay_volume = (
        clean_replay_volume
        if replay_input is adapted_latent
        else adapted_autoencoder.decoder(replay_input)
    )
    adapted_from_original = adapted_autoencoder.decoder(original_latent)

    alignment_reference = (
        original_latent if alignment_target == "original" else adapted_latent
    )
    generation_dense = F.mse_loss(generated_volume.float(), target_volume.float())
    generation_foreground = foreground_mse(generated_volume, target_volume)
    replay_dense = F.mse_loss(replay_volume.float(), target_volume.float())
    replay_foreground = foreground_mse(replay_volume, target_volume)
    components = {
        "latent_alignment": _alignment_loss(
            predicted_latent,
            alignment_reference.detach() if alignment_target == "original" else alignment_reference,
            latent_standardization,
        ),
        "generation_image_loss": (
            generation_dense + weights.generation_foreground * generation_foreground
        ),
        "generation_dense_mse": generation_dense,
        "generation_foreground_mse": generation_foreground,
        "AE_reconstruction_replay": (
            replay_dense + weights.replay_foreground * replay_foreground
        ),
        "replay_dense_mse": replay_dense,
        "replay_foreground_mse": replay_foreground,
        "decoder_output_distillation": F.mse_loss(
            adapted_from_original.float(), original_reconstruction.float()
        ),
        "parameter_distance_regularization": parameter_distance(
            adapted_autoencoder, initial_parameters
        ),
        "adapted_vs_original_latent_mse": F.mse_loss(
            adapted_latent.float(), original_latent.float()
        ),
    }
    weighted = {
        "latent_alignment": weights.generation_latent * components["latent_alignment"],
        "generation_image_loss": weights.generation_image
        * components["generation_image_loss"],
        "AE_reconstruction_replay": weights.replay
        * components["AE_reconstruction_replay"],
        "decoder_output_distillation": weights.distill
        * components["decoder_output_distillation"],
        "parameter_distance_regularization": weights.parameter
        * components["parameter_distance_regularization"],
    }
    total = torch.stack(tuple(weighted.values())).sum()
    return JointLossResult(
        total=total,
        components=components,
        weighted=weighted,
        original_latent=original_latent,
        adapted_latent=adapted_latent,
        predicted_latent=predicted_latent,
        generated_volume=generated_volume,
        clean_replay_volume=clean_replay_volume,
        replay_volume=replay_volume,
        original_reconstruction=original_reconstruction,
    )


def latent_metrics(target: Tensor, prediction: Tensor) -> dict[str, float]:
    """Core raw-latent metrics required by the joint experiment."""

    target = torch.as_tensor(target).double()
    prediction = torch.as_tensor(prediction).double()
    if target.shape != prediction.shape or target.ndim != 2:
        raise ValueError("target and prediction must have matching N x D shapes")
    error = prediction - target
    centered = target - target.mean(dim=0, keepdim=True)
    target_variance = target.var(dim=0, unbiased=False).mean()
    prediction_variance = prediction.var(dim=0, unbiased=False).mean()
    target_norm = target.norm(dim=1).mean()
    prediction_norm = prediction.norm(dim=1).mean()
    sst = centered.square().sum()
    return {
        "latent_mse": float(error.square().mean()),
        "latent_variance_ratio": float(
            prediction_variance / target_variance.clamp_min(1e-12)
        ),
        "latent_norm_ratio": float(prediction_norm / target_norm.clamp_min(1e-12)),
        "explained_variance": float(1.0 - error.square().sum() / sst.clamp_min(1e-12)),
    }


def ae_retention_decision(
    original: Mapping[str, float],
    adapted: Mapping[str, float],
    *,
    maximum_top5_dice_degradation_percent: float = 5.0,
) -> dict[str, Any]:
    """Apply the AE safety rule and report the 1%, 2%, and 5% thresholds."""

    required = ("top5_dice", "spatial_corr", "mse")
    missing = [
        f"{side}.{name}"
        for side, values in (("original", original), ("adapted", adapted))
        for name in required
        if name not in values
    ]
    if missing:
        raise KeyError("Missing retention metrics: " + ", ".join(missing))

    def decrease(name: str) -> float:
        baseline = float(original[name])
        return 100.0 * (baseline - float(adapted[name])) / max(abs(baseline), 1e-12)

    def increase(name: str) -> float:
        baseline = float(original[name])
        return 100.0 * (float(adapted[name]) - baseline) / max(abs(baseline), 1e-12)

    top5_degradation = decrease("top5_dice")
    tolerance = 1e-9
    decision = {
        "top5_dice_degradation_percent": top5_degradation,
        "spatial_corr_degradation_percent": decrease("spatial_corr"),
        "mse_degradation_percent": increase("mse"),
        "maximum_top5_dice_degradation_percent": float(
            maximum_top5_dice_degradation_percent
        ),
        "satisfies_1_percent": top5_degradation <= 1.0 + tolerance,
        "satisfies_2_percent": top5_degradation <= 2.0 + tolerance,
        "satisfies_5_percent": top5_degradation <= 5.0 + tolerance,
        "safe": (
            top5_degradation
            <= maximum_top5_dice_degradation_percent + tolerance
        ),
    }
    decision["action"] = "accept" if decision["safe"] else "reject_and_stop"
    return decision


def checkpoint_binding(
    *,
    original_ae_identity: Mapping[str, Any],
    starting_ae: nn.Module,
    adapted_ae: nn.Module,
    projector: nn.Module,
    text_cache_identity: Mapping[str, Any],
    split_fingerprints: Mapping[str, Any],
    unfrozen_parameter_names: Sequence[str],
    loss_weights: Mapping[str, Any],
    optimizer_groups: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Create the complete identity payload required in every checkpoint."""

    payload = {
        "original_ae_identity": dict(original_ae_identity),
        "starting_ae_state_identity": sha256_state_dict(starting_ae),
        "current_trainable_module_identity": {
            "adapted_ae_state_sha256": sha256_state_dict(adapted_ae),
            "projector_state_sha256": sha256_state_dict(projector),
        },
        "text_cache_identity": dict(text_cache_identity),
        "split_fingerprints": dict(split_fingerprints),
        "exact_unfrozen_parameter_names": list(unfrozen_parameter_names),
        "loss_weights": dict(loss_weights),
        "optimizer_group_settings": [dict(value) for value in optimizer_groups],
    }
    payload["binding_sha256"] = sha256_value(payload)
    return payload


def validate_checkpoint_binding(
    recorded: Mapping[str, Any],
    expected_static: Mapping[str, Any],
) -> None:
    """Reject resume if any immutable experiment identity has changed."""

    immutable = (
        "original_ae_identity",
        "starting_ae_state_identity",
        "text_cache_identity",
        "split_fingerprints",
        "exact_unfrozen_parameter_names",
        "loss_weights",
        "optimizer_group_settings",
    )
    mismatches = [
        name
        for name in immutable
        if sha256_value(recorded.get(name)) != sha256_value(expected_static.get(name))
    ]
    if mismatches:
        raise ValueError(
            "Checkpoint identity mismatch for immutable fields: "
            + ", ".join(mismatches)
        )


def assert_original_untouched(original: nn.Module, identity: str) -> None:
    """Fail immediately if the evaluation-only Stage 1 copy was modified."""

    actual = sha256_state_dict(original)
    if actual != identity:
        raise RuntimeError(
            f"Untouched original AE changed: expected {identity}, observed {actual}"
        )


def assert_frozen_parameters_unchanged(
    module: nn.Module,
    initial_state: Mapping[str, Tensor],
    unfrozen_parameter_names: Sequence[str],
) -> None:
    """Verify every unselected parameter and every buffer stayed bit-identical."""

    allowed = set(unfrozen_parameter_names)
    for name, value in module.state_dict().items():
        if name in allowed:
            continue
        if name not in initial_state or not torch.equal(
            value.detach().cpu(), initial_state[name].detach().cpu()
        ):
            raise RuntimeError(f"Frozen state entry changed: {name}")


__all__ = [
    "DEFAULT_NOISE_DECODER_COMPONENTS",
    "JOINT_FINETUNING_VARIANTS",
    "JointLossResult",
    "JointLossWeights",
    "VARIANT_COMPONENTS",
    "ae_retention_decision",
    "assert_frozen_parameters_unchanged",
    "assert_original_untouched",
    "checkpoint_binding",
    "component_parameter_names",
    "compute_joint_loss",
    "configure_trainable_variant",
    "fit_latent_standardization",
    "foreground_mse",
    "latent_metrics",
    "optimizer_group_settings",
    "parameter_distance",
    "parameter_snapshot",
    "trainable_parameter_manifest",
    "untouched_autoencoder",
    "validate_checkpoint_binding",
]
