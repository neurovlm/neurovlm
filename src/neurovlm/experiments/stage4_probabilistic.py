"""Conditional probabilistic generation in the frozen Stage 1 AE latent space.

This module is deliberately opt-in and does not alter the released Stage 4
projector or any Stage 1 autoencoder.  The conditional VAE predicts
standardized 384-dimensional Stage 1 latents; callers must use
``LatentStandardization.inverse`` before invoking the frozen decoder.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from neurovlm.pipelines import sha256_state_dict, sha256_value


TEXT_DIM = 768
AE_LATENT_DIM = 384
POSTERIOR_INPUT_DIM = TEXT_DIM + AE_LATENT_DIM
SUPPORTED_U_DIMS = (32, 64, 128)
CHECKPOINT_FORMAT_VERSION = 1
ACTIVE_DIM_KL_THRESHOLD = 0.01


@dataclass(frozen=True)
class ConditionalVAEConfig:
    """Identity-bearing conditional VAE configuration."""

    u_dim: int = 64
    condition_dropout: float = 0.0
    posterior_dropout: float = 0.0
    posterior_dropout_justification: str | None = None
    text_dim: int = TEXT_DIM
    ae_latent_dim: int = AE_LATENT_DIM

    def __post_init__(self) -> None:
        if self.u_dim not in SUPPORTED_U_DIMS:
            raise ValueError(f"u_dim must be one of {SUPPORTED_U_DIMS}")
        for name, value in (
            ("condition_dropout", self.condition_dropout),
            ("posterior_dropout", self.posterior_dropout),
        ):
            if not 0.0 <= value < 1.0:
                raise ValueError(f"{name} must be in [0, 1)")
        if self.posterior_dropout and not self.posterior_dropout_justification:
            raise ValueError(
                "posterior_dropout requires a recorded scientific justification"
            )
        if self.text_dim != TEXT_DIM or self.ae_latent_dim != AE_LATENT_DIM:
            raise ValueError("Stage 4 probabilistic generation requires dimensions 768 -> 384")


class LatentStandardization(nn.Module):
    """Train-split-only per-dimension standardization with an exact inverse."""

    FORMAT_VERSION = 1

    def __init__(self, mean: Tensor, scale: Tensor, *, epsilon: float = 1e-4) -> None:
        super().__init__()
        mean = torch.as_tensor(mean, dtype=torch.float32).flatten()
        scale = torch.as_tensor(scale, dtype=torch.float32).flatten()
        if mean.shape != (AE_LATENT_DIM,) or scale.shape != mean.shape:
            raise ValueError("mean and scale must each contain 384 values")
        if not math.isfinite(epsilon) or epsilon <= 0:
            raise ValueError("epsilon must be finite and positive")
        if not bool(torch.isfinite(mean).all() and torch.isfinite(scale).all()):
            raise ValueError("standardization statistics must be finite")
        if bool((scale <= 0).any()):
            raise ValueError("standardization scale must be positive")
        self.epsilon = float(epsilon)
        self.register_buffer("mean", mean)
        self.register_buffer("scale", scale)

    @classmethod
    def fit(cls, training_latents: Tensor, *, epsilon: float = 1e-4) -> "LatentStandardization":
        """Fit only the explicitly supplied ordered training latents."""

        values = torch.as_tensor(training_latents).detach().to(torch.float64).cpu()
        if values.ndim != 2 or values.shape[0] < 2 or values.shape[1] != AE_LATENT_DIM:
            raise ValueError("training_latents must have shape N x 384 with N >= 2")
        if not bool(torch.isfinite(values).all()):
            raise ValueError("training_latents contain non-finite values")
        mean = values.mean(dim=0)
        scale = (values - mean).square().mean(dim=0).sqrt().clamp_min(epsilon)
        return cls(mean.float(), scale.float(), epsilon=epsilon)

    def transform(self, raw_latent: Tensor) -> Tensor:
        if raw_latent.shape[-1] != AE_LATENT_DIM:
            raise ValueError("raw Stage 1 AE latent must have 384 values")
        return (raw_latent - self.mean) / self.scale

    def inverse(self, standardized_latent: Tensor) -> Tensor:
        if standardized_latent.shape[-1] != AE_LATENT_DIM:
            raise ValueError("standardized Stage 1 AE latent must have 384 values")
        return standardized_latent * self.scale + self.mean

    def forward(self, raw_latent: Tensor) -> Tensor:
        return self.transform(raw_latent)

    def metadata(self) -> dict[str, Any]:
        return {
            "format_version": self.FORMAT_VERSION,
            "kind": "per_dimension_population_standardization",
            "latent_dim": AE_LATENT_DIM,
            "epsilon": self.epsilon,
            "fit_split": "train_only",
            "decoder_input": "exact_inverse_to_raw_stage1_ae_latent",
            "state_sha256": sha256_state_dict(self),
        }

    def to_payload(self) -> dict[str, Any]:
        return {"metadata": self.metadata(), "state_dict": self.state_dict()}

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "LatentStandardization":
        metadata = payload.get("metadata")
        state = payload.get("state_dict")
        if not isinstance(metadata, Mapping) or not isinstance(state, Mapping):
            raise TypeError("standardization payload requires metadata and state_dict")
        result = cls(
            state["mean"],
            state["scale"],
            epsilon=float(metadata["epsilon"]),
        )
        result.load_state_dict(state, strict=True)
        expected = metadata.get("state_sha256")
        if expected and sha256_state_dict(result) != expected:
            raise ValueError("latent standardization state SHA256 mismatch")
        return result


class Residual1024Block(nn.Module):
    """A pre-normalized 1024-wide residual MLP block."""

    def __init__(self) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(1024)
        self.fc1 = nn.Linear(1024, 1024)
        self.activation = nn.GELU()
        self.fc2 = nn.Linear(1024, 1024)

    def forward(self, value: Tensor) -> Tensor:
        branch = self.fc2(self.activation(self.fc1(self.norm(value))))
        return value + branch


def _normal_sample(
    shape: tuple[int, ...],
    *,
    reference: Tensor,
    generator: torch.Generator | None = None,
) -> Tensor:
    return torch.randn(
        shape,
        dtype=reference.dtype,
        device=reference.device,
        generator=generator,
    )


class ConditionalLatentVAE(nn.Module):
    """Conditional VAE operating in standardized Stage 1 AE latent space."""

    def __init__(self, config: ConditionalVAEConfig | None = None) -> None:
        super().__init__()
        self.config = config or ConditionalVAEConfig()
        posterior_layers: list[nn.Module] = [
            nn.Linear(POSTERIOR_INPUT_DIM, 1024),
            nn.GELU(),
            nn.Linear(1024, 512),
            nn.GELU(),
        ]
        if self.config.posterior_dropout:
            posterior_layers.append(nn.Dropout(self.config.posterior_dropout))
        self.posterior = nn.Sequential(*posterior_layers)
        self.posterior_mu = nn.Linear(512, self.config.u_dim)
        self.posterior_logvar = nn.Linear(512, self.config.u_dim)
        self.generator_input = nn.Linear(TEXT_DIM + self.config.u_dim, 1024)
        self.generator_activation = nn.GELU()
        self.generator_residual = Residual1024Block()
        self.generator_output = nn.Linear(1024, AE_LATENT_DIM)

    def _condition(self, condition: Tensor, *, apply_dropout: bool) -> Tensor:
        if condition.ndim != 2 or condition.shape[1] != TEXT_DIM:
            raise ValueError("condition must have shape B x 768")
        condition = F.normalize(condition, dim=-1, eps=1e-8)
        if apply_dropout and self.training and self.config.condition_dropout:
            keep = torch.rand(
                (len(condition), 1),
                device=condition.device,
                dtype=condition.dtype,
            ) >= self.config.condition_dropout
            condition = condition * keep
        return condition

    def encode_posterior(
        self,
        condition: Tensor,
        standardized_target: Tensor,
    ) -> tuple[Tensor, Tensor]:
        condition = self._condition(condition, apply_dropout=True)
        return self._encode_posterior_from_normalized(condition, standardized_target)

    def _encode_posterior_from_normalized(
        self,
        condition: Tensor,
        standardized_target: Tensor,
    ) -> tuple[Tensor, Tensor]:
        if standardized_target.ndim != 2 or standardized_target.shape[1] != AE_LATENT_DIM:
            raise ValueError("standardized_target must have shape B x 384")
        hidden = self.posterior(torch.cat((condition, standardized_target), dim=-1))
        return self.posterior_mu(hidden), self.posterior_logvar(hidden).clamp(-20.0, 20.0)

    @staticmethod
    def reparameterize(
        mu: Tensor,
        logvar: Tensor,
        *,
        generator: torch.Generator | None = None,
    ) -> Tensor:
        if mu.shape != logvar.shape:
            raise ValueError("mu and logvar must have identical shapes")
        epsilon = _normal_sample(tuple(mu.shape), reference=mu, generator=generator)
        return mu + torch.exp(0.5 * logvar) * epsilon

    def decode_standardized(
        self,
        condition: Tensor,
        u: Tensor,
        *,
        apply_condition_dropout: bool = False,
    ) -> Tensor:
        condition = self._condition(condition, apply_dropout=apply_condition_dropout)
        if u.ndim != 2 or u.shape != (len(condition), self.config.u_dim):
            raise ValueError(f"u must have shape B x {self.config.u_dim}")
        hidden = self.generator_activation(
            self.generator_input(torch.cat((condition, u), dim=-1))
        )
        return self.generator_output(self.generator_residual(hidden))

    def forward(
        self,
        condition: Tensor,
        standardized_target: Tensor,
        *,
        generator: torch.Generator | None = None,
    ) -> dict[str, Tensor]:
        condition = self._condition(condition, apply_dropout=True)
        mu, logvar = self._encode_posterior_from_normalized(
            condition, standardized_target
        )
        u = self.reparameterize(mu, logvar, generator=generator)
        prediction = self.decode_standardized(condition, u)
        return {"standardized_prediction": prediction, "mu": mu, "logvar": logvar, "u": u}

    def mean_path(
        self,
        condition: Tensor,
        *,
        standardized_target: Tensor | None = None,
        mode: Literal["zero", "posterior_mean"] = "zero",
    ) -> Tensor:
        if mode == "zero":
            u = condition.new_zeros((len(condition), self.config.u_dim))
        elif mode == "posterior_mean":
            if standardized_target is None:
                raise ValueError("posterior_mean requires standardized_target")
            u, _ = self.encode_posterior(condition, standardized_target)
        else:
            raise ValueError("mean path mode must be 'zero' or 'posterior_mean'")
        return self.decode_standardized(condition, u)

    def sample_prior(
        self,
        condition: Tensor,
        *,
        k: int,
        seed: int | None = None,
        generator: torch.Generator | None = None,
    ) -> Tensor:
        """Return B x K x 384 standardized text-only prior samples."""

        if k < 1:
            raise ValueError("k must be positive")
        if seed is not None and generator is not None:
            raise ValueError("pass seed or generator, not both")
        if seed is not None:
            generator = torch.Generator(device=condition.device).manual_seed(seed)
        batch = len(condition)
        u = _normal_sample(
            (batch, k, self.config.u_dim),
            reference=condition,
            generator=generator,
        )
        repeated_condition = condition[:, None, :].expand(-1, k, -1)
        decoded = self.decode_standardized(
            repeated_condition.reshape(batch * k, TEXT_DIM),
            u.reshape(batch * k, self.config.u_dim),
        )
        return decoded.reshape(batch, k, AE_LATENT_DIM)

    def sample_posterior(
        self,
        condition: Tensor,
        standardized_target: Tensor,
        *,
        k: int,
        seed: int | None = None,
    ) -> Tensor:
        """Return B x K x 384 posterior reconstruction samples."""

        if k < 1:
            raise ValueError("k must be positive")
        generator = (
            None
            if seed is None
            else torch.Generator(device=condition.device).manual_seed(seed)
        )
        mu, logvar = self.encode_posterior(condition, standardized_target)
        epsilon = _normal_sample(
            (len(condition), k, self.config.u_dim),
            reference=condition,
            generator=generator,
        )
        u = mu[:, None, :] + torch.exp(0.5 * logvar)[:, None, :] * epsilon
        repeated_condition = condition[:, None, :].expand(-1, k, -1)
        decoded = self.decode_standardized(
            repeated_condition.reshape(-1, TEXT_DIM),
            u.reshape(-1, self.config.u_dim),
        )
        return decoded.reshape(len(condition), k, AE_LATENT_DIM)


@dataclass(frozen=True)
class CVAELoss:
    total: Tensor
    standardized_latent_mse: Tensor
    true_kl: Tensor
    optimized_kl: Tensor
    decoded_volume_mse: Tensor
    foreground_mse: Tensor
    latent_cosine_loss: Tensor
    beta: float


def kl_per_dimension(mu: Tensor, logvar: Tensor) -> Tensor:
    """Return KL[q(u|x)||N(0,I)] for every example and stochastic dimension."""

    if mu.shape != logvar.shape:
        raise ValueError("mu and logvar must have identical shapes")
    return 0.5 * (mu.square() + logvar.exp() - 1.0 - logvar)


def annealed_beta(
    step: int,
    *,
    beta_max: float,
    warmup_steps: int,
    schedule: Literal["linear", "cyclical"] = "linear",
    cycle_steps: int | None = None,
) -> float:
    """Compute linear-warmup or cyclical KL weight without using validation/test."""

    if step < 0 or beta_max < 0 or warmup_steps < 0:
        raise ValueError("step, beta_max, and warmup_steps must be non-negative")
    if schedule == "linear":
        fraction = 1.0 if warmup_steps == 0 else min(1.0, step / warmup_steps)
    elif schedule == "cyclical":
        if cycle_steps is None or cycle_steps < 1:
            raise ValueError("cyclical annealing requires cycle_steps >= 1")
        within_cycle = step % cycle_steps
        ramp = warmup_steps or cycle_steps
        fraction = min(1.0, within_cycle / max(1, min(ramp, cycle_steps)))
    else:
        raise ValueError("schedule must be 'linear' or 'cyclical'")
    return float(beta_max * fraction)


def compute_cvae_loss(
    output: Mapping[str, Tensor],
    standardized_target: Tensor,
    target_volume: Tensor,
    *,
    standardization: LatentStandardization,
    decoder: nn.Module,
    beta: float,
    free_bits_per_dim: float = 0.0,
    w_latent: float = 1.0,
    w_image: float = 1.0,
    w_fg: float = 0.0,
    w_cos: float = 0.0,
) -> CVAELoss:
    """Compute the requested loss and exactly invert before frozen decoding."""

    for name, value in (
        ("beta", beta),
        ("free_bits_per_dim", free_bits_per_dim),
        ("w_latent", w_latent),
        ("w_image", w_image),
        ("w_fg", w_fg),
        ("w_cos", w_cos),
    ):
        if value < 0:
            raise ValueError(f"{name} must be non-negative")
    prediction = output["standardized_prediction"]
    mu, logvar = output["mu"], output["logvar"]
    if prediction.shape != standardized_target.shape:
        raise ValueError("prediction and standardized target shapes differ")
    raw_prediction = standardization.inverse(prediction)
    prediction_volume = decoder(raw_prediction)
    latent_mse = F.mse_loss(prediction.float(), standardized_target.float())
    kl_dims = kl_per_dimension(mu.float(), logvar.float())
    true_kl = kl_dims.sum(dim=1).mean()
    optimized_kl = kl_dims.mean(dim=0).clamp_min(free_bits_per_dim).sum()
    decoded_mse = F.mse_loss(prediction_volume.float(), target_volume.float())
    foreground = target_volume.float() > 0
    foreground_mse = (
        (prediction_volume.float() - target_volume.float()).square()[foreground].mean()
        if bool(foreground.any())
        else decoded_mse
    )
    cosine_loss = 1.0 - F.cosine_similarity(
        prediction.float(), standardized_target.float(), dim=-1, eps=1e-8
    ).mean()
    total = (
        w_latent * latent_mse
        + beta * optimized_kl
        + w_image * decoded_mse
        + w_fg * foreground_mse
        + w_cos * cosine_loss
    )
    return CVAELoss(
        total=total,
        standardized_latent_mse=latent_mse,
        true_kl=true_kl,
        optimized_kl=optimized_kl,
        decoded_volume_mse=decoded_mse,
        foreground_mse=foreground_mse,
        latent_cosine_loss=cosine_loss,
        beta=float(beta),
    )


@torch.no_grad()
def posterior_diagnostics(
    mu: Tensor,
    logvar: Tensor,
    *,
    active_threshold: float = ACTIVE_DIM_KL_THRESHOLD,
) -> dict[str, Any]:
    """Summarize posterior collapse and a moment-matched MI proxy."""

    mu = torch.as_tensor(mu).detach().float()
    logvar = torch.as_tensor(logvar).detach().float()
    if mu.ndim != 2 or mu.shape != logvar.shape or len(mu) < 2:
        raise ValueError("mu and logvar must be matching N x U tensors with N >= 2")
    kl_dims = kl_per_dimension(mu, logvar)
    mean_kl_dims = kl_dims.mean(dim=0)
    posterior_mean_variance = mu.var(dim=0, unbiased=False)
    posterior_std = torch.exp(0.5 * logvar)
    aggregate_mean = mu.mean(dim=0)
    aggregate_variance = (
        logvar.exp() + mu.square()
    ).mean(dim=0) - aggregate_mean.square()
    aggregate_variance = aggregate_variance.clamp_min(1e-8)
    aggregate_kl = 0.5 * (
        aggregate_mean.square()
        + aggregate_variance
        - 1.0
        - aggregate_variance.log()
    ).sum()
    expected_conditional_kl = kl_dims.sum(dim=1).mean()
    mi_proxy = (expected_conditional_kl - aggregate_kl).clamp_min(0.0)
    return {
        "mean_kl": float(expected_conditional_kl),
        "mean_kl_per_dimension": float(mean_kl_dims.mean()),
        "active_latent_dimensions": int((mean_kl_dims > active_threshold).sum()),
        "active_dimension_threshold": float(active_threshold),
        "posterior_mean_variance": float(posterior_mean_variance.mean()),
        "posterior_standard_deviation": float(posterior_std.mean()),
        "aggregate_posterior_kl": float(aggregate_kl),
        "mutual_information_proxy": float(mi_proxy),
        "per_dimension_kl": mean_kl_dims.cpu().tolist(),
        "per_dimension_posterior_mean_variance": posterior_mean_variance.cpu().tolist(),
    }


def pairwise_sample_distances(samples: Tensor) -> Tensor:
    """Return B x K x K Euclidean distances for B x K x D samples."""

    samples = torch.as_tensor(samples)
    if samples.ndim != 3 or samples.shape[1] < 1:
        raise ValueError("samples must have shape B x K x D")
    return torch.cdist(samples.float(), samples.float())


def medoid_indices(samples: Tensor) -> Tensor:
    """Return the consensus/medoid sample index for every batch item."""

    return pairwise_sample_distances(samples).mean(dim=-1).argmin(dim=-1)


def gather_samples(samples: Tensor, indices: Tensor) -> Tensor:
    """Gather one B x D sample from a B x K x D sample tensor."""

    if samples.ndim != 3 or indices.shape != (samples.shape[0],):
        raise ValueError("expected samples B x K x D and indices B")
    batch = torch.arange(samples.shape[0], device=samples.device)
    return samples[batch, indices]


def sample_interval_coverage(
    samples: Tensor,
    target: Tensor,
    *,
    levels: tuple[float, ...] = (0.5, 0.8, 0.9, 0.95),
) -> dict[str, float]:
    """Marginal sample-quantile coverage of target latent coordinates."""

    if samples.ndim != 3 or target.shape != (samples.shape[0], samples.shape[2]):
        raise ValueError("expected samples B x K x D and target B x D")
    output: dict[str, float] = {}
    for level in levels:
        if not 0.0 < level < 1.0:
            raise ValueError("coverage levels must be in (0, 1)")
        tail = (1.0 - level) / 2.0
        lower = torch.quantile(samples.float(), tail, dim=1)
        upper = torch.quantile(samples.float(), 1.0 - tail, dim=1)
        output[f"coverage_{int(round(level * 100))}"] = float(
            ((target >= lower) & (target <= upper)).float().mean()
        )
    return output


def architecture_record(model: ConditionalLatentVAE) -> dict[str, Any]:
    """Return a stable architecture and parameter-count record."""

    return {
        "namespace": "neurovlm.experiments.stage4_probabilistic",
        "name": "ConditionalLatentVAE",
        "config": asdict(model.config),
        "posterior": [
            1152,
            1024,
            "GELU",
            512,
            "GELU",
            {"heads": ["mu_u", "logvar_u"], "dimension": model.config.u_dim},
        ],
        "generator": [
            768 + model.config.u_dim,
            1024,
            "GELU",
            "residual_1024_block",
            384,
        ],
        "output": "standardized_stage1_ae_latent",
        "decoder_input": "exact_inverse_standardized_raw_stage1_ae_latent",
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
    }


def provenance_binding(provenance: Mapping[str, Any]) -> dict[str, Any]:
    """Extract and validate the immutable identities bound into checkpoints."""

    required = ("autoencoder", "text_cache", "splits", "branch")
    missing = [name for name in required if name not in provenance]
    if missing:
        raise ValueError(f"provenance missing required identities: {missing}")
    splits = provenance["splits"]
    if not isinstance(splits, Mapping) or set(("train", "val", "test")) - set(splits):
        raise ValueError("provenance must bind train, val, and test splits")
    optional = ("run_identity", "latent_standardization")
    return {
        name: provenance[name]
        for name in (*required, *optional)
        if name in provenance
    }


def validate_provenance(
    recorded: Mapping[str, Any],
    expected: Mapping[str, Any],
) -> None:
    """Reject checkpoint/data reuse when any immutable identity differs."""

    recorded_binding = provenance_binding(recorded)
    expected_binding = provenance_binding(expected)
    if sha256_value(recorded_binding) != sha256_value(expected_binding):
        raise ValueError("conditional Stage 4 provenance binding mismatch")


def checkpoint_payload(
    model: ConditionalLatentVAE,
    *,
    standardization: LatentStandardization,
    provenance: Mapping[str, Any],
    epoch: int,
    global_step: int,
    metrics: Mapping[str, Any],
    optimizer: torch.optim.Optimizer | None = None,
    scaler: Any | None = None,
) -> dict[str, Any]:
    """Build a strict, resume-capable conditional Stage 4 checkpoint."""

    binding = provenance_binding(provenance)
    payload: dict[str, Any] = {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "architecture": architecture_record(model),
        "model_config": asdict(model.config),
        "model_state_dict": model.state_dict(),
        "model_state_sha256": sha256_state_dict(model),
        "latent_standardization": standardization.to_payload(),
        "provenance": dict(provenance),
        "provenance_binding_sha256": sha256_value(binding),
        "epoch": int(epoch),
        "global_step": int(global_step),
        "metrics": dict(metrics),
    }
    if optimizer is not None:
        payload["optimizer_state_dict"] = optimizer.state_dict()
    if scaler is not None:
        payload["scaler_state_dict"] = scaler.state_dict()
    return payload


def load_checkpoint(
    path: str | Path,
    *,
    expected_provenance: Mapping[str, Any],
    device: str | torch.device = "cpu",
    optimizer: torch.optim.Optimizer | None = None,
    scaler: Any | None = None,
) -> tuple[ConditionalLatentVAE, LatentStandardization, dict[str, Any]]:
    """Reload a checkpoint and validate model, standardization, and provenance."""

    payload = torch.load(path, map_location=device, weights_only=False)
    if payload.get("format_version") != CHECKPOINT_FORMAT_VERSION:
        raise ValueError("unsupported conditional Stage 4 checkpoint format")
    validate_provenance(payload["provenance"], expected_provenance)
    if payload.get("provenance_binding_sha256") != sha256_value(
        provenance_binding(expected_provenance)
    ):
        raise ValueError("checkpoint provenance digest mismatch")
    model = ConditionalLatentVAE(ConditionalVAEConfig(**payload["model_config"]))
    model.load_state_dict(payload["model_state_dict"], strict=True)
    if sha256_state_dict(model) != payload.get("model_state_sha256"):
        raise ValueError("checkpoint model state SHA256 mismatch")
    standardization = LatentStandardization.from_payload(
        payload["latent_standardization"]
    )
    if optimizer is not None:
        optimizer.load_state_dict(payload["optimizer_state_dict"])
    if scaler is not None and "scaler_state_dict" in payload:
        scaler.load_state_dict(payload["scaler_state_dict"])
    return model.to(device), standardization.to(device), payload


__all__ = [
    "ACTIVE_DIM_KL_THRESHOLD",
    "AE_LATENT_DIM",
    "CHECKPOINT_FORMAT_VERSION",
    "CVAELoss",
    "ConditionalLatentVAE",
    "ConditionalVAEConfig",
    "LatentStandardization",
    "SUPPORTED_U_DIMS",
    "TEXT_DIM",
    "annealed_beta",
    "architecture_record",
    "checkpoint_payload",
    "compute_cvae_loss",
    "gather_samples",
    "kl_per_dimension",
    "load_checkpoint",
    "medoid_indices",
    "pairwise_sample_distances",
    "posterior_diagnostics",
    "provenance_binding",
    "sample_interval_coverage",
    "validate_provenance",
]
