"""Experimental Stage 4 projector architectures and sweep utilities.

This module is intentionally opt-in.  The retained production
``GenerativeTextToAELatent`` implementation and its checkpoint loader are not
changed.  Every projector built here maps a 768-dimensional cached text
embedding directly to an unconstrained 384-dimensional raw Stage 1
autoencoder latent.
"""

from __future__ import annotations

import copy
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any, Literal

import torch
from torch import Tensor, nn

from neurovlm.cnn import GenerativeTextToAELatent
from neurovlm.pipelines import sha256_state_dict, sha256_value


PROJECTOR_FORMAT_VERSION = 1
PROJECTOR_INPUT_DIM = 768
PROJECTOR_OUTPUT_DIM = 384
PROJECTOR_NAMES = (
    "retained_mlp",
    "wider_mlp",
    "deep_mlp",
    "layernorm_deep",
    "residual_1024",
    "residual_2048",
    "gated_residual",
)
SchedulerName = Literal[
    "constant",
    "warmup_cosine",
    "reduce_on_plateau",
    "cosine_restarts",
]


class ResidualMLPBlock(nn.Module):
    """Pre-normalized residual MLP block with an identity skip."""

    def __init__(self, width: int, dropout: float) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.fc1 = nn.Linear(width, width)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(width, width)

    def forward(self, value: Tensor) -> Tensor:
        branch = self.fc2(self.dropout(self.activation(self.fc1(self.norm(value)))))
        return value + branch


class GatedResidualMLPBlock(nn.Module):
    """Residual block with a learned feature-wise sigmoid branch gate."""

    def __init__(self, width: int, dropout: float) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.fc1 = nn.Linear(width, width)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(width, width)
        # A negative initialization starts close to the identity while still
        # allowing every gate to learn independently.
        self.gate_logits = nn.Parameter(torch.full((width,), -2.0))

    def forward(self, value: Tensor) -> Tensor:
        branch = self.fc2(self.dropout(self.activation(self.fc1(self.norm(value)))))
        return value + torch.sigmoid(self.gate_logits) * branch


class ResidualProjector(nn.Module):
    """Experimental residual text-to-raw-latent projector."""

    def __init__(
        self,
        *,
        width: int,
        blocks: int,
        dropout: float,
        gated: bool = False,
    ) -> None:
        super().__init__()
        block_type = GatedResidualMLPBlock if gated else ResidualMLPBlock
        self.input_norm = nn.LayerNorm(PROJECTOR_INPUT_DIM)
        self.input_projection = nn.Linear(PROJECTOR_INPUT_DIM, width)
        self.blocks = nn.ModuleList(
            block_type(width, dropout) for _ in range(blocks)
        )
        self.output_norm = nn.LayerNorm(width)
        self.output_projection = nn.Linear(width, PROJECTOR_OUTPUT_DIM)

    def forward(self, value: Tensor) -> Tensor:
        value = self.input_projection(self.input_norm(value))
        for block in self.blocks:
            value = block(value)
        # The final Linear has deliberately no activation, normalization,
        # clipping, sigmoid, or tanh after it.
        return self.output_projection(self.output_norm(value))


@dataclass(frozen=True)
class ProjectorBuildConfig:
    """Identity-bearing arguments for one experimental projector."""

    name: str
    dropout: float = 0.0
    residual_2048_blocks: int = 2
    input_dim: int = PROJECTOR_INPUT_DIM
    output_dim: int = PROJECTOR_OUTPUT_DIM

    def __post_init__(self) -> None:
        if self.name not in PROJECTOR_NAMES:
            raise ValueError(
                f"Unknown projector {self.name!r}; expected one of {PROJECTOR_NAMES}"
            )
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if self.residual_2048_blocks not in {1, 2}:
            raise ValueError("residual_2048_blocks must be 1 or 2")
        if self.input_dim != PROJECTOR_INPUT_DIM:
            raise ValueError("Stage 4 experimental projectors require input_dim=768")
        if self.output_dim != PROJECTOR_OUTPUT_DIM:
            raise ValueError("Stage 4 experimental projectors require output_dim=384")


def projector_definition(config: ProjectorBuildConfig) -> dict[str, Any]:
    """Return a stable, serializable architecture definition."""

    name = config.name
    if name == "retained_mlp":
        layers: list[Any] = [768, 512, "ReLU", 384]
    elif name == "wider_mlp":
        layers = [768, 1024, "GELU", 384]
    elif name == "deep_mlp":
        layers = [768, 1024, "GELU", 1024, "GELU", 384]
    elif name == "layernorm_deep":
        layers = [
            "LayerNorm(768)",
            768,
            1024,
            "GELU",
            "LayerNorm(1024)",
            1024,
            1024,
            "GELU",
            384,
        ]
    else:
        width = 2048 if name == "residual_2048" else 1024
        blocks = config.residual_2048_blocks if name == "residual_2048" else 2
        layers = [
            "LayerNorm(768)",
            f"Linear(768,{width})",
            *(
                [
                    {
                        "block": "gated_residual_mlp"
                        if name == "gated_residual"
                        else "residual_mlp",
                        "width": width,
                        "pre_norm": True,
                        "activation": "GELU",
                        "dropout": config.dropout,
                        "gate": "learned_featurewise_sigmoid"
                        if name == "gated_residual"
                        else None,
                    }
                    for _ in range(blocks)
                ]
            ),
            f"LayerNorm({width})",
            f"Linear({width},384)",
        ]
    return {
        "format_version": PROJECTOR_FORMAT_VERSION,
        "namespace": "neurovlm.experiments",
        "name": name,
        "input_dim": PROJECTOR_INPUT_DIM,
        "output_dim": PROJECTOR_OUTPUT_DIM,
        "dropout": config.dropout,
        "residual_2048_blocks": config.residual_2048_blocks,
        "layers": layers,
        "decoder_input": "raw_384d_stage1_ae_latent",
        "final_output_transform": None,
    }


def build_stage4_projector(
    name: str,
    *,
    dropout: float = 0.0,
    residual_2048_blocks: int = 2,
) -> nn.Module:
    """Build a named projector without altering production model defaults."""

    config = ProjectorBuildConfig(
        name=name,
        dropout=dropout,
        residual_2048_blocks=residual_2048_blocks,
    )
    if name == "retained_mlp":
        return GenerativeTextToAELatent(768, 512, 384)
    if name == "wider_mlp":
        return nn.Sequential(
            nn.Linear(768, 1024),
            nn.GELU(),
            nn.Linear(1024, 384),
        )
    if name == "deep_mlp":
        return nn.Sequential(
            nn.Linear(768, 1024),
            nn.GELU(),
            nn.Linear(1024, 1024),
            nn.GELU(),
            nn.Linear(1024, 384),
        )
    if name == "layernorm_deep":
        return nn.Sequential(
            nn.LayerNorm(768),
            nn.Linear(768, 1024),
            nn.GELU(),
            nn.LayerNorm(1024),
            nn.Linear(1024, 1024),
            nn.GELU(),
            nn.Linear(1024, 384),
        )
    if name == "residual_1024":
        return ResidualProjector(width=1024, blocks=2, dropout=dropout)
    if name == "residual_2048":
        return ResidualProjector(
            width=2048,
            blocks=residual_2048_blocks,
            dropout=dropout,
        )
    if name == "gated_residual":
        return ResidualProjector(
            width=1024,
            blocks=2,
            dropout=dropout,
            gated=True,
        )
    raise AssertionError("ProjectorBuildConfig accepted an unhandled architecture")


def count_parameters(module: nn.Module, *, trainable_only: bool = True) -> int:
    """Count scalar parameters."""

    return sum(
        parameter.numel()
        for parameter in module.parameters()
        if parameter.requires_grad or not trainable_only
    )


def _activation_widths(config: ProjectorBuildConfig) -> tuple[int, ...]:
    """Major saved activation widths used for a transparent memory estimate."""

    if config.name == "retained_mlp":
        return (512, 512, 384)
    if config.name == "wider_mlp":
        return (1024, 1024, 384)
    if config.name == "deep_mlp":
        return (1024, 1024, 1024, 1024, 384)
    if config.name == "layernorm_deep":
        return (768, 1024, 1024, 1024, 1024, 1024, 384)
    width = 2048 if config.name == "residual_2048" else 1024
    blocks = config.residual_2048_blocks if config.name == "residual_2048" else 2
    # Input norm/projection, five tensors per residual block (norm, two
    # linears, activation/dropout), output norm, and output projection.
    return (768, width, *(width for _ in range(5 * blocks)), width, 384)


def estimate_activation_memory(
    config: ProjectorBuildConfig,
    *,
    batch_size: int,
    dtype: torch.dtype = torch.bfloat16,
) -> dict[str, int | str]:
    """Estimate projector activation memory, excluding frozen AE activations.

    ``forward_bytes`` is the sum of major layer output tensors.  The
    ``training_saved_bytes_estimate`` uses a documented 3x multiplier for
    autograd-saved tensors and gradients; measured peak CUDA memory remains the
    authoritative run-level diagnostic.
    """

    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    element_size = torch.empty((), dtype=dtype).element_size()
    forward_bytes = batch_size * sum(_activation_widths(config)) * element_size
    return {
        "dtype": str(dtype),
        "batch_size": batch_size,
        "forward_bytes": forward_bytes,
        "training_saved_bytes_estimate": 3 * forward_bytes,
        "scope": "projector_major_layer_outputs_only",
    }


def architecture_record(
    config: ProjectorBuildConfig,
    module: nn.Module,
    *,
    batch_size: int,
    dtype: torch.dtype,
) -> dict[str, Any]:
    """Combine definition, parameter count, and memory estimate."""

    definition = projector_definition(config)
    return {
        **definition,
        "definition_sha256": sha256_value(definition),
        "parameter_count": count_parameters(module),
        "activation_memory": estimate_activation_memory(
            config,
            batch_size=batch_size,
            dtype=dtype,
        ),
    }


class ActivationMonitor:
    """Aggregate forward activation diagnostics for major projector layers."""

    def __init__(self, module: nn.Module) -> None:
        self.rows: dict[str, dict[str, float]] = {}
        self._handles: list[Any] = []
        leaf_types = (nn.Linear, nn.LayerNorm, nn.ReLU, nn.GELU)
        for name, child in module.named_modules():
            if name and isinstance(child, leaf_types):
                self._handles.append(
                    child.register_forward_hook(self._hook(name, child))
                )

    def _hook(self, name: str, child: nn.Module):
        def record(_module: nn.Module, _inputs: Any, output: Any) -> None:
            if not torch.is_tensor(output):
                return
            value = output.detach().float()
            finite = torch.isfinite(value)
            safe = value[finite]
            row = self.rows.setdefault(
                name,
                {
                    "batches": 0.0,
                    "mean_sum": 0.0,
                    "std_sum": 0.0,
                    "zero_fraction_sum": 0.0,
                    "nonfinite_fraction_sum": 0.0,
                    "max_abs": 0.0,
                    "layer_type": child.__class__.__name__,
                },
            )
            row["batches"] += 1
            row["mean_sum"] += float(safe.mean()) if safe.numel() else math.nan
            row["std_sum"] += float(safe.std(unbiased=False)) if safe.numel() else math.nan
            row["zero_fraction_sum"] += float((value == 0).float().mean())
            row["nonfinite_fraction_sum"] += float((~finite).float().mean())
            if safe.numel():
                row["max_abs"] = max(row["max_abs"], float(safe.abs().max()))

        return record

    def summary(self) -> list[dict[str, Any]]:
        output = []
        for name, row in self.rows.items():
            batches = max(row["batches"], 1.0)
            output.append(
                {
                    "layer": name,
                    "layer_type": row["layer_type"],
                    "activation_mean": row["mean_sum"] / batches,
                    "activation_std": row["std_sum"] / batches,
                    "activation_zero_percent": 100.0
                    * row["zero_fraction_sum"]
                    / batches,
                    "activation_nonfinite_percent": 100.0
                    * row["nonfinite_fraction_sum"]
                    / batches,
                    "activation_max_abs": row["max_abs"],
                    "batches": int(row["batches"]),
                }
            )
        return output

    def close(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def __enter__(self) -> "ActivationMonitor":
        return self

    def __exit__(self, *_args: Any) -> None:
        self.close()


def gradient_diagnostics(module: nn.Module) -> tuple[dict[str, float], list[dict[str, Any]]]:
    """Return global norm and zero-gradient percentages by parameter tensor."""

    squared_norm = 0.0
    zero_values = 0
    total_values = 0
    rows = []
    for name, parameter in module.named_parameters():
        gradient = parameter.grad
        if gradient is None:
            row = {
                "layer": name,
                "gradient_norm": 0.0,
                "zero_gradient_percent": 100.0,
                "gradient_missing": True,
                "parameter_count": parameter.numel(),
            }
        else:
            value = gradient.detach().float()
            norm = float(value.norm())
            zero_count = int((value == 0).sum())
            squared_norm += norm * norm
            zero_values += zero_count
            total_values += value.numel()
            row = {
                "layer": name,
                "gradient_norm": norm,
                "zero_gradient_percent": 100.0 * zero_count / value.numel(),
                "gradient_missing": False,
                "parameter_count": parameter.numel(),
            }
        rows.append(row)
    return {
        "total_gradient_norm": math.sqrt(squared_norm),
        "zero_gradient_percent": 100.0 * zero_values / max(total_values, 1),
    }, rows


def clone_trainable_parameters(module: nn.Module) -> dict[str, Tensor]:
    """Take a detached parameter snapshot for update-norm diagnostics."""

    return {
        name: parameter.detach().clone()
        for name, parameter in module.named_parameters()
        if parameter.requires_grad
    }


def parameter_update_norm(
    module: nn.Module,
    before: Mapping[str, Tensor],
) -> float:
    """Compute the global parameter delta norm after an optimizer step."""

    squared = 0.0
    for name, parameter in module.named_parameters():
        if name in before:
            delta_norm = float((parameter.detach() - before[name]).float().norm())
            squared += delta_norm * delta_norm
    return math.sqrt(squared)


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    name: SchedulerName,
    *,
    epochs: int,
    warmup_epochs: int = 5,
    restart_period: int = 10,
) -> torch.optim.lr_scheduler.LRScheduler | torch.optim.lr_scheduler.ReduceLROnPlateau | None:
    """Build one of the controlled epoch-level learning-rate schedules."""

    if epochs < 1:
        raise ValueError("epochs must be positive")
    if name == "constant":
        return None
    if name == "warmup_cosine":
        if epochs <= warmup_epochs:
            raise ValueError("warmup_cosine requires epochs > warmup_epochs")
        warmup = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=1e-3,
            end_factor=1.0,
            total_iters=warmup_epochs,
        )
        cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=epochs - warmup_epochs,
        )
        return torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup, cosine],
            milestones=[warmup_epochs],
        )
    if name == "reduce_on_plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=3,
            threshold=1e-4,
            min_lr=1e-7,
        )
    if name == "cosine_restarts":
        return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=restart_period,
            T_mult=2,
        )
    raise ValueError(f"Unknown scheduler {name!r}")


def step_scheduler(
    scheduler: Any,
    name: SchedulerName,
    *,
    validation_loss: float,
) -> None:
    """Apply the correct epoch-level scheduler step."""

    if scheduler is None:
        return
    if name == "reduce_on_plateau":
        scheduler.step(validation_loss)
    else:
        scheduler.step()


def projector_checkpoint_metadata(
    config: ProjectorBuildConfig,
    module: nn.Module,
    *,
    binding: Mapping[str, Any],
    effective_config: Mapping[str, Any],
) -> dict[str, Any]:
    """Create strict metadata for an experimental projector checkpoint."""

    definition = projector_definition(config)
    return {
        "format_version": PROJECTOR_FORMAT_VERSION,
        "experimental_namespace": "neurovlm.experiments.stage4_projectors",
        "architecture": definition,
        "architecture_sha256": sha256_value(definition),
        "projector_state_sha256": sha256_state_dict(module),
        "binding": copy.deepcopy(dict(binding)),
        "binding_sha256": sha256_value(binding),
        "effective_config_sha256": sha256_value(effective_config),
        "production_projector_unchanged": True,
    }


def validate_projector_checkpoint_metadata(
    metadata: Mapping[str, Any],
    config: ProjectorBuildConfig,
    *,
    binding: Mapping[str, Any],
    effective_config: Mapping[str, Any],
    module: nn.Module | None = None,
) -> None:
    """Reject architecture, data/AE/cache, config, or state mismatches."""

    expected_definition = projector_definition(config)
    mismatches = []
    expected = {
        "format_version": PROJECTOR_FORMAT_VERSION,
        "experimental_namespace": "neurovlm.experiments.stage4_projectors",
        "architecture_sha256": sha256_value(expected_definition),
        "binding_sha256": sha256_value(binding),
        "effective_config_sha256": sha256_value(effective_config),
    }
    for key, value in expected.items():
        if metadata.get(key) != value:
            mismatches.append(key)
    if metadata.get("architecture") != expected_definition:
        mismatches.append("architecture")
    if module is not None:
        actual_state = sha256_state_dict(module)
        if metadata.get("projector_state_sha256") != actual_state:
            mismatches.append("projector_state_sha256")
    if mismatches:
        raise ValueError(
            "Experimental Stage 4 checkpoint metadata mismatch: "
            + ", ".join(sorted(set(mismatches)))
        )


def pareto_front(
    rows: Sequence[Mapping[str, Any]],
    objectives: Mapping[str, Literal["max", "min"]],
    *,
    epsilon: float = 0.0,
) -> list[bool]:
    """Return a mask for the non-dominated finite rows.

    A row is dominated when another row is at least as good on every objective
    (within ``epsilon``) and strictly better on at least one.
    """

    if epsilon < 0:
        raise ValueError("epsilon must be non-negative")
    if not objectives:
        raise ValueError("At least one Pareto objective is required")
    values: list[list[float] | None] = []
    for row in rows:
        converted = []
        valid = True
        for name, direction in objectives.items():
            if direction not in {"max", "min"}:
                raise ValueError(f"Invalid direction for {name!r}: {direction!r}")
            try:
                value = float(row[name])
            except (KeyError, TypeError, ValueError):
                valid = False
                break
            if not math.isfinite(value):
                valid = False
                break
            converted.append(value if direction == "max" else -value)
        values.append(converted if valid else None)
    mask = [value is not None for value in values]
    for index, candidate in enumerate(values):
        if candidate is None:
            continue
        for other_index, other in enumerate(values):
            if index == other_index or other is None:
                continue
            no_worse = all(
                other_value >= candidate_value - epsilon
                for other_value, candidate_value in zip(other, candidate, strict=True)
            )
            better = any(
                other_value > candidate_value + epsilon
                for other_value, candidate_value in zip(other, candidate, strict=True)
            )
            if no_worse and better:
                mask[index] = False
                break
    return mask


def detect_training_pathologies(
    metrics: Mapping[str, float],
    *,
    dead_activation_zero_percent: float = 99.0,
    near_zero_update_ratio: float = 1e-9,
    exploding_activation_abs: float = 1e4,
    output_norm_ratio_min: float = 0.10,
    divergence_ratio: float = 2.0,
) -> dict[str, bool]:
    """Convert logged diagnostics into explicit, configurable warning flags."""

    parameter_norm = max(float(metrics.get("parameter_norm", 0.0)), 1e-12)
    return {
        "dead_activations": float(
            metrics.get("max_layer_activation_zero_percent", 0.0)
        )
        >= dead_activation_zero_percent,
        "near_zero_updates": float(metrics.get("parameter_update_norm", 0.0))
        / parameter_norm
        <= near_zero_update_ratio,
        "exploding_activations": float(
            metrics.get("max_layer_activation_abs", 0.0)
        )
        >= exploding_activation_abs,
        "output_norm_collapse": float(metrics.get("latent_norm_ratio", 1.0))
        <= output_norm_ratio_min,
        "training_validation_divergence": float(
            metrics.get("validation_raw_latent_mse", 0.0)
        )
        > divergence_ratio
        * max(float(metrics.get("training_raw_latent_mse", 0.0)), 1e-12),
    }


__all__ = [
    "ActivationMonitor",
    "GatedResidualMLPBlock",
    "PROJECTOR_FORMAT_VERSION",
    "PROJECTOR_INPUT_DIM",
    "PROJECTOR_NAMES",
    "PROJECTOR_OUTPUT_DIM",
    "ProjectorBuildConfig",
    "ResidualMLPBlock",
    "ResidualProjector",
    "SchedulerName",
    "architecture_record",
    "build_scheduler",
    "build_stage4_projector",
    "clone_trainable_parameters",
    "count_parameters",
    "detect_training_pathologies",
    "estimate_activation_memory",
    "gradient_diagnostics",
    "parameter_update_norm",
    "pareto_front",
    "projector_checkpoint_metadata",
    "projector_definition",
    "step_scheduler",
    "validate_projector_checkpoint_metadata",
]
