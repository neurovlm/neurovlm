"""Versioned, model-aware run configuration without model loading."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Mapping
from uuid import uuid4

from neurovlm.model_registry import ModelSpec, resolve_model_spec


RUN_CONFIG_VERSION = 1


class MetricDirection(str, Enum):
    """Direction used to select the best checkpoint."""

    MIN = "min"
    MAX = "max"


def _run_id(spec: ModelSpec) -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    slug = "-".join(
        part for part in (spec.family.value, spec.task.value, spec.domain.value if spec.domain else None) if part
    )
    return f"{timestamp}-{slug}-{uuid4().hex[:8]}"


@dataclass(frozen=True)
class RunConfig:
    """Resolved configuration shared by all future training runners.

    ``requested`` retains user intent and ``effective`` records canonical
    values after defaults and model-registry resolution.  Neither resolution
    nor construction imports model code or accesses the network.
    """

    model_spec: ModelSpec
    output_root: Path
    run_id: str
    seed: int = 42
    device: str = "auto"
    primary_metric: str = "val_loss"
    metric_direction: MetricDirection = MetricDirection.MIN
    data: Mapping[str, Any] = field(default_factory=dict)
    resources: Mapping[str, Any] = field(default_factory=dict)
    initialization: Mapping[str, Any] = field(default_factory=dict)
    requested: Mapping[str, Any] = field(default_factory=dict)
    effective: Mapping[str, Any] = field(default_factory=dict)
    version: int = RUN_CONFIG_VERSION

    @classmethod
    def resolve(
        cls,
        *,
        task: str,
        family: str = "mlp",
        domain: str | None = None,
        variant: str | None = None,
        output_root: str | Path = "runs",
        run_id: str | None = None,
        seed: int = 42,
        device: str = "auto",
        primary_metric: str = "val_loss",
        metric_direction: MetricDirection | str = MetricDirection.MIN,
        data: Mapping[str, Any] | None = None,
        resources: Mapping[str, Any] | None = None,
        initialization: Mapping[str, Any] | None = None,
        requested: Mapping[str, Any] | None = None,
        effective: Mapping[str, Any] | None = None,
    ) -> "RunConfig":
        """Resolve a structured selection through the canonical registry."""

        spec = resolve_model_spec(
            family=family,
            task=task,
            domain=domain,
            variant=variant,
        )
        try:
            direction = MetricDirection(metric_direction)
        except ValueError as error:
            raise ValueError("metric_direction must be 'min' or 'max'") from error
        if not primary_metric:
            raise ValueError("primary_metric must not be empty")
        if int(seed) < 0:
            raise ValueError("seed must be non-negative")

        requested_values = {
            "family": family,
            "task": task,
            "domain": domain,
            "variant": variant,
            "device": device,
            **dict(requested or {}),
        }
        effective_values = {
            "family": spec.family.value,
            "task": spec.task.value,
            "domain": spec.domain.value if spec.domain else None,
            "variant": spec.variant.value,
            "model": spec.canonical_name,
            "device": device,
            **dict(effective or {}),
        }
        return cls(
            model_spec=spec,
            output_root=Path(output_root),
            run_id=run_id or _run_id(spec),
            seed=int(seed),
            device=str(device),
            primary_metric=primary_metric,
            metric_direction=direction,
            data=dict(data or {}),
            resources=dict(resources or {}),
            initialization=dict(initialization or {}),
            requested=requested_values,
            effective=effective_values,
        )

    @property
    def run_dir(self) -> Path:
        return self.output_root / self.run_id

    @property
    def task(self) -> str:
        return self.model_spec.task.value

    @property
    def family(self) -> str:
        return self.model_spec.family.value

    @property
    def domain(self) -> str | None:
        return self.model_spec.domain.value if self.model_spec.domain else None

    @property
    def variant(self) -> str:
        return self.model_spec.variant.value

    def model_spec_dict(self) -> dict[str, Any]:
        return {
            "canonical_name": self.model_spec.canonical_name,
            "family": self.family,
            "task": self.task,
            "domain": self.domain,
            "variant": self.variant,
            "loader": self.model_spec.loader.value,
            "loader_variant": self.model_spec.loader_variant,
        }

    def requested_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "run_id": self.run_id,
            "output_root": self.output_root,
            "seed": self.seed,
            "primary_metric": self.primary_metric,
            "metric_direction": self.metric_direction,
            "data": self.data,
            "resources": self.resources,
            "initialization": self.initialization,
            "values": self.requested,
        }

    def effective_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "run_id": self.run_id,
            "seed": self.seed,
            "device": self.device,
            "primary_metric": self.primary_metric,
            "metric_direction": self.metric_direction,
            "model_spec": self.model_spec_dict(),
            "data": self.data,
            "resources": self.resources,
            "initialization": self.initialization,
            "values": self.effective,
        }


__all__ = ["MetricDirection", "RUN_CONFIG_VERSION", "RunConfig"]
