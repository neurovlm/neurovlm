"""State-dict checkpoint selection, manifests, integrity, and safe resume."""

from __future__ import annotations

import json
import math
import os
import shutil
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping

import torch

from .config import MetricDirection, RunConfig
from .provenance import sha256_file
from .serialization import atomic_write_json, json_safe


CHECKPOINT_FORMAT_VERSION = 1


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _state_dict(value: Any) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    method = getattr(value, "state_dict", None)
    if callable(method):
        return method()
    raise TypeError("Checkpoint values must be state-dict mappings or expose state_dict()")


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


def _atomic_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        shutil.copyfile(source, temporary)
        os.replace(temporary, destination)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


@dataclass(frozen=True)
class ResumeState:
    """Metadata returned after a validated checkpoint resume."""

    path: Path
    epoch: int | None
    step: int | None
    metrics: Mapping[str, Any]
    payload: Mapping[str, Any]


class CheckpointManager:
    """Save best/last checkpoints and safely restore state dictionaries."""

    def __init__(
        self,
        config: RunConfig,
        *,
        checkpoint_dir: str | Path | None = None,
        expected_architecture: Mapping[str, Any] | None = None,
        validation_hook: Callable[[Mapping[str, Any]], None] | None = None,
    ):
        self.config = config
        self.checkpoint_dir = (
            Path(checkpoint_dir) if checkpoint_dir is not None else config.run_dir / "checkpoints"
        )
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        try:
            self.checkpoint_dir.resolve().relative_to(config.run_dir.resolve())
        except ValueError as error:
            raise ValueError("checkpoint_dir must be inside the configured run directory") from error
        self.manifest_path = self.checkpoint_dir / "checkpoint_manifest.json"
        self.expected_architecture = dict(expected_architecture or {})
        self.validation_hook = validation_hook

    def _relative(self, path: Path) -> str:
        return path.resolve().relative_to(self.config.run_dir.resolve()).as_posix()

    def _manifest(self) -> dict[str, Any]:
        if self.manifest_path.exists():
            manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
            if manifest.get("run_id") != self.config.run_id or manifest.get(
                "model_spec", {}
            ).get("canonical_name") != self.config.model_spec.canonical_name:
                raise ValueError("Checkpoint manifest is incompatible with the run configuration")
            return manifest
        return {
            "version": 1,
            "run_id": self.config.run_id,
            "model_spec": self.config.model_spec_dict(),
            "primary_metric": self.config.primary_metric,
            "metric_direction": self.config.metric_direction.value,
            "best_value": None,
            "checkpoints": [],
            "aliases": {},
        }

    def _write_manifest(self, manifest: Mapping[str, Any]) -> None:
        atomic_write_json(self.manifest_path, manifest)

    @property
    def best_value(self) -> float | None:
        """Return the selected best metric value, if a best checkpoint exists."""

        value = self._manifest().get("best_value")
        return None if value is None else float(value)

    def _payload(
        self,
        model: Any,
        *,
        epoch: int | None,
        step: int | None,
        metrics: Mapping[str, Any] | None,
        optimizer: Any | None,
        scheduler: Any | None,
        architecture: Mapping[str, Any] | None,
        extra: Mapping[str, Any] | None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "format_version": CHECKPOINT_FORMAT_VERSION,
            "run_id": self.config.run_id,
            "model_spec": self.config.model_spec_dict(),
            "architecture": dict(architecture or {}),
            "model_state_dict": _state_dict(model),
            "epoch": epoch,
            "step": step,
            "metrics": dict(metrics or {}),
        }
        if optimizer is not None:
            payload["optimizer_state_dict"] = _state_dict(optimizer)
        if scheduler is not None:
            payload["scheduler_state_dict"] = _state_dict(scheduler)
        if extra:
            reserved = set(payload).intersection(extra)
            if reserved:
                raise ValueError(f"extra cannot replace reserved checkpoint fields: {sorted(reserved)}")
            payload["extra"] = dict(extra)
        return payload

    def _save(
        self,
        name: str,
        role: str,
        model: Any,
        *,
        epoch: int | None,
        step: int | None,
        metrics: Mapping[str, Any] | None,
        metric_value: float | None,
        optimizer: Any | None,
        scheduler: Any | None,
        architecture: Mapping[str, Any] | None,
        extra: Mapping[str, Any] | None,
        aliases: tuple[str, ...],
    ) -> Path:
        if Path(name).name != name:
            raise ValueError("checkpoint names must be plain file names")
        path = self.checkpoint_dir / name
        payload = self._payload(
            model,
            epoch=epoch,
            step=step,
            metrics=metrics,
            optimizer=optimizer,
            scheduler=scheduler,
            architecture=architecture,
            extra=extra,
        )
        _atomic_torch_save(path, payload)
        record = {
            "role": role,
            "path": self._relative(path),
            "epoch": epoch,
            "step": step,
            "metric": self.config.primary_metric if metric_value is not None else None,
            "value": metric_value,
            "sha256": sha256_file(path),
            "size": path.stat().st_size,
            "saved_at": _utc_now(),
        }
        manifest = self._manifest()
        manifest["checkpoints"] = [
            item for item in manifest["checkpoints"] if item.get("role") != role
        ] + [record]
        if role == "best":
            manifest["best_value"] = metric_value
        for alias in aliases:
            if Path(alias).name != alias:
                raise ValueError("checkpoint aliases must be plain file names")
            alias_path = self.checkpoint_dir / alias
            _atomic_copy(path, alias_path)
            manifest["aliases"][self._relative(alias_path)] = {
                "target": self._relative(path),
                "sha256": sha256_file(alias_path),
            }
        self._write_manifest(manifest)
        return path

    def save_last(
        self,
        model: Any,
        *,
        epoch: int | None = None,
        step: int | None = None,
        metrics: Mapping[str, Any] | None = None,
        optimizer: Any | None = None,
        scheduler: Any | None = None,
        architecture: Mapping[str, Any] | None = None,
        extra: Mapping[str, Any] | None = None,
        aliases: tuple[str, ...] = (),
    ) -> Path:
        return self._save(
            "last.pt",
            "last",
            model,
            epoch=epoch,
            step=step,
            metrics=metrics,
            metric_value=None,
            optimizer=optimizer,
            scheduler=scheduler,
            architecture=architecture,
            extra=extra,
            aliases=aliases,
        )

    def save_best(
        self,
        model: Any,
        metric_value: float,
        *,
        epoch: int | None = None,
        step: int | None = None,
        metrics: Mapping[str, Any] | None = None,
        optimizer: Any | None = None,
        scheduler: Any | None = None,
        architecture: Mapping[str, Any] | None = None,
        extra: Mapping[str, Any] | None = None,
        aliases: tuple[str, ...] = (),
    ) -> Path | None:
        """Save ``best.pt`` only when *metric_value* improves."""

        value = float(metric_value)
        if not math.isfinite(value):
            raise ValueError("Best-checkpoint metric_value must be finite")
        current = self._manifest().get("best_value")
        improves = current is None or (
            value < float(current)
            if self.config.metric_direction is MetricDirection.MIN
            else value > float(current)
        )
        if not improves:
            return None
        combined_metrics = dict(metrics or {})
        combined_metrics.setdefault(self.config.primary_metric, value)
        return self._save(
            "best.pt",
            "best",
            model,
            epoch=epoch,
            step=step,
            metrics=combined_metrics,
            metric_value=value,
            optimizer=optimizer,
            scheduler=scheduler,
            architecture=architecture,
            extra=extra,
            aliases=aliases,
        )

    def _resolve_resume_path(self, path: str | Path | None) -> Path:
        if path is None:
            return self.checkpoint_dir / "last.pt"
        path = Path(path)
        if not path.is_absolute():
            candidate = self.checkpoint_dir / path
            path = candidate if candidate.exists() else self.config.run_dir / path
        return path

    def _validate(self, payload: Mapping[str, Any]) -> None:
        spec = payload.get("model_spec")
        if spec is not None and spec.get("canonical_name") != self.config.model_spec.canonical_name:
            raise ValueError(
                "Checkpoint model spec mismatch: "
                f"expected {self.config.model_spec.canonical_name!r}, "
                f"got {spec.get('canonical_name')!r}"
            )
        architecture = payload.get("architecture") or {}
        for key, expected in self.expected_architecture.items():
            if architecture.get(key) != expected:
                raise ValueError(
                    f"Checkpoint architecture mismatch for {key!r}: "
                    f"expected {expected!r}, got {architecture.get(key)!r}"
                )
        if self.validation_hook is not None:
            self.validation_hook(payload)

    def _verify_hash(self, path: Path) -> None:
        manifest = self._manifest()
        relative = self._relative(path)
        records = [item for item in manifest["checkpoints"] if item.get("path") == relative]
        expected = records[-1].get("sha256") if records else None
        if expected is None:
            expected = (manifest.get("aliases", {}).get(relative) or {}).get("sha256")
        if expected and sha256_file(path) != expected:
            raise ValueError(f"Checkpoint SHA256 mismatch: {relative}")

    def load_resume(
        self,
        path: str | Path | None = None,
        *,
        model: Any | None = None,
        optimizer: Any | None = None,
        scheduler: Any | None = None,
        map_location: str | torch.device = "cpu",
        strict: bool = True,
        verify_hash: bool = True,
    ) -> ResumeState:
        """Safely load tensor/state-dict payloads and optionally restore objects."""

        resolved = self._resolve_resume_path(path)
        if not resolved.exists():
            raise FileNotFoundError(f"Resume checkpoint does not exist: {resolved}")
        if verify_hash:
            self._verify_hash(resolved)
        payload = torch.load(resolved, map_location=map_location, weights_only=True)
        if not isinstance(payload, Mapping):
            raise TypeError("Checkpoint payload must be a mapping")
        self._validate(payload)
        model_state = payload.get("model_state_dict", payload.get("state_dict"))
        if model_state is None:
            raise KeyError("Checkpoint has no model_state_dict or legacy state_dict")
        if model is not None:
            model.load_state_dict(model_state, strict=strict)
        if optimizer is not None and "optimizer_state_dict" in payload:
            optimizer.load_state_dict(payload["optimizer_state_dict"])
        if scheduler is not None and "scheduler_state_dict" in payload:
            scheduler.load_state_dict(payload["scheduler_state_dict"])
        return ResumeState(
            path=resolved,
            epoch=payload.get("epoch"),
            step=payload.get("step"),
            metrics=dict(payload.get("metrics") or {}),
            payload=payload,
        )


__all__ = ["CHECKPOINT_FORMAT_VERSION", "CheckpointManager", "ResumeState"]
