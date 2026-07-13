"""Validate the four finalized AE checkpoints and build Notebook 5's six-run manifest."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


DEFAULT_DOWNSTREAM_RUNS = (
    {"run": "mixed_stage1a_on_pubmed", "domain": "pubmed", "type": "baseline", "branch": "baseline_mixed_stage1a", "ae_registry_key": "mixed_stage1a"},
    {"run": "mixed_to_pubmed_stage1b_on_pubmed", "domain": "pubmed", "type": "specialized", "branch": "specialized_mixed_to_pubmed", "ae_registry_key": "mixed_to_pubmed_stage1b"},
    {"run": "mixed_stage1a_on_nilearn", "domain": "nilearn", "type": "baseline", "branch": "baseline_mixed_stage1a", "ae_registry_key": "mixed_stage1a"},
    {"run": "mixed_to_nilearn_stage1b_on_nilearn", "domain": "nilearn", "type": "specialized", "branch": "specialized_mixed_to_nilearn", "ae_registry_key": "mixed_to_nilearn_stage1b"},
    {"run": "mixed_stage1a_on_neurovault", "domain": "neurovault", "type": "baseline", "branch": "baseline_mixed_stage1a", "ae_registry_key": "mixed_stage1a"},
    {"run": "mixed_to_neurovault_stage1b_on_neurovault", "domain": "neurovault", "type": "specialized", "branch": "specialized_mixed_to_neurovault", "ae_registry_key": "mixed_to_neurovault_stage1b"},
)


@dataclass(frozen=True)
class IntegrationConfig:
    output_root: Path
    selected_checkpoints: dict[str, Any]
    required_selection_keys: tuple[str, ...] | None = None
    downstream_runs: tuple[dict[str, str], ...] | list[dict[str, str]] | None = None


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _state_dict(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise TypeError("checkpoint payload must be a dictionary")
    state = payload.get("model") or payload.get("autoencoder") or payload.get("state_dict")
    if not isinstance(state, dict):
        raise KeyError("checkpoint does not contain model, autoencoder, or state_dict")
    return state


def _state_checksum(state: dict[str, Any]) -> str:
    import torch

    digest = hashlib.sha256()
    for key in sorted(state):
        value = state[key]
        if not torch.is_tensor(value):
            continue
        tensor = value.detach().cpu().contiguous()
        digest.update(key.encode())
        digest.update(str(tensor.dtype).encode())
        digest.update(str(tuple(tensor.shape)).encode())
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _validate_checkpoint(key: str, entry: dict[str, Any]) -> dict[str, Any]:
    import torch

    path = Path(str(entry.get("path") or entry.get("checkpoint_path") or "")).expanduser()
    row: dict[str, Any] = {
        "key": key,
        "checkpoint_path": str(path),
        "checkpoint_name": path.name,
        "status": "missing_checkpoint",
        "model_state_checksum": "",
        "warnings": "",
    }
    if not path.is_file():
        row["warnings"] = "checkpoint file does not exist"
        return row
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        state = _state_dict(payload)
        keys = tuple(str(name) for name in state)
        if not any(name.startswith("encoder.") for name in keys):
            raise KeyError("checkpoint has no encoder parameters")
        if not any(name.startswith("decoder.") for name in keys):
            raise KeyError("checkpoint has no decoder parameters")
        row["model_state_checksum"] = _state_checksum(state)
        row["status"] = "completed"
    except Exception as exc:
        row["status"] = "incompatible_checkpoint"
        row["warnings"] = repr(exc)
    return row


def integrate_completed_stage1_selection(config: IntegrationConfig) -> dict[str, Any]:
    output_dir = config.output_root / f"stage1_selection_integration_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    registry_dir = output_dir / "02_selected_checkpoint_registry"
    downstream_dir = output_dir / "03_downstream_usage"
    required_keys = tuple(config.required_selection_keys or config.selected_checkpoints.keys())
    missing_config = [key for key in required_keys if key not in config.selected_checkpoints]
    if missing_config:
        raise KeyError(f"Missing configured Stage 1 checkpoints: {missing_config}")

    validation_rows = [
        _validate_checkpoint(key, dict(config.selected_checkpoints[key]))
        for key in required_keys
    ]
    validation_by_key = {row["key"]: row for row in validation_rows}
    blocking = [row for row in validation_rows if row["status"] != "completed"]

    selected_registry: dict[str, dict[str, Any]] = {}
    for key in required_keys:
        configured = dict(config.selected_checkpoints[key])
        validation = validation_by_key[key]
        selected_registry[key] = {
            "path": validation["checkpoint_path"],
            "checkpoint_name": validation["checkpoint_name"],
            "stage": configured.get("stage", ""),
            "training_domain": configured.get("training_domain", ""),
            "selection_reason": configured.get("selection_reason", "finalized_pipeline"),
            "model_state_checksum": validation["model_state_checksum"],
            "validation_status": validation["status"],
        }

    downstream_specs = [dict(row) for row in (config.downstream_runs or DEFAULT_DOWNSTREAM_RUNS)]
    downstream_manifest: list[dict[str, Any]] = []
    assignment_rows: list[dict[str, Any]] = []
    for spec in downstream_specs:
        ae_key = spec["ae_registry_key"]
        if ae_key not in selected_registry:
            raise KeyError(f"Downstream run {spec['run']!r} requires missing checkpoint {ae_key!r}")
        selected = selected_registry[ae_key]
        row = {
            **spec,
            "ae_checkpoint_path": selected["path"],
            "selection_reason": selected["selection_reason"],
            "stage3_data_split": spec["domain"],
            "stage4_data_split": spec["domain"],
        }
        downstream_manifest.append(row)
        assignment_rows.append({
            "Run": spec["run"],
            "Domain": spec["domain"],
            "Type": spec["type"],
            "AE checkpoint": selected["checkpoint_name"],
            "absolute_checkpoint_path": selected["path"],
            "checksum": selected["model_state_checksum"],
            "ae_registry_key": ae_key,
        })

    status = "completed" if not blocking else (
        "missing_checkpoint" if any(row["status"] == "missing_checkpoint" for row in blocking)
        else "incompatible_checkpoint"
    )
    manifest_path = downstream_dir / "stage2_stage3_stage4_input_manifest.json"
    validation_path = registry_dir / "selected_checkpoint_validation.json"
    _write_json(validation_path, validation_rows)
    _write_json(registry_dir / "selected_ae_checkpoints_for_stage2_stage3_stage4.json", selected_registry)
    _write_csv(downstream_dir / "selected_ae_branch_assignment.csv", assignment_rows)
    _write_csv(downstream_dir / "six_run_ae_assignment.csv", assignment_rows)
    _write_json(manifest_path, {
        "selected_ae_checkpoints": selected_registry,
        "six_stage2_stage3_stage4_runs": downstream_manifest,
        "selected_stage2_stage3_stage4_runs": downstream_manifest,
        "validation_report": str(validation_path),
    })
    _write_json(output_dir / "00_metadata/run_status.json", {
        "status": status,
        "required_selection_keys": list(required_keys),
        "downstream_run_count": len(downstream_manifest),
        "blocking_checkpoints": blocking,
    })
    return {
        "status": status,
        "output_dir": str(output_dir),
        "blocking_checkpoints": blocking,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selected-checkpoints-json", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    selected = json.loads(args.selected_checkpoints_json.read_text())
    result = integrate_completed_stage1_selection(IntegrationConfig(args.output_root, selected))
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
