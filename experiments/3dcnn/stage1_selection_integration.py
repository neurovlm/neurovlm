"""Integrate completed Stage 1 checkpoint-evaluation outputs for downstream runs.

This module is intentionally evaluation-results-only. It loads the completed
Stage 1A/1B checkpoint-evaluation tables produced by notebook 7, validates the
empirically selected checkpoint files, and writes a downstream manifest for the
Stage 2/3/4 controlled experiments. It never trains, fine-tunes, resumes, or
runs checkpoint inference by default.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


TARGET_SHAPE = (36, 45, 38)
LATENT_DIM = 384

REQUIRED_SELECTIONS: dict[str, dict[str, str]] = {
    "mixed_stage1a": {
        "stage": "stage1a",
        "training_domain": "mixed",
        "checkpoint_name": "best_top1_dice.pt",
        "selection_reason": "held_out_multi_source_rank_1",
        "source_kind": "stage1a",
        "selection_csv": "01_stage1a/mixed_stage1a_checkpoint_selection.csv",
    },
    "mixed_to_pubmed_stage1b": {
        "stage": "stage1b",
        "training_domain": "pubmed",
        "checkpoint_name": "best_top1_dice.pt",
        "selection_reason": "held_out_domain_rank_1",
        "source_kind": "stage1b",
        "selection_csv": "02_stage1b/pubmed/pubmed_stage1b_checkpoint_selection.csv",
    },
    "mixed_to_nilearn_stage1b": {
        "stage": "stage1b",
        "training_domain": "nilearn",
        "checkpoint_name": "best_val_loss.pt",
        "selection_reason": "held_out_top5_dice_rank_1",
        "source_kind": "stage1b",
        "selection_csv": "02_stage1b/nilearn/nilearn_stage1b_checkpoint_selection.csv",
    },
    "mixed_to_neurovault_stage1b": {
        "stage": "stage1b",
        "training_domain": "neurovault",
        "checkpoint_name": "best_top5_dice.pt",
        "selection_reason": "held_out_top5_dice_rank_1",
        "source_kind": "stage1b",
        "selection_csv": "02_stage1b/neurovault/neurovault_stage1b_checkpoint_selection.csv",
    },
}

SIX_RUNS = [
    ("mixed_stage1a_on_pubmed", "PubMed", "baseline", "mixed_stage1a"),
    ("mixed_to_pubmed_stage1b_on_pubmed", "PubMed", "specialized", "mixed_to_pubmed_stage1b"),
    ("mixed_stage1a_on_nilearn", "Nilearn", "baseline", "mixed_stage1a"),
    ("mixed_to_nilearn_stage1b_on_nilearn", "Nilearn", "specialized", "mixed_to_nilearn_stage1b"),
    ("mixed_stage1a_on_neurovault", "NeuroVault", "baseline", "mixed_stage1a"),
    ("mixed_to_neurovault_stage1b_on_neurovault", "NeuroVault", "specialized", "mixed_to_neurovault_stage1b"),
]


@dataclass(frozen=True)
class IntegrationConfig:
    stage1a_evaluation_dir: Path
    stage1b_evaluation_dir: Path
    output_root: Path
    rerun_stage1_checkpoint_evaluation: bool = False


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open() as f:
        return json.load(f)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(value, f, indent=2, sort_keys=True)
        f.write("\n")


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def write_csv_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(v) for v in value]
    return value


def source_dir_for(config: IntegrationConfig, spec: dict[str, str]) -> Path:
    if spec["source_kind"] == "stage1a":
        return config.stage1a_evaluation_dir.expanduser()
    return config.stage1b_evaluation_dir.expanduser()


def selection_manifest_path(source_dir: Path) -> Path:
    return source_dir / "04_final_selection/selected_stage2_checkpoints.json"


def checkpoint_manifest_path(source_dir: Path) -> Path:
    return source_dir / "00_metadata/checkpoint_manifest.csv"


def split_fingerprint_path(source_dir: Path) -> Path:
    return source_dir / "00_metadata/test_split_fingerprints.json"


def find_manifest_row(manifest_rows: list[dict[str, str]], checkpoint_path: Path, checkpoint_name: str) -> dict[str, str]:
    resolved = str(checkpoint_path.expanduser().resolve()) if checkpoint_path.exists() else str(checkpoint_path.expanduser())
    for row in manifest_rows:
        if row.get("checkpoint_path") == resolved:
            return row
    matches = [row for row in manifest_rows if row.get("checkpoint_name") == checkpoint_name]
    return matches[0] if len(matches) == 1 else {}


def completed_row_for(selection_csv: Path, checkpoint_name: str) -> dict[str, str]:
    rows = read_csv_rows(selection_csv)
    for row in rows:
        if row.get("checkpoint_name") == checkpoint_name:
            return row
    return {}


def selected_checkpoint_path(source_dir: Path, key: str, spec: dict[str, str]) -> tuple[Path | None, dict[str, Any], list[str]]:
    warnings: list[str] = []
    selected = read_json(selection_manifest_path(source_dir)).get(key, {})
    selection_csv = source_dir / spec["selection_csv"]
    csv_row = completed_row_for(selection_csv, spec["checkpoint_name"])

    selected_name = selected.get("checkpoint_name")
    selected_path = selected.get("checkpoint_path")
    if selected_name and selected_name != spec["checkpoint_name"]:
        warnings.append(
            f"completed manifest selected {selected_name}, but downstream registry requires {spec['checkpoint_name']}"
        )
    if selected_path and Path(str(selected_path)).name != spec["checkpoint_name"]:
        warnings.append(
            f"completed manifest path points to {Path(str(selected_path)).name}, expected {spec['checkpoint_name']}"
        )

    path_value = csv_row.get("checkpoint_path") or selected_path
    if not path_value:
        return None, csv_row or selected, warnings
    path = Path(str(path_value)).expanduser()
    if path.name != spec["checkpoint_name"]:
        warnings.append(f"selection table path points to {path.name}, expected {spec['checkpoint_name']}")
    return path, csv_row or selected, warnings


def state_checksum(state: dict[str, Any], prefix: str | None = None) -> str:
    import torch

    h = hashlib.sha256()
    keys = sorted(k for k in state if prefix is None or str(k).startswith(prefix))
    for key in keys:
        value = state[key]
        if not torch.is_tensor(value):
            continue
        tensor = value.detach().cpu().contiguous()
        h.update(str(key).encode("utf-8"))
        h.update(str(tensor.dtype).encode("utf-8"))
        h.update(json.dumps(list(tensor.shape)).encode("utf-8"))
        h.update(tensor.numpy().tobytes())
    return h.hexdigest()


def extract_model_state(payload: Any) -> dict[str, Any]:
    if isinstance(payload, dict):
        state = payload.get("model") or payload.get("autoencoder") or payload.get("state_dict")
    else:
        state = payload
    if not isinstance(state, dict):
        raise KeyError("checkpoint does not contain model, autoencoder, or state_dict")
    keys = [str(k) for k in state]
    if not any(k.startswith("encoder.") for k in keys):
        raise KeyError("checkpoint does not contain encoder state")
    if not any(k.startswith("decoder.") for k in keys):
        raise KeyError("checkpoint does not contain decoder state")
    return state


def nested_get(mapping: dict[str, Any], *keys: str) -> Any:
    cur: Any = mapping
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            return None
        cur = cur[key]
    return cur


def checkpoint_architecture(payload: dict[str, Any]) -> tuple[Any, Any]:
    latent_dim = (
        nested_get(payload, "config", "model", "latent_dim")
        or nested_get(payload, "model_architecture", "latent_dim")
        or payload.get("latent_dim")
    )
    target_shape = (
        payload.get("target_shape")
        or nested_get(payload, "config", "target_shape")
        or nested_get(payload, "model_architecture", "target_shape")
    )
    if isinstance(target_shape, list):
        target_shape = tuple(target_shape)
    return latent_dim, target_shape


def validate_checkpoint(
    checkpoint_path: Path | None,
    checkpoint_name: str,
    manifest_row: dict[str, str],
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "checkpoint_path": str(checkpoint_path) if checkpoint_path else "",
        "checkpoint_name": checkpoint_name,
        "exists": bool(checkpoint_path and checkpoint_path.exists()),
        "readable": False,
        "model_state_exists": False,
        "encoder_state_exists": False,
        "decoder_state_exists": False,
        "latent_dim": "",
        "target_shape": "",
        "checkpoint_epoch": "",
        "model_state_checksum": "",
        "encoder_checksum": "",
        "decoder_checksum": "",
        "manifest_checksum": manifest_row.get("model_state_checksum", ""),
        "checksum_matches_manifest": "",
        "status": "missing_checkpoint",
        "warnings": "",
    }
    warnings: list[str] = []
    if not checkpoint_path or not checkpoint_path.exists():
        return row
    try:
        import torch

        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        row["readable"] = True
        state = extract_model_state(payload)
        row["model_state_exists"] = True
        row["encoder_state_exists"] = any(str(k).startswith("encoder.") for k in state)
        row["decoder_state_exists"] = any(str(k).startswith("decoder.") for k in state)
        row["model_state_checksum"] = state_checksum(state)
        row["encoder_checksum"] = state_checksum(state, "encoder.")
        row["decoder_checksum"] = state_checksum(state, "decoder.")
        if isinstance(payload, dict):
            row["checkpoint_epoch"] = payload.get("epoch", "")
            latent_dim, target_shape = checkpoint_architecture(payload)
            row["latent_dim"] = latent_dim if latent_dim is not None else ""
            row["target_shape"] = list(target_shape) if isinstance(target_shape, tuple) else (target_shape or "")
        observed_target_shape = tuple(row["target_shape"]) if isinstance(row["target_shape"], list) else row["target_shape"]
        if str(row["latent_dim"]) != str(LATENT_DIM):
            warnings.append(f"latent_dim is {row['latent_dim']!r}, expected {LATENT_DIM}")
        if observed_target_shape != TARGET_SHAPE:
            if row["target_shape"]:
                warnings.append(f"target_shape is {row['target_shape']!r}, expected {TARGET_SHAPE}")
            else:
                warnings.append("target_shape missing from checkpoint payload")
        manifest_checksum = row["manifest_checksum"]
        if manifest_checksum:
            row["checksum_matches_manifest"] = manifest_checksum == row["model_state_checksum"]
            if not row["checksum_matches_manifest"]:
                row["status"] = "checksum_mismatch"
                row["warnings"] = "; ".join(warnings)
                return row
        else:
            row["checksum_matches_manifest"] = "not_available"
            warnings.append("checkpoint_manifest.csv did not provide a checksum for this selected checkpoint")
        if str(row["latent_dim"]) == str(LATENT_DIM) and observed_target_shape == TARGET_SHAPE:
            row["status"] = "completed" if not warnings else "completed_with_warnings"
        else:
            row["status"] = "incompatible_checkpoint"
    except Exception as exc:
        row["status"] = "incompatible_checkpoint"
        warnings.append(repr(exc))
    row["warnings"] = "; ".join(warnings)
    return row


def held_out_metrics(source: dict[str, Any]) -> dict[str, Any]:
    if isinstance(source.get("held_out_metrics"), dict):
        return source["held_out_metrics"]
    skip = {"checkpoint_path", "checkpoint_name", "canonical_checkpoint_name", "rank", "domain", "variant"}
    return {k: v for k, v in source.items() if k not in skip}


def split_fingerprint(source_dir: Path, training_domain: str) -> Any:
    data = read_json(split_fingerprint_path(source_dir))
    if training_domain == "mixed":
        return data
    domain_data = data.get(training_domain) if isinstance(data, dict) else None
    if isinstance(domain_data, dict):
        return domain_data.get("fingerprint") or domain_data
    return domain_data or data


def copy_if_exists(src: Path, dst: Path) -> bool:
    if not src.exists():
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return True


def write_readme(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        """# Stage 1 Selection Integration

Stage 1 training is complete. Stage 1 checkpoint evaluation is complete.

Notebook 5 does not train, fine-tune, resume, optimize, or rerun checkpoint inference for the autoencoders. It loads the completed checkpoint-evaluation outputs from notebooks 6 and 7, validates the selected checkpoint files, and writes the downstream manifests used by Stage 2/3/4.

The Stage 1A and Stage 1B evaluations may come from separate timestamped folders. The integration output records both source directories and preserves the original selection tables.

Selected checkpoints:

* Stage 1A mixed: `best_top1_dice.pt`, selected by `held_out_multi_source_rank_1`.
* Stage 1B PubMed: `best_top1_dice.pt`, selected by `held_out_domain_rank_1`.
* Stage 1B Nilearn: `best_val_loss.pt`, selected by `held_out_top5_dice_rank_1`.
* Stage 1B NeuroVault: `best_top5_dice.pt`, selected by `held_out_top5_dice_rank_1`.

The Nilearn checkpoint is named `best_val_loss.pt` because the completed held-out checkpoint evaluation showed that exact checkpoint was the strongest relevant held-out top-5 Dice choice for the Nilearn branch. Do not replace it with `best_top5_dice.pt` based only on filename.

Downstream notebooks should load:

* `02_selected_checkpoint_registry/selected_ae_checkpoints_for_stage2_stage3_stage4.json`
* `03_downstream_usage/stage2_stage3_stage4_input_manifest.json`
* `03_downstream_usage/six_run_ae_assignment.csv`

Rerun Stage 1 checkpoint evaluation only if checkpoint files changed, test splits changed, metric definitions changed, evaluation outputs are missing/corrupted, or explicit reproduction is requested.
""",
        encoding="utf-8",
    )


def integrate_completed_stage1_selection(config: IntegrationConfig) -> dict[str, Any]:
    if config.rerun_stage1_checkpoint_evaluation:
        raise RuntimeError(
            "RERUN_STAGE1_CHECKPOINT_EVALUATION=True is a reproduction workflow. "
            "Run notebook 7 into a new timestamped evaluation folder, then compare results; "
            "this integration path does not silently replace established selections."
        )

    output_dir = config.output_root / f"stage1_selection_integration_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    metadata_dir = output_dir / "00_metadata"
    existing_tables_dir = output_dir / "01_existing_evaluation_tables"
    registry_dir = output_dir / "02_selected_checkpoint_registry"
    downstream_dir = output_dir / "03_downstream_usage"
    for directory in [metadata_dir, existing_tables_dir, registry_dir, downstream_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    write_json(metadata_dir / "integration_config.json", json_ready(config.__dict__))
    write_json(
        metadata_dir / "source_evaluation_directories.json",
        {
            "stage1a_evaluation_dir": str(config.stage1a_evaluation_dir),
            "stage1b_evaluation_dir": str(config.stage1b_evaluation_dir),
        },
    )

    unified_manifest: dict[str, Any] = {"evaluation_state": "completed", "rerun_required": False}
    validation_rows: list[dict[str, Any]] = []
    checksum_rows: list[dict[str, Any]] = []
    table_copy_rows: list[dict[str, Any]] = []
    selected_registry: dict[str, Any] = {}

    for key, spec in REQUIRED_SELECTIONS.items():
        source_dir = source_dir_for(config, spec)
        selection_csv = source_dir / spec["selection_csv"]
        selected_path, source_row, selection_warnings = selected_checkpoint_path(source_dir, key, spec)
        manifest_rows = read_csv_rows(checkpoint_manifest_path(source_dir))
        manifest_row = find_manifest_row(manifest_rows, selected_path, spec["checkpoint_name"]) if selected_path else {}
        validation = validate_checkpoint(selected_path, spec["checkpoint_name"], manifest_row)
        if selection_warnings:
            validation["warnings"] = "; ".join([w for w in [validation.get("warnings"), *selection_warnings] if w])
        validation.update(
            {
                "key": key,
                "stage": spec["stage"],
                "training_domain": spec["training_domain"],
                "selection_reason": spec["selection_reason"],
                "source_evaluation_dir": str(source_dir),
                "source_selection_csv": str(selection_csv),
            }
        )
        validation_rows.append(validation)
        checksum_rows.append(
            {
                "key": key,
                "checkpoint_path": validation["checkpoint_path"],
                "model_state_checksum": validation["model_state_checksum"],
                "encoder_checksum": validation["encoder_checksum"],
                "decoder_checksum": validation["decoder_checksum"],
                "manifest_checksum": validation["manifest_checksum"],
                "checksum_matches_manifest": validation["checksum_matches_manifest"],
            }
        )

        entry = {
            "checkpoint_path": validation["checkpoint_path"],
            "checkpoint_name": spec["checkpoint_name"],
            "stage": spec["stage"],
            "training_domain": spec["training_domain"],
            "selection_reason": spec["selection_reason"],
            "source_evaluation_dir": str(source_dir),
            "source_selection_csv": str(selection_csv),
            "checkpoint_epoch": validation["checkpoint_epoch"],
            "model_state_checksum": validation["model_state_checksum"],
            "encoder_checksum": validation["encoder_checksum"],
            "decoder_checksum": validation["decoder_checksum"],
            "held_out_metrics": held_out_metrics(source_row),
            "test_split_fingerprint": split_fingerprint(source_dir, spec["training_domain"]),
            "evaluation_status": "completed",
            "validation_status": validation["status"],
            "warnings": validation["warnings"],
        }
        unified_manifest[key] = entry
        selected_registry[key] = {
            "path": validation["checkpoint_path"],
            "stage": spec["stage"],
            "training_domain": spec["training_domain"],
            "checkpoint_name": spec["checkpoint_name"],
            "selection_reason": spec["selection_reason"],
            "evaluation_status": "completed",
        }

        copied = copy_if_exists(selection_csv, existing_tables_dir / Path(spec["selection_csv"]).name)
        table_copy_rows.append({"source": str(selection_csv), "copied": copied})

    for source_kind, source_dir in [
        ("stage1a", config.stage1a_evaluation_dir),
        ("stage1b", config.stage1b_evaluation_dir),
    ]:
        copied = copy_if_exists(
            source_dir / "04_final_selection/all_checkpoint_leaderboard.csv",
            existing_tables_dir / f"{source_kind}_all_checkpoint_leaderboard.csv",
        )
        table_copy_rows.append(
            {
                "source": str(source_dir / "04_final_selection/all_checkpoint_leaderboard.csv"),
                "copied": copied,
            }
        )

    six_run_rows = []
    six_run_manifest = []
    for run_name, domain, run_type, ae_key in SIX_RUNS:
        entry = unified_manifest[ae_key]
        six_run_rows.append(
            {
                "Run": run_name,
                "Domain": domain,
                "Type": run_type,
                "AE checkpoint": f"{entry['training_domain']} {entry['stage']} `{entry['checkpoint_name']}`",
                "absolute_checkpoint_path": entry["checkpoint_path"],
                "checkpoint_epoch": entry["checkpoint_epoch"],
                "checksum": entry["model_state_checksum"],
                "selection_reason": entry["selection_reason"],
                "source_evaluation_folder": entry["source_evaluation_dir"],
                "intended_stage3_data_split": domain.lower(),
                "intended_stage4_data_split": domain.lower(),
                "ae_registry_key": ae_key,
            }
        )
        six_run_manifest.append(
            {
                "run": run_name,
                "domain": domain.lower(),
                "type": run_type,
                "ae_registry_key": ae_key,
                "ae_checkpoint_path": entry["checkpoint_path"],
                "selection_reason": entry["selection_reason"],
                "stage3_data_split": domain.lower(),
                "stage4_data_split": domain.lower(),
            }
        )

    blocking = [
        row for row in validation_rows
        if row["status"] not in {"completed", "completed_with_warnings"}
        or not row.get("checkpoint_path")
    ]
    if blocking:
        run_status = "missing_checkpoint" if any(row["status"] == "missing_checkpoint" for row in blocking) else "incompatible_checkpoint"
        unified_manifest["evaluation_state"] = run_status
    elif any(row["status"] == "completed_with_warnings" or row.get("warnings") for row in validation_rows):
        run_status = "completed_with_warnings"
        unified_manifest["evaluation_state"] = "completed"
    else:
        run_status = "completed"
        unified_manifest["evaluation_state"] = "completed"
    unified_manifest["integration_status"] = run_status

    write_json(registry_dir / "selected_ae_checkpoints_for_stage2_stage3_stage4.json", unified_manifest)
    write_json(registry_dir / "selected_checkpoint_validation.json", validation_rows)
    write_csv_rows(registry_dir / "checkpoint_checksums.csv", checksum_rows)
    write_csv_rows(downstream_dir / "six_run_ae_assignment.csv", six_run_rows)
    write_json(
        downstream_dir / "stage2_stage3_stage4_input_manifest.json",
        {
            "selected_ae_checkpoints": selected_registry,
            "six_stage2_stage3_stage4_runs": six_run_manifest,
            "selected_manifest": str(registry_dir / "selected_ae_checkpoints_for_stage2_stage3_stage4.json"),
            "validation_report": str(registry_dir / "selected_checkpoint_validation.json"),
        },
    )
    write_readme(downstream_dir / "README_WHAT_TO_LOOK_AT.md")
    write_json(
        metadata_dir / "run_status.json",
        {
            "status": run_status,
            "output_dir": str(output_dir),
            "all_selected_checkpoint_entries_valid": not blocking,
            "table_copies": table_copy_rows,
            "rerun_stage1_checkpoint_evaluation": False,
        },
    )
    return {"status": run_status, "output_dir": str(output_dir), "manifest": unified_manifest}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage1a-evaluation-dir", required=True, type=Path)
    parser.add_argument("--stage1b-evaluation-dir", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--rerun-stage1-checkpoint-evaluation", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = integrate_completed_stage1_selection(
        IntegrationConfig(
            stage1a_evaluation_dir=args.stage1a_evaluation_dir,
            stage1b_evaluation_dir=args.stage1b_evaluation_dir,
            output_root=args.output_root,
            rerun_stage1_checkpoint_evaluation=args.rerun_stage1_checkpoint_evaluation,
        )
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
