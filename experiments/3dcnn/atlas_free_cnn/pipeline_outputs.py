"""Run directory, checkpoint selection, and final status helpers."""

from __future__ import annotations

import csv
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any


STAGE_DIRS = {
    "metadata": "00_run_metadata",
    "stage1": "01_stage1_ae_pretraining",
    "stage1b": "02_stage1b_ae_finetuning",
    "stage2": "03_stage2_encoder_initialization",
    "stage3": "04_stage3_contrastive",
    "stage4": "05_stage4_text_to_brain_projection",
    "stage5": "06_stage5_generation_eval",
    "final": "07_final_comparison",
}

AE_SELECTION_TO_FILE = {
    "best_val_loss": "best_val_loss.pt",
    "best_spatial_corr": "best_spatial_corr.pt",
    "best_top1_dice": "best_top1_dice.pt",
    "best_top5_dice": "best_top5_dice.pt",
    "best_foreground_mse": "best_foreground_mse.pt",
    "last": "last.pt",
}


def git_info(repo_root: str | Path = ".") -> dict[str, Any]:
    root = Path(repo_root)
    out: dict[str, Any] = {}
    for key, cmd in {
        "commit": ["git", "rev-parse", "HEAD"],
        "branch": ["git", "rev-parse", "--abbrev-ref", "HEAD"],
        "dirty_short_status": ["git", "status", "--short"],
    }.items():
        try:
            value = subprocess.check_output(cmd, cwd=root, text=True, stderr=subprocess.DEVNULL).strip()
        except Exception:
            value = ""
        out[key] = value
    out["is_dirty"] = bool(out.get("dirty_short_status"))
    return out


def create_full_pipeline_run_dir(
    base_dir: str | Path = "runs_atlas_free_cnn_full_pipeline",
    *,
    prefix: str = "full_atlas_free_cnn",
    overwrite: bool = False,
) -> dict[str, str]:
    base = Path(base_dir)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    run_dir = base / f"{prefix}_{stamp}"
    if run_dir.exists() and not overwrite:
        suffix = 1
        while (base / f"{prefix}_{stamp}_{suffix:02d}").exists():
            suffix += 1
        run_dir = base / f"{prefix}_{stamp}_{suffix:02d}"
    for rel in STAGE_DIRS.values():
        (run_dir / rel).mkdir(parents=True, exist_ok=True)
    for rel in ["config", "checkpoints", "metrics", "plots"]:
        (run_dir / STAGE_DIRS["stage1"] / rel).mkdir(parents=True, exist_ok=True)
        (run_dir / STAGE_DIRS["stage3"] / rel).mkdir(parents=True, exist_ok=True)
    for domain in ["pubmed", "statmaps", "comparison"]:
        (run_dir / STAGE_DIRS["stage1b"] / domain).mkdir(parents=True, exist_ok=True)
    return {"run_dir": str(run_dir), **{key: str(run_dir / value) for key, value in STAGE_DIRS.items()}}


def write_json(path: str | Path, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(payload, f, indent=2)


def read_json(path: str | Path) -> Any:
    with Path(path).open() as f:
        return json.load(f)


def select_ae_checkpoint(
    *,
    checkpoint_dir: str | Path | None = None,
    explicit_path: str | Path | None = None,
    selection: str = "best_val_loss",
) -> dict[str, Any]:
    if explicit_path:
        path = Path(explicit_path)
        return {
            "ae_checkpoint_path": str(path),
            "ae_checkpoint_selection": "custom_path",
            "exists": path.exists(),
        }
    if checkpoint_dir is None:
        raise ValueError("checkpoint_dir is required when explicit_path is not supplied")
    filename = AE_SELECTION_TO_FILE.get(selection, selection)
    path = Path(checkpoint_dir) / filename
    return {
        "ae_checkpoint_path": str(path),
        "ae_checkpoint_selection": selection,
        "exists": path.exists(),
    }


def _valid_json(path: Path, required: list[str] | None = None) -> bool:
    if not path.exists():
        return False
    try:
        payload = read_json(path)
    except Exception:
        return False
    if required:
        return all(key in payload for key in required)
    return True


def detect_stage_status(
    stage_name: str,
    *,
    requested: bool,
    stage_dir: str | Path,
) -> dict[str, Any]:
    path = Path(stage_dir)
    if not requested:
        return {"stage": stage_name, "status": "not requested", "warnings": []}
    warnings: list[str] = []
    if stage_name == "stage1":
        ok = any((path / "checkpoints" / name).exists() for name in AE_SELECTION_TO_FILE.values())
        metrics = (path / "metrics" / "reconstruction_summary_by_source.csv").exists() or (path / "autoencoder_reconstruction_metrics.csv").exists()
    elif stage_name == "stage1b":
        ok = any(path.glob("*/checkpoints/best_*.pt"))
        metrics = any(path.glob("*/metrics/reconstruction_summary_by_source.csv"))
    elif stage_name == "stage3":
        ok = (path / "checkpoints" / "best_contrastive.pt").exists() or (path / "checkpoints" / "best_ale_cnn.pt").exists()
        metrics = _valid_json(path / "metrics" / "test_metrics.json") or _valid_json(path / "eval_results.json")
        ok = ok or metrics
    elif stage_name == "stage4":
        ok = any(path.glob("checkpoints/*.pt"))
        metrics = any(path.glob("metrics/*metrics*.json")) or any(path.glob("metrics/*history*.csv"))
    elif stage_name == "stage5":
        ok = _valid_json(path / "metrics" / "generation_eval_metrics.json")
        metrics = ok and ((path / "generated_maps" / "predictions").exists() or (path / "metrics" / "generated_vs_target_metrics_all_rows.csv").exists())
    else:
        ok = path.exists()
        metrics = ok
    if ok and metrics:
        status = "ran successfully"
    elif ok or metrics:
        status = "ran but missing expected outputs"
        warnings.append("Some expected output files were not found.")
    else:
        status = "requested but skipped"
        warnings.append("No expected output files were found.")
    return {"stage": stage_name, "status": status, "warnings": warnings}


def write_status_report(
    run_dir: str | Path,
    requested: dict[str, bool],
) -> list[dict[str, Any]]:
    run = Path(run_dir)
    statuses = [
        detect_stage_status("stage1", requested=requested.get("stage1", False), stage_dir=run / STAGE_DIRS["stage1"]),
        detect_stage_status("stage1b", requested=requested.get("stage1b", False), stage_dir=run / STAGE_DIRS["stage1b"]),
        detect_stage_status("stage3", requested=requested.get("stage3", False), stage_dir=run / STAGE_DIRS["stage3"]),
        detect_stage_status("stage4", requested=requested.get("stage4", False), stage_dir=run / STAGE_DIRS["stage4"]),
        detect_stage_status("stage5", requested=requested.get("stage5", False), stage_dir=run / STAGE_DIRS["stage5"]),
    ]
    write_json(run / "00_run_metadata" / "run_status.json", statuses)
    write_json(run / "07_final_comparison" / "final_model_card.json", {"run_dir": str(run), "stage_status": statuses})
    csv_path = run / "00_run_metadata" / "run_status.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["stage", "status", "warnings"])
        writer.writeheader()
        for row in statuses:
            writer.writerow({**row, "warnings": "; ".join(row.get("warnings", []))})
    return statuses


def write_readme_what_to_look_at(
    path: str | Path,
    *,
    ae_variant: str,
    ae_selection: str,
    ae_checkpoint_path: str,
    stage_status: list[dict[str, Any]],
    warnings: list[str] | None = None,
) -> None:
    warnings = warnings or []
    stage_lines = "\n".join(f"- {row['stage']}: {row['status']}" for row in stage_status)
    warning_lines = "\n".join(f"- {item}" for item in warnings) if warnings else "- None recorded."
    text = f"""# What To Look At

AE variant: `{ae_variant}`
Selected AE checkpoint: `{ae_checkpoint_path}`
Selection metric: `{ae_selection}`

Stage status:
{stage_lines}

Main files to inspect:
- `07_final_comparison/final_summary_table.csv`
- `07_final_comparison/ae_to_downstream_comparison.csv`
- `07_final_comparison/best_checkpoints_to_inspect.csv`
- `01_stage1_ae_pretraining/metrics/reconstruction_summary_by_source.csv`
- `04_stage3_contrastive/metrics/test_metrics.json`
- `06_stage5_generation_eval/metrics/generation_eval_metrics.json`

Warnings or failed stages:
{warning_lines}
"""
    Path(path).write_text(text)


def write_table(path: str | Path, rows: list[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def flatten_stage3_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    test = metrics.get("test", metrics)
    out: dict[str, Any] = {}
    mapping = {
        "stage3_i2t_recall@1": ["i2t_recall@1", "image_to_text_recall@1", "mean_recall@1"],
        "stage3_i2t_recall@10": ["i2t_recall@10", "image_to_text_recall@10", "mean_recall@10"],
        "stage3_i2t_recall@50": ["i2t_recall@50", "image_to_text_recall@50", "mean_recall@50"],
        "stage3_i2t_auc": ["i2t_auc", "image_to_text_auc", "paper_recall_curve_auc"],
        "stage3_t2i_recall@1": ["t2i_recall@1", "text_to_image_recall@1", "mean_recall@1"],
        "stage3_t2i_recall@10": ["t2i_recall@10", "text_to_image_recall@10", "mean_recall@10"],
        "stage3_t2i_recall@50": ["t2i_recall@50", "text_to_image_recall@50", "mean_recall@50"],
        "stage3_t2i_auc": ["t2i_auc", "text_to_image_auc", "paper_recall_curve_auc"],
    }
    for dst, candidates in mapping.items():
        out[dst] = next((test.get(key) for key in candidates if key in test), "")
    return out


def ae_source_metric_columns(summary_csv: str | Path) -> dict[str, Any]:
    path = Path(summary_csv)
    out: dict[str, Any] = {}
    if not path.exists():
        return out
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            if row.get("split") not in {"val", "test", ""}:
                continue
            if row.get("source_detail") not in {"ALL_DETAILS", "", None}:
                continue
            src = row.get("source", "")
            if src not in {"pubmed", "neurovault", "nilearn"}:
                continue
            out[f"ae_{src}_spatial_corr"] = row.get("spatial_corr", "")
            out[f"ae_{src}_top5_dice"] = row.get("top5_dice", "")
    return out
