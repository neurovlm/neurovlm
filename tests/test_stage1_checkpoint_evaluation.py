from __future__ import annotations

import csv
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
THREEDCNN = REPO_ROOT / "experiments" / "3dcnn"
if str(THREEDCNN) not in sys.path:
    sys.path.insert(0, str(THREEDCNN))
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from atlas_free_cnn.evaluation.stage1_checkpoint_evaluation import (  # noqa: E402
    CHECKPOINT_FILENAMES,
    STAGE1A_RECIPE_BEST_COLUMNS,
    create_stage1a_selection,
)


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def _stage1a_row(recipe: str, checkpoint: str, domain: str, *, spatial: float, top5: float, mse: float) -> dict[str, object]:
    return {
        "variant": recipe,
        "stage": "stage1a",
        "training_domain": "mixed",
        "test_domain": domain,
        "eval_scope": "primary",
        "checkpoint_name": checkpoint,
        "checkpoint_path": f"/tmp/{recipe}/checkpoints/{checkpoint}",
        "canonical_checkpoint_name": checkpoint,
        "alias_status": "canonical",
        "checkpoint_epoch": 7 if checkpoint != "last.pt" else 9,
        "test_split_fingerprint": f"fp-{domain}",
        "load_status": "loaded",
        "reconstruction_mse": mse,
        "foreground_mse": mse / 2,
        "spatial_corr": spatial,
        "top1_dice": top5 / 2,
        "top5_dice": top5,
        "top10_dice": top5 + 0.1,
    }


def test_stage1a_selector_writes_recipe_best_after_per_recipe_checkpoint_selection(tmp_path: Path) -> None:
    registry = {
        "mixed_baseline_raw_mse": {"stage": "stage1a"},
    }
    comparison_rows = []
    domains = ["mixed", "pubmed", "nilearn", "neurovault"]
    for domain in domains:
        comparison_rows.extend(
            [
                _stage1a_row("mixed_baseline_raw_mse", "best_top1_dice.pt", domain, spatial=0.80, top5=0.70, mse=0.20),
                _stage1a_row("mixed_baseline_raw_mse", "last.pt", domain, spatial=0.20, top5=0.30, mse=0.80),
            ]
        )

    selection_rows, baseline_selected = create_stage1a_selection(comparison_rows, tmp_path, registry)

    assert baseline_selected is not None
    assert baseline_selected["checkpoint_name"] == "best_top1_dice.pt"
    assert {row["recipe"] for row in selection_rows} == set(registry)

    recipe_best = _read_rows(tmp_path / "01_stage1a" / "stage1a_recipe_best_checkpoint_comparison.csv")
    all_checkpoints = _read_rows(tmp_path / "01_stage1a" / "stage1a_all_checkpoint_eval.csv")
    compat_baseline = _read_rows(tmp_path / "01_stage1a" / "mixed_stage1a_checkpoint_selection.csv")

    assert [column for column in STAGE1A_RECIPE_BEST_COLUMNS if column not in recipe_best[0]] == []
    assert {row["recipe"]: row["best_checkpoint_name"] for row in recipe_best} == {
        "mixed_baseline_raw_mse": "best_top1_dice.pt",
    }
    assert len(all_checkpoints) == 2
    assert {row["checkpoint_name"] for row in all_checkpoints if row["is_recipe_best"] == "True"} == {
        "best_top1_dice.pt",
    }
    assert {row["recipe"] for row in compat_baseline} == {"mixed_baseline_raw_mse"}


def test_stage1a_checkpoint_discovery_includes_top10_candidate() -> None:
    assert "best_top10_dice.pt" in CHECKPOINT_FILENAMES


def test_stage1a_selector_marks_missing_recipe_explicitly(tmp_path: Path) -> None:
    registry = {"mixed_baseline_raw_mse": {"stage": "stage1a"}}
    manifest = [
        {
            "variant": "mixed_baseline_raw_mse",
            "stage": "stage1a",
            "run_dir": "/missing/run",
            "load_status": "missing_run_or_checkpoints",
            "error_message": "No requested checkpoint files found",
        }
    ]

    selection_rows, selected = create_stage1a_selection([], tmp_path, registry, manifest)

    assert selection_rows == []
    assert selected is None
    rows = _read_rows(tmp_path / "01_stage1a" / "stage1a_all_checkpoint_eval.csv")
    assert rows[0]["recipe"] == "mixed_baseline_raw_mse"
    assert rows[0]["status"] == "missing_run_or_checkpoints"
    assert "No requested checkpoint" in rows[0]["error_message"]
