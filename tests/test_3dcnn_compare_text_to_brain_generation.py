from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
THREEDCNN = REPO_ROOT / "experiments" / "3dcnn"
if str(THREEDCNN) not in sys.path:
    sys.path.insert(0, str(THREEDCNN))
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from atlas_free_cnn.evaluation import compare_text_to_brain_generation as compare


def _fake_metric_row(dataset: str, model_id: str, model_family: str, sample: int) -> dict[str, object]:
    true = np.asarray([sample, sample + 1, sample + 2, sample + 3], dtype=np.float32)
    pred = true + np.asarray([0.0, 0.1, -0.1, 0.0], dtype=np.float32)
    return {
        "dataset": dataset,
        "model_id": model_id,
        "model_family": model_family,
        "sample_id": f"{dataset}-{sample}",
        "comparison_space": "fake_space",
        "status": "ok",
        "pearson_r": 0.99,
        "spearman_rho": 1.0,
        "_brain_pred": pred,
        "_brain_true": true,
    }


def test_random_baseline_columns_are_grouped_by_dataset_and_family() -> None:
    rows = [
        _fake_metric_row("pubmed", "cnn_t2b_pubmed", "cnn_t2b", 0),
        _fake_metric_row("pubmed", "cnn_t2b_pubmed", "cnn_t2b", 1),
        _fake_metric_row("nilearn", "cnn_t2b_nilearn", "cnn_t2b", 0),
        _fake_metric_row("nilearn", "cnn_t2b_nilearn", "cnn_t2b", 1),
    ]

    enriched, baseline = compare.add_grouped_random_baselines(rows, n_random=2, seed=7, max_voxels=4)

    assert len(enriched) == 4
    assert len(baseline) == 4
    expected = {
        "pearson_random_mean",
        "pearson_minus_random",
        "pearson_random_percentile",
        "spearman_random_mean",
        "spearman_minus_random",
        "spearman_random_percentile",
        "random_baseline_group_dataset",
        "random_baseline_group_model_family",
    }
    assert expected <= set(baseline[0])
    assert {row["random_baseline_group_dataset"] for row in baseline} == {"pubmed", "nilearn"}
    assert {row["random_baseline_group_model_family"] for row in baseline} == {"cnn_t2b"}


def test_fake_generation_metrics_include_cnn_native_outputs() -> None:
    target = torch.tensor([[[[[0.0, 1.0], [0.2, 0.0]], [[0.0, 0.7], [0.0, 0.1]]]]])
    pred = target.clone()

    row = compare.evaluate_cnn_generated_sample(
        pred,
        target,
        sample_name="map-1",
        dataset_name="pubmed",
        dice_pct=90.0,
    )
    native = compare.generation_metrics(pred, target, include_voxel_auroc=False)
    row.update(native)

    assert row["name"] == "map-1"
    assert row["pearson_r"] == 1.0
    assert row["spearman_rho"] == 1.0
    assert row["dice_pct90"] == 1.0
    assert row["mse"] == 0.0
    assert row["foreground_mse"] == 0.0
    assert row["spatial_corr"] > 0.99
    assert {"top1_dice", "top5_dice", "top10_dice"} <= set(row)


def test_cli_help_parser_accepts_prompt_model_alias() -> None:
    parser = compare.build_arg_parser()
    args = parser.parse_args(["--datasets", "pubmed", "--models", "cnn_t2b_mixed", "--limit", "8", "--skip-spin"])

    assert args.datasets == ["pubmed"]
    assert args.models == ["cnn_t2b_mixed"]
    assert args.limit == 8
    assert args.skip_spin
    assert compare.normalize_model_id_for_dataset("cnn_t2b_mixed", "pubmed") == (
        "cnn_t2b_mixed_to_pubmed",
        "mixed_to_pubmed",
    )
