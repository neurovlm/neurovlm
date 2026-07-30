from __future__ import annotations

import ast
import json
from pathlib import Path


NOTEBOOK = (
    Path(__file__).parents[1]
    / "docs"
    / "cnn"
    / "evaluation"
    / "stage4_probabilistic_latent_generation.ipynb"
)


def _source() -> tuple[dict, str]:
    notebook = json.loads(NOTEBOOK.read_text())
    source = "\n".join("".join(cell["source"]) for cell in notebook["cells"])
    return notebook, source


def test_probabilistic_notebook_is_valid_colab_with_parseable_code() -> None:
    notebook, _ = _source()
    assert notebook["nbformat"] == 4
    assert notebook["metadata"]["accelerator"] == "GPU"
    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] == "code":
            ast.parse("".join(cell["source"]), filename=f"notebook-cell-{index}")


def test_probabilistic_notebook_contains_protocol_and_outputs() -> None:
    _, source = _source()
    for required in (
        "U_DIMS = [64] if SMOKE_MODE else [32, 64, 128]",
        "BETA_VALUES = [0.01] if SMOKE_MODE else [0.001, 0.01, 0.05, 0.1]",
        "released_deterministic",
        "retrained_deterministic",
        "cvae_mean",
        "expected_one_sample",
        "average_of_k",
        "consensus_medoid",
        "oracle_best_top5_dice_diagnostic_only",
        "shuffled_control",
        "posterior_reconstruction_sample",
        "coverage_50",
        "coverage_80",
        "coverage_90",
        "coverage_95",
        "semantic_normalized_auc",
        "test_used_for_selection",
    ):
        assert required in source
    for artifact in (
        "effective_config.json",
        "provenance.json",
        "latent_standardization.pt",
        "model_architecture.json",
        "training_history.csv",
        "posterior_diagnostics.csv",
        "validation_metrics.csv",
        "test_metrics.csv",
        "sample_level_metrics.parquet",
        "checkpoint_manifest.json",
        "generated_samples",
        "final_comparison.csv",
        "final_report.md",
    ):
        assert artifact in source
