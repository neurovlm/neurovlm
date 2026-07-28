from __future__ import annotations

import ast
import json
from pathlib import Path


NOTEBOOK = (
    Path(__file__).parents[1]
    / "docs"
    / "cnn"
    / "evaluation"
    / "stage4_projector_architecture_optimization_sweep.ipynb"
)


def _notebook() -> dict:
    return json.loads(NOTEBOOK.read_text())


def test_projector_sweep_is_a_valid_colab_notebook_with_parseable_code() -> None:
    notebook = _notebook()
    assert notebook["nbformat"] == 4
    assert notebook["metadata"]["accelerator"] == "GPU"
    assert notebook["metadata"]["colab"]["gpuType"] == "A100"
    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] == "code":
            ast.parse("".join(cell["source"]), filename=f"notebook-cell-{index}")


def test_projector_sweep_contains_every_required_architecture_and_control() -> None:
    source = "\n".join(
        "".join(cell["source"]) for cell in _notebook()["cells"]
    )
    for architecture in (
        "retained_mlp",
        "wider_mlp",
        "deep_mlp",
        "layernorm_deep",
        "residual_1024",
        "residual_2048",
        "gated_residual",
    ):
        assert architecture in source
    for control in (
        "FAST_SWEEP = True",
        "FULL_SWEEP = False",
        "NEUROVLM_PINNED_COMMIT",
        "TINY_OVERFIT_N = 32",
        "MIXED_PRECISION_DTYPE",
        "test_used_for_selection",
        "RUN_SECONDARY_STANDARDIZED_SWEEP",
    ):
        assert control in source


def test_projector_sweep_logs_required_metrics_and_outputs() -> None:
    source = "\n".join(
        "".join(cell["source"]) for cell in _notebook()["cells"]
    )
    for metric in (
        "raw_latent_mse",
        "reconstruction_mse",
        "latent_variance_ratio",
        "latent_norm_ratio",
        "global_explained_variance",
        "mean_per_dimension_r_squared",
        "highest_target_variance_quartile_mean_r_squared",
        "target_prediction_per_dimension_variance_correlation",
        "spatial_corr",
        "top5_dice",
        "foreground_mse",
        "semantic_normalized_auc",
        "total_gradient_norm",
        "parameter_update_norm",
        "zero_gradient_percent",
        "activation_mean",
        "projected_latent_std",
        "peak_gpu_memory_bytes",
        "epoch_time_seconds",
    ):
        assert metric in source
    for artifact in (
        "sweep_config.json",
        "effective_config.json",
        "provenance.json",
        "architecture_definitions.json",
        "training_history.csv",
        "validation_leaderboard.csv",
        "pareto_front.csv",
        "test_results_finalists_only.csv",
        "checkpoint_manifest.json",
        "parameter_count_table.csv",
        "time_memory_table.csv",
        "latent_collapse_plots.png",
        "architecture_comparison_plots.png",
        "final_report.md",
    ):
        assert artifact in source


def test_primary_loss_is_raw_and_test_is_finalist_only() -> None:
    source = "\n".join(
        "".join(cell["source"]) for cell in _notebook()["cells"]
    )
    assert "optimized_latent_mse = raw_latent_mse" in source
    assert "prediction_volume.float(), target_volume.float()" in source
    assert "if FULL_SWEEP:" in source
    assert "not_evaluated_in_fast_sweep" in source
    assert "Lion(" not in source
