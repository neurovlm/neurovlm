from __future__ import annotations

import ast
import json
from pathlib import Path


NOTEBOOK = (
    Path(__file__).parents[1]
    / "docs"
    / "cnn"
    / "evaluation"
    / "stage4_stage3_semantic_bridge.ipynb"
)


def _notebook() -> dict:
    return json.loads(NOTEBOOK.read_text())


def _source() -> str:
    return "\n".join(
        "".join(cell["source"]) for cell in _notebook()["cells"]
    )


def test_semantic_bridge_is_a_valid_parseable_gpu_colab_notebook() -> None:
    notebook = _notebook()
    assert notebook["nbformat"] == 4
    assert notebook["metadata"]["accelerator"] == "GPU"
    assert notebook["metadata"]["colab"]["gpuType"] == "A100"
    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] == "code":
            ast.parse("".join(cell["source"]), filename=f"notebook-cell-{index}")


def test_notebook_contains_six_branches_six_paths_and_three_bridges() -> None:
    source = _source()
    for branch in (
        "mixed_to_pubmed",
        "mixed_to_nilearn",
        "mixed_to_neurovault",
        "pubmed",
        "nilearn",
        "neurovault",
    ):
        assert branch in source
    for path in (
        "direct_baseline",
        "stage3_text_bridge",
        "stage3_brain_bridge_oracle",
        "shared_bridge_dual_supervision",
        "concatenated_text_semantic",
        "residual_direct_plus_semantic",
    ):
        assert path in source
    for architecture in (
        "mlp_512",
        "deep_mlp_1024",
        "residual_mlp_1024",
    ):
        assert architecture in source


def test_notebook_has_strict_provenance_controls_and_selection() -> None:
    source = _source()
    for requirement in (
        "Google Drive",
        "EXPECTED_COMMIT",
        "BF16_AVAILABLE",
        "seed_everything",
        "RESUME = True",
        "split_fingerprint",
        "autoencoder_identity",
        "stage3_identity",
        "text_cache_identity",
        "fixed_derangement",
        "test_used_for_selection",
        "FINAL_CHECKPOINT_ROLE",
        "best_top5_dice.pt",
        "best_spatial_correlation.pt",
        "best_latent_explained_variance.pt",
        "best_semantic_normalized_auc.pt",
        "last.pt",
    ):
        assert requirement in source


def test_notebook_records_required_metrics_outputs_and_interpretation() -> None:
    source = _source()
    for metric in (
        "raw_latent_mse",
        "standardized_latent_mse",
        "latent_cosine",
        "latent_variance_ratio",
        "latent_norm_ratio",
        "global_explained_variance",
        "mean_per_dimension_r_squared",
        "nearest_real_latent_distance",
        "decoded_mse",
        "foreground_mse",
        "spatial_corr",
        "top5_dice",
        "semantic_normalized_auc",
        "stage3_text_brain_matched_cosine",
        "stage3_text_brain_shuffled_cosine",
    ):
        assert metric in source
    for artifact in (
        "provenance.json",
        "bridge_architecture_configs.json",
        "training_history.csv",
        "branch_metrics.csv",
        "text_versus_brain_semantic_comparison.csv",
        "oracle_gap_plots.png",
        "latent_variance_plots.png",
        "generated_examples.png",
        "final_comparison.csv",
        "final_comparison.json",
        "final_report.md",
    ):
        assert artifact in source
    for conclusion in (
        "cross-modal alignment remains insufficient",
        "discard raw spatial information",
        "direct SPECTER2-to-AE projector is the main bottleneck",
        "supporting complementary information",
    ):
        assert conclusion in source


def test_architecture_and_loss_axes_are_not_pooled() -> None:
    source = _source()
    assert 'experiment_axis": "primary_architecture"' in source
    assert 'experiment_axis": "loss_sensitivity"' in source
    assert "primary_raw_decoded" in source
    assert "standardized_decoded" in source
    assert "standardized_cosine_decoded" in source
    assert "standardized_cosine_norm_decoded" in source
    assert 'validation_frame["experiment_axis"] == "primary_architecture"' in source
