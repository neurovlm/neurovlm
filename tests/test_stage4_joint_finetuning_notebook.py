from __future__ import annotations

import ast
import json
from pathlib import Path


NOTEBOOK = (
    Path(__file__).parents[1]
    / "docs"
    / "cnn"
    / "evaluation"
    / "stage4_joint_ae_projector_finetuning.ipynb"
)


def _notebook() -> dict:
    return json.loads(NOTEBOOK.read_text(encoding="utf-8"))


def _source() -> str:
    return "\n".join(
        "".join(cell["source"]) for cell in _notebook()["cells"]
    )


def test_joint_finetuning_notebook_is_valid_parseable_colab() -> None:
    notebook = _notebook()
    assert notebook["nbformat"] == 4
    assert notebook["metadata"]["accelerator"] == "GPU"
    assert notebook["metadata"]["colab"]["gpuType"] == "A100"
    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] == "code":
            ast.parse(
                "".join(cell["source"]),
                filename=f"stage4-joint-cell-{index}",
            )


def test_notebook_contains_every_variant_and_required_control() -> None:
    source = _source()
    for variant in (
        "projector_only_baseline",
        "projector_plus_decoder_output",
        "projector_plus_last_decoder_block",
        "projector_plus_decoder_seed",
        "projector_plus_seed_and_last_block",
        "projector_plus_encoder_head_and_decoder",
        "latent_noise_decoder_adaptation",
    ):
        assert variant in source
    for control in (
        "true_latent_bypass_control",
        "TINY_OVERFIT_N = 32",
        "latent_noise_reconstruction_test",
        "shuffled_text_pair_control",
        "zero_noise_replay",
        "LATENT_NOISE_SCALES = [0.25, 0.5, 1.0, 2.0]",
    ):
        assert control in source


def test_notebook_contains_losses_safety_metrics_and_identity_bindings() -> None:
    source = _source()
    for required in (
        "latent_alignment",
        "generation_image_loss",
        "AE_reconstruction_replay",
        "decoder_output_distillation",
        "parameter_distance_regularization",
        "MAXIMUM_AE_TOP5_DEGRADATION_PERCENT = 5.0",
        "satisfies_1_percent",
        "satisfies_2_percent",
        "satisfies_5_percent",
        "latent_variance_ratio",
        "latent_norm_ratio",
        "explained_variance",
        "semantic_normalized_recall_auc",
        "gradient_norm_",
        "update_norm_",
        "parameter_drift_",
        "peak_gpu_memory_bytes",
        "epoch_time_seconds",
        "original_ae_identity",
        "starting_ae_state_identity",
        "current_trainable_module_identity",
        "text_cache_identity",
        "split_fingerprints",
        "exact_unfrozen_parameter_names",
        "optimizer_group_settings",
        '"test_used_for_selection": False',
    ):
        assert required in source


def test_notebook_writes_all_requested_outputs_and_never_saves_released_models() -> None:
    source = _source()
    for artifact in (
        "effective_config.json",
        "provenance.json",
        "trainable_parameter_manifest.csv",
        "histories.csv",
        "generation_metrics.csv",
        "ae_retention_metrics.csv",
        "safety_rule_decisions.csv",
        "parameter_drift_plots.png",
        "latent_noise_robustness_plots.png",
        "generated_maps",
        "comparison.csv",
        "comparison.json",
        "final_report.md",
        "best_generation_top5_safe.pt",
        "best_semantic_auc_safe.pt",
        "best_ae_preserving.pt",
        "last.pt",
    ):
        assert artifact in source
    assert '"released_models_overwritten": False' in source
    assert "provider.test" in source
    assert source.index("validation_selected_runs.json") < source.rindex(
        "provider.test"
    )
