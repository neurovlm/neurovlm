from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).parents[1] / "docs" / "cnn" / "evaluation"
NOTEBOOKS = (
    "stage4_standardized_latent_ablation.ipynb",
    "stage4_sparse_spatial_loss_ablation.ipynb",
    "stage4_projector_architecture_optimization_sweep.ipynb",
    "stage4_joint_ae_projector_finetuning.ipynb",
    "stage4_stage3_semantic_bridge.ipynb",
    "stage4_probabilistic_latent_generation.ipynb",
)


def _source(name: str) -> str:
    notebook = json.loads((ROOT / name).read_text())
    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] == "code":
            ast.parse("".join(cell["source"]), filename=f"{name}-cell-{index}")
    return "\n".join("".join(cell["source"]) for cell in notebook["cells"])


@pytest.mark.parametrize("name", NOTEBOOKS)
def test_stage4_experiments_exclude_domain_finetuned_ae_branches(name: str) -> None:
    source = _source(name)
    for branch in (
        "mixed_to_pubmed",
        "mixed_to_nilearn",
        "mixed_to_neurovault",
    ):
        assert branch in source
    for specialized in (
        '"ae_variant": "pubmed"',
        '"ae_variant": "nilearn"',
        '"ae_variant": "neurovault"',
        '"stage1": "1B"',
    ):
        assert specialized not in source


@pytest.mark.parametrize("name", NOTEBOOKS)
def test_stage4_experiments_enable_blackwell_safe_throughput_settings(
    name: str,
) -> None:
    source = _source(name)
    assert 'torch.set_float32_matmul_precision("high")' in source
    if name != "stage4_standardized_latent_ablation.ipynb":
        assert "pin_memory=" in source
        assert "persistent_workers=" in source
    else:
        assert "train_stage4_ablation" in source
    assert "PREFETCH_FACTOR = 4" in source
    assert "NUM_WORKERS = 8 if IN_COLAB else 0" in source
