"""NeuroQFormer.from_state_dict_payload with the released MLP projector (config.json["projector"] == "mlp")."""

import json
import warnings
from pathlib import Path

import pytest
import torch

from neurovlm.models.base import ProjHead
from neurovlm.models.qformer import CanonicalProjection, NeuroQFormer, QFormer, ResidualMLPProjector

RUN_DIR = Path(__file__).resolve().parents[2] / "scripts/outputs/runs/final"


def _mlp_model(**kwargs) -> NeuroQFormer:
    torch.manual_seed(0)
    projector = ResidualMLPProjector(16, 8, 12, num_queries=3, width=20, depth=2, align_hidden=10, **kwargs)
    banks = {name: torch.randn(5, 8) for name in CanonicalProjection.bank_names}
    return NeuroQFormer(
        qformer=projector, proj_head_image=ProjHead(16, 24, 8), canonical_banks=banks, semantic_dim=8
    ).eval()


def _payload(model: NeuroQFormer) -> dict:
    # the layout the released packages use: config.json["qformer"] holds "projector": "mlp" and the shape
    return {"state_dict": model.state_dict(), "config": model.architecture_config()}


@pytest.mark.parametrize("head", ["tokens", "single"])
def test_mlp_payload_round_trip(head):
    model = _mlp_model(head=head)
    config = model.architecture_config()
    assert config["projector"] == "mlp" and config["width"] == 20 and config["head"] == head
    reloaded = NeuroQFormer.from_state_dict_payload(_payload(model))
    assert isinstance(reloaded.qformer, ResidualMLPProjector)
    raw = torch.randn(4, 16)
    with torch.no_grad():
        assert torch.equal(model(raw), reloaded(raw))
    assert reloaded(raw).shape == (4, 3, 12)


def test_mlp_shape_is_read_from_the_state_dict_when_the_config_has_none():
    model = _mlp_model()
    reloaded = NeuroQFormer.from_state_dict_payload({"state_dict": model.state_dict(), "config": {"projector": "mlp"}})
    assert reloaded.qformer.projector_config() == model.qformer.projector_config()


def test_default_projector_is_the_mlp():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        model = NeuroQFormer(proj_head_image=ProjHead(16, 24, 8), image_dim=16, semantic_dim=8, lm_dim=12, num_queries=3,
                             width=20, depth=1, align_hidden=10, use_canonical_projection=False)
    assert isinstance(model.qformer, ResidualMLPProjector) and model.architecture_config()["projector"] == "mlp"
    assert model(torch.randn(2, 16)).shape == (2, 3, 12)


def test_legacy_qformer_payload_builds_the_deprecated_qformer():
    torch.manual_seed(0)
    with pytest.warns(DeprecationWarning, match="QFormer is deprecated"):
        model = NeuroQFormer(image_dim=16, semantic_dim=8, lm_dim=12, num_queries=3, hidden_dim=8, num_heads=2, num_layers=1).eval()
    with pytest.warns(DeprecationWarning):
        reloaded = NeuroQFormer.from_state_dict_payload(_payload(model))
    assert isinstance(reloaded.qformer, QFormer) and "projector" not in model.architecture_config()


@pytest.mark.parametrize("package", ["neuro_qformer_canonical", "neuro_qformer_pubmed"])
def test_released_package_loads(package):
    package_dir = RUN_DIR / package
    if not (package_dir / "model.safetensors").is_file():
        pytest.skip(f"{package_dir} is not present")
    from safetensors.torch import load_file

    config = json.loads((package_dir / "config.json").read_text())
    model = NeuroQFormer.from_state_dict_payload(
        {"state_dict": load_file(package_dir / "model.safetensors"), "config": config["qformer"]}
    )
    assert isinstance(model.qformer, ResidualMLPProjector)
    assert model.qformer.projector_config() == {
        k: v for k, v in config["projector_config"].items() if k not in ("class", "params")
    }
