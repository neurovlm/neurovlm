"""Unit tests of the MLP trainer's helpers: device and seed selection, split and loader handling, model builders and
checkpoint reloads. The end-to-end training runs live in tests/integration/training/test_mlp.py."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from neurovlm.models.base import NeuroAutoEncoder, ProjHead
from neurovlm.training import mlp as tm
from neurovlm.training.mlp import (
    MLPContrastiveTrainConfig,
    MLPTextToBrainTrainConfig,
    build_mlp_contrastive,
    build_mlp_text_to_brain,
    mlp_autoencoder_from_checkpoint,
    mlp_contrastive_from_checkpoint,
    mlp_text_to_brain_from_checkpoint,
)

TINY = {"dim_neuro": 8, "dim_h0": 6, "dim_h1": 5, "dim_latent": 4}


def test_device_selection(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    assert tm._device("auto").type == "cpu" and tm._device("cpu").type == "cpu"
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    assert tm._device("auto").type == "mps"
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert tm._device("auto").type == "cuda"


def test_seed_is_reproducible():
    tm._seed(3)
    first = torch.rand(2)
    tm._seed(3)
    assert torch.equal(first, torch.rand(2))


def test_splits_accept_mappings_and_objects():
    assert tm._splits({"train": 1, "val": 2, "test": 3}) == (1, 2, 3)
    assert tm._splits(SimpleNamespace(train=1, val=2, test=3)) == (1, 2, 3)
    with pytest.raises(ValueError):
        tm._splits({"train": 1})


def test_loader_passthrough_and_size():
    dataset = TensorDataset(torch.zeros(4, 2))
    loader = DataLoader(dataset, batch_size=2)
    assert tm._loader(loader, batch_size=1, shuffle=False, seed=0, num_workers=0) is loader
    built = tm._loader(dataset, batch_size=2, shuffle=True, seed=0, num_workers=0)
    assert len(built) == 2
    assert tm._n(loader) == 4 and tm._n(dataset) == 4 and tm._n(iter([1])) is None


def test_build_text_to_brain_needs_an_autoencoder_for_custom_dimensions(monkeypatch):
    config = MLPTextToBrainTrainConfig(preset="custom", text_dim=6, hidden_dim=5, latent_dim=4, brain_dim=8)
    with pytest.raises(ValueError):
        build_mlp_text_to_brain(config)
    model = build_mlp_text_to_brain(config, autoencoder=NeuroAutoEncoder(out="logit", **TINY))
    assert model.text_projection.aligner[0].in_features == 6
    assert model.decode(torch.randn(2, 6)).shape == (2, 8)
    released = NeuroAutoEncoder(out="logit", **TINY)
    monkeypatch.setattr("neurovlm.models.base.load_model", lambda **kwargs: released)
    assert build_mlp_text_to_brain(MLPTextToBrainTrainConfig()).autoencoder is released


def test_build_contrastive_initialization_sources(monkeypatch):
    custom = dict(preset="custom", text_dim=6, text_hidden_dim=5, brain_dim=4, brain_hidden_dim=4, shared_dim=3)
    with pytest.raises(ValueError):
        build_mlp_contrastive(MLPContrastiveTrainConfig(**custom))
    scratch = build_mlp_contrastive(MLPContrastiveTrainConfig(initialize_text_from_mse=False, **custom))
    assert scratch.text_projection.aligner[0].in_features == 6
    brain, text = scratch(torch.randn(2, 4), torch.randn(2, 6))
    assert brain.shape == (2, 3) and text.shape == (2, 3)
    monkeypatch.setattr(ProjHead, "from_pretrained", staticmethod(lambda name: ProjHead(768, 512, 384, seed=0)))
    assert build_mlp_contrastive(MLPContrastiveTrainConfig()).text_projection.aligner[0].in_features == 768


def test_checkpoint_reloads_check_architecture_and_state(tmp_path, monkeypatch):
    for reload in (mlp_autoencoder_from_checkpoint, mlp_text_to_brain_from_checkpoint,
                   mlp_contrastive_from_checkpoint):
        with pytest.raises(ValueError):
            reload({"architecture": {"architecture": "Other"}})

    arch = {"architecture": "NeuroAutoEncoder", **TINY}
    with pytest.raises(ValueError):
        mlp_autoencoder_from_checkpoint({"architecture": arch, "model_state_dict": {}})
    model = NeuroAutoEncoder(out="logit", **TINY)
    path = tmp_path / "autoencoder.pt"
    torch.save({"architecture": arch, "model_state_dict": model.state_dict()}, path)
    reloaded = mlp_autoencoder_from_checkpoint(path)
    assert not reloaded.training and torch.equal(reloaded.encoder[0].weight, model.encoder[0].weight)

    text_to_brain = {"architecture": "MLPTextToBrain", "preset": "custom", "text_dim": 6, "hidden_dim": 5,
                     "latent_dim": 4}
    with pytest.raises(ValueError):
        mlp_text_to_brain_from_checkpoint({"architecture": text_to_brain, "model_state_dict": {}})
    with pytest.raises(ValueError):
        mlp_text_to_brain_from_checkpoint({"architecture": text_to_brain, "model_state_dict": {}}, autoencoder=model)
    projection = ProjHead(6, 5, 4)
    good = mlp_text_to_brain_from_checkpoint({"architecture": text_to_brain,
                                              "model_state_dict": projection.state_dict()}, autoencoder=model)
    assert not good.training and good.autoencoder is model
    retained = {"architecture": "MLPTextToBrain", "preset": "retained", "text_dim": 768, "hidden_dim": 512,
                "latent_dim": 384}
    monkeypatch.setattr("neurovlm.models.base.load_model", lambda **kwargs: model)
    with pytest.raises(ValueError):
        mlp_text_to_brain_from_checkpoint({"architecture": retained, "extra": {"initialization_source": "custom"},
                                           "model_state_dict": {}})
    assert mlp_text_to_brain_from_checkpoint({"architecture": retained,
                                              "model_state_dict": ProjHead(768, 512, 384).state_dict()}).autoencoder is model

    contrastive = {"architecture": "MLPContrastiveModel", "brain_dim": 4, "brain_hidden_dim": 4, "shared_dim": 3,
                   "text_dim": 6, "text_hidden_dim": 5}
    with pytest.raises(ValueError):
        mlp_contrastive_from_checkpoint({"architecture": contrastive, "model_state_dict": {}})
    pair = tm.MLPContrastiveModel(ProjHead(4, 4, 3), ProjHead(6, 5, 3))
    assert not mlp_contrastive_from_checkpoint({"architecture": contrastive,
                                                "model_state_dict": pair.state_dict()}).training
