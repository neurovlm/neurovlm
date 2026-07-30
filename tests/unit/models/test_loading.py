"""Tests for the installable atlas-free CNN model API."""

from __future__ import annotations

import torch
from torch import nn

from neurovlm.cnn.architectures import ALE3DCNNAutoEncoder
from neurovlm.models import load_model
from neurovlm.resources import loaders as rr


def test_load_model_routes_cnn_names_to_resource_loaders(monkeypatch) -> None:
    sentinels = {
        "ae": nn.Identity(),
        "contrastive": nn.Identity(),
        "t2b": nn.Identity(),
    }
    calls = []

    monkeypatch.setattr(
        rr,
        "_load_cnn_autoencoder",
        lambda variant: calls.append(("ae", variant)) or sentinels["ae"],
    )
    monkeypatch.setattr(
        rr,
        "_load_cnn_contrastive",
        lambda variant: calls.append(("contrastive", variant)) or sentinels["contrastive"],
    )
    monkeypatch.setattr(
        rr,
        "_load_cnn_text_to_brain",
        lambda variant: calls.append(("t2b", variant)) or sentinels["t2b"],
    )

    assert load_model("autoencoder_cnn") is sentinels["ae"]
    assert load_model("autoencoder_cnn_nilearn") is sentinels["ae"]
    assert load_model("contrastive_cnn_mixed_to_pubmed") is sentinels["contrastive"]
    assert load_model("text_to_brain_cnn_neurovault") is sentinels["t2b"]
    assert calls == [
        ("ae", "mixed"),
        ("ae", "nilearn"),
        ("contrastive", "mixed_to_pubmed"),
        ("t2b", "neurovault"),
    ]


def test_cnn_autoencoder_loader_uses_fixed_repo_and_restricted_unpickler(monkeypatch) -> None:
    template = ALE3DCNNAutoEncoder(
        output_shape=(8, 8, 8),
        base_channels=4,
        num_blocks=2,
        latent_dim=5,
        dropout=0.0,
    )
    payload = {
        "config": {
            "model": {
                "base_channels": 4,
                "num_blocks": 2,
                "latent_dim": 5,
                "dropout": 0.0,
                "norm": "group",
                "pooling": "max",
                "encoder_arch": "plain",
            }
        },
        "target_shape": [8, 8, 8],
        "model": template.state_dict(),
    }
    downloads = []
    loads = []

    def fake_download(repo_id: str, filename: str, repo_type: str = "dataset") -> str:
        downloads.append((repo_id, filename, repo_type))
        return "/trusted/mixed_ae.pt"

    def fake_torch_load(path, *, map_location, weights_only):
        loads.append((path, map_location, weights_only))
        return payload

    monkeypatch.setattr(rr, "_download_from_hf", fake_download)
    monkeypatch.setattr(torch, "load", fake_torch_load)
    model = rr._load_cnn_autoencoder("mixed")

    assert tuple(model(torch.zeros(1, 1, 8, 8, 8)).shape) == (1, 1, 8, 8, 8)
    assert downloads == [("neurovlm/3d_cnn", "mixed_ae.pt", "model")]
    assert loads == [("/trusted/mixed_ae.pt", "cpu", True)]
    assert not model.training
    assert not any(parameter.requires_grad for parameter in model.parameters())


def test_cnn_resource_loaders_reject_unknown_variants() -> None:
    for loader in (rr._load_cnn_autoencoder, rr._load_cnn_contrastive, rr._load_cnn_text_to_brain):
        try:
            loader("unknown")
        except ValueError as error:
            assert "Unknown CNN" in str(error)
        else:  # pragma: no cover
            raise AssertionError("unknown CNN variant was accepted")
