from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch import nn

import neurovlm.core.runtime as runtime_module
from neurovlm import NeuroVLMRuntime, load_pipeline
from neurovlm.cnn.models import ATLAS_FREE_VOLUME_SHAPE, MLP_MASKER_VOXEL_COUNT


class _MLPAE(nn.Module):
    def __init__(self, logits: bool = False):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(MLP_MASKER_VOXEL_COUNT, 3))
        decoder = [nn.Linear(3, MLP_MASKER_VOXEL_COUNT)]
        if not logits:
            decoder.append(nn.Sigmoid())
        self.decoder = nn.Sequential(*decoder)

    def forward(self, value):
        return self.decoder(self.encoder(value))


class _CNNAE(nn.Module):
    def __init__(self):
        super().__init__()
        voxels = 36 * 45 * 38
        self.encoder = nn.Sequential(nn.Flatten(), nn.Linear(voxels, 3))
        self.decoder = nn.Sequential(nn.Linear(3, voxels), nn.Unflatten(1, (1, 36, 45, 38)))

    def forward(self, value):
        return self.decoder(self.encoder(value))


class _CNNContrastive(nn.Module):
    def __init__(self):
        super().__init__()
        self.brain = nn.Linear(1, 4)
        self.text = nn.Linear(768, 4)

    def encode_brain(self, value):
        pooled = value.mean((1, 2, 3, 4), keepdim=False).unsqueeze(1)
        return torch.nn.functional.normalize(self.brain(pooled), dim=1)

    def encode_text(self, value):
        return torch.nn.functional.normalize(self.text(value), dim=1)


class _CNNT2B(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(768, 1)

    def forward(self, value):
        return self.proj(value).reshape(-1, 1, 1, 1, 1).expand(-1, 1, 36, 45, 38)


def _fake_load_model(**kwargs):
    family = str(kwargs.get("family", "mlp"))
    task = str(kwargs["task"])
    variant = str(kwargs.get("variant", ""))
    if family == "cnn":
        if task == "autoencoder":
            return _CNNAE()
        if task == "contrastive":
            return _CNNContrastive()
        if task == "text_to_brain":
            return _CNNT2B()
    if task == "autoencoder":
        return _MLPAE()
    if task == "contrastive":
        return nn.Linear(3 if variant == "brain" else 768, 4)
    if task == "text_to_brain":
        return nn.Linear(768, 3)
    if task == "text_encoder":
        return lambda _: torch.ones(1, 768)
    raise AssertionError(kwargs)


@pytest.fixture(autouse=True)
def _offline_models(monkeypatch):
    monkeypatch.setattr(runtime_module, "load_model", _fake_load_model)


def test_public_runtime_mlp_default_and_tensor_methods():
    pipe = load_pipeline()
    assert isinstance(pipe, NeuroVLMRuntime)
    assert pipe.metadata.family == "mlp"
    assert pipe.metadata.task == "autoencoder"
    flat = torch.rand(2, MLP_MASKER_VOXEL_COUNT)
    latent = pipe.encode(flat)
    assert pipe.decode(latent).shape == flat.shape
    assert pipe.reconstruct(flat).shape == flat.shape
    assert not pipe.model.training
    assert not any(parameter.requires_grad for parameter in pipe.model.parameters())


def test_mlp_decode_converts_standardized_logits_to_probabilities(monkeypatch):
    monkeypatch.setattr(runtime_module, "_released_mlp", lambda spec: (_MLPAE(logits=True), None, None, None))
    pipe = load_pipeline(task="autoencoder")
    decoded = pipe.decode(torch.randn(2, 3))
    assert torch.all((decoded >= 0) & (decoded <= 1))


@pytest.mark.parametrize("domain", ["pubmed", "nilearn", "neurovault"])
def test_cnn_domain_switch_is_visible_and_mixed_by_default(domain):
    pipe = load_pipeline(family="cnn", task="contrastive", domain=domain)
    assert pipe.metadata.domain == domain
    assert pipe.metadata.variant == "mixed_baseline"
    assert pipe.metadata.loader_variant == f"mixed_to_{domain}"
    volume = torch.rand(2, 1, *ATLAS_FREE_VOLUME_SHAPE)
    text = torch.randn(3, 768)
    assert pipe.encode_brain(volume).shape == (2, 4)
    assert pipe.encode_text(text).shape == (3, 4)
    assert torch.allclose(pipe.encode_brain(volume).norm(dim=1), torch.ones(2), atol=1e-6)
    assert pipe.similarity(volume, text).shape == (2, 3)


def test_finetuned_and_text_to_brain_generation_are_explicit():
    pipe = load_pipeline(
        family="cnn", task="text_to_brain", domain="pubmed", variant="finetuned"
    )
    assert pipe.metadata.variant == "finetuned"
    assert pipe.metadata.loader_variant == "pubmed"
    assert pipe.generate(torch.randn(2, 768)).shape == (2, 1, *ATLAS_FREE_VOLUME_SHAPE)


def test_raw_text_uses_injected_encoder_without_loading_specter():
    pipe = load_pipeline(
        task="contrastive", text_encoder=lambda text: torch.ones(len(text), 768)
    )
    output = pipe.encode_text(["one", "two"])
    assert output.shape == (2, 4)


def test_shape_and_task_errors_are_precise():
    pipe = load_pipeline()
    with pytest.raises(ValueError, match="MLP brain input"):
        pipe.reconstruct(torch.rand(2, 10))
    with pytest.raises(RuntimeError, match="does not support"):
        pipe.generate(torch.rand(1, 768))
    with pytest.raises(ValueError, match="domain is required"):
        load_pipeline(family="cnn", task="contrastive")


def test_from_run_resolves_best_checkpoint_and_uses_reloader(monkeypatch, tmp_path: Path):
    checkpoint = tmp_path / "checkpoints" / "best.pt"
    checkpoint.parent.mkdir()
    checkpoint.touch()
    calls = []
    monkeypatch.setattr(
        "neurovlm.training.autoencoder.autoencoder_from_checkpoint",
        lambda path, device: calls.append((Path(path), device)) or _CNNAE(),
    )
    pipe = load_pipeline(family="cnn", task="autoencoder", from_run=tmp_path)
    assert pipe.metadata.source == "from_run"
    assert pipe.metadata.checkpoint == str(checkpoint.resolve())
    assert calls[0][0] == checkpoint


def test_mlp_text_to_brain_is_flat_and_contrastive_is_normalized():
    generator = load_pipeline(task="text_to_brain")
    assert generator.generate(torch.randn(2, 768)).shape == (2, MLP_MASKER_VOXEL_COUNT)
    contrastive = load_pipeline(task="contrastive")
    flat = torch.rand(2, MLP_MASKER_VOXEL_COUNT)
    brain = contrastive.encode_brain(flat)
    text = contrastive.encode_text(torch.randn(2, 768))
    assert torch.allclose(brain.norm(dim=1), torch.ones(2), atol=1e-6)
    assert torch.allclose(text.norm(dim=1), torch.ones(2), atol=1e-6)
