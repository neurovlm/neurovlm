"""Unit tests of the MLP evaluation helpers: batch unpacking, metadata, AUROC guards, batch limits."""

from __future__ import annotations

import math

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from neurovlm.evaluation import mlp as em
from neurovlm.models.base import NeuroAutoEncoder, ProjHead
from neurovlm.training.mlp import MLPContrastiveModel, MLPTextToBrainModel

TINY = {"dim_neuro": 8, "dim_h0": 6, "dim_h1": 5, "dim_latent": 4}


def test_loader_passthrough():
    dataset = TensorDataset(torch.zeros(4, 2))
    loader = DataLoader(dataset, batch_size=2)
    assert em._loader(loader, 2, 0) is loader
    assert len(em._loader(dataset, 2, 0)) == 2


def test_mapping_tensor_and_metadata():
    with pytest.raises(KeyError):
        em._mapping_tensor({"a": 1}, ("brain",))
    assert em._metadata(torch.zeros(2), 2, 5) == (["unknown", "unknown"], ["5", "6"])
    assert em._metadata({"source": "pubmed", "sample_id": "s"}, 2, 0) == (["pubmed", "pubmed"], ["s", "s"])
    assert em._metadata({"id": [7, 8]}, 2, 0) == (["unknown", "unknown"], ["7", "8"])
    assert em._metadata({}, 2, 3) == (["unknown", "unknown"], ["3", "4"])


def test_batch_input_helpers():
    x = torch.zeros(2, 3)
    assert em._autoencoder_input((x, 1)) is x and em._autoencoder_input(x) is x
    assert em._autoencoder_input({"flatmap": x}) is x
    with pytest.raises(TypeError):
        em._autoencoder_input("x")
    assert em._paired_input((x, x)) == (x, x)
    with pytest.raises(TypeError):
        em._paired_input("x")
    with pytest.raises(TypeError):
        em._paired_input((x,))
    text, brain = em._paired_input({"text": x, "brain_latent": x}, latent_brain=True)
    assert text is x and brain is x
    text, brain = em._paired_input({"specter": x, "y": x})
    assert text is x and brain is x


def test_safe_auroc_handles_a_single_class():
    assert math.isnan(em._safe_auroc(torch.zeros(4), torch.rand(4)))
    assert em._safe_auroc(torch.tensor([1.0, 0.0]), torch.tensor([0.9, 0.1])) == 1.0


def test_evaluations_respect_batch_limits_and_dataloaders():
    autoencoder = NeuroAutoEncoder(seed=0, out="logit", **TINY)
    brains = TensorDataset((torch.rand(6, 8) > 0.5).float())
    loader = DataLoader(brains, batch_size=2)
    assert em.evaluate_mlp_autoencoder(autoencoder, loader, max_batches=1).n == 2
    with pytest.raises(RuntimeError):
        em.evaluate_mlp_autoencoder(autoencoder, loader, max_batches=0)

    contrastive = MLPContrastiveModel(ProjHead(4, 4, 3), ProjHead(6, 5, 3))
    pairs = TensorDataset(torch.randn(6, 6), torch.randn(6, 4))
    assert em.evaluate_mlp_contrastive(contrastive, DataLoader(pairs, batch_size=3), max_batches=1).n == 3
    assert em.evaluate_mlp_brain_to_text_retrieval(contrastive, pairs, batch_size=3).n == 6
    with pytest.raises(ValueError):
        em.evaluate_mlp_contrastive(contrastive, pairs, temperature=0)
    with pytest.raises(RuntimeError):
        em.evaluate_mlp_contrastive(contrastive, pairs, max_batches=0)

    text_to_brain = MLPTextToBrainModel(ProjHead(6, 5, 4), autoencoder)
    raw_targets = TensorDataset(torch.randn(6, 6), (torch.rand(6, 8) > 0.5).float())
    evaluation = em.evaluate_mlp_text_to_brain(text_to_brain, raw_targets, batch_size=3, max_batches=1)
    assert evaluation.n == 3 and "loss" in evaluation.summary
    latent_targets = TensorDataset(torch.randn(6, 6), torch.randn(6, 4))
    assert em.evaluate_mlp_text_to_brain(text_to_brain, latent_targets, batch_size=3).n == 6
    with pytest.raises(RuntimeError):
        em.evaluate_mlp_text_to_brain(text_to_brain, raw_targets, max_batches=0)
    with pytest.raises(ValueError):
        em.evaluate_mlp_text_to_brain(ProjHead(6, 5, 4), raw_targets)
