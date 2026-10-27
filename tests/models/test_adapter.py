"""The text-to-brain NeuroAdapter: residual blocks interleaved with a pretrained MLP decoder."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from neurovlm.models.adapter import InterleavedDecoderAdapter, InterleavedResidualBlock, LogitCalibrator
from neurovlm.models.base import NeuroAutoEncoder, ProjHead

DIMS = {"dim_neuro": 32, "dim_h0": 12, "dim_h1": 10, "dim_latent": 8}


def test_residual_block_starts_as_the_identity():
    block = InterleavedResidualBlock(6, hidden_dim=4)
    x = torch.randn(3, 6)
    assert torch.allclose(block(x), x)
    with torch.no_grad():
        block.fc1.weight.fill_(0.1)
    assert not torch.allclose(block(x), x)


def test_logit_calibrator_variants():
    learned = LogitCalibrator(5)
    assert learned.bias.shape == (5,) and isinstance(learned.log_temperature, nn.Parameter)
    fixed = LogitCalibrator(5, per_feature_bias=False, learn_temperature=False)
    assert fixed.bias.shape == (1,) and "log_temperature" in dict(fixed.named_buffers())
    logits = torch.randn(2, 5)
    assert torch.allclose(learned(logits), logits) and torch.allclose(fixed(logits), logits)
    with torch.no_grad():
        learned.log_temperature.fill_(torch.log(torch.tensor(2.0)))
    assert torch.allclose(learned(logits), logits / 2)


def test_adapter_wraps_the_pretrained_decoder_as_the_identity_at_init():
    autoencoder = NeuroAutoEncoder(seed=0, out="probability", **DIMS)
    head = ProjHead(16, 12, 8, seed=0)
    adapter = InterleavedDecoderAdapter(autoencoder, head, hidden_dim=6)
    x = torch.randn(4, 16)
    logits = adapter(x)
    assert logits.shape == (4, 32)
    with torch.no_grad():
        expected = autoencoder.decoder[:-1](head(x))  # the Sigmoid is stripped
    assert torch.allclose(logits, expected, atol=1e-5)
    assert torch.allclose(InterleavedDecoderAdapter(autoencoder, ProjHead(16, 12, 8, seed=0), return_logits=False)(x),
                          torch.sigmoid(expected), atol=1e-5)
    assert not any(p.requires_grad for module in (adapter.dec0, adapter.dec1, adapter.dec2)
                   for p in module.parameters())
    assert all(p.requires_grad for p in head.parameters())
    inserted = list(adapter.inserted_parameters())
    assert len(list(adapter.trainable_parameters())) == len(inserted) + len(list(head.parameters()))
    assert sum(p.numel() for p in inserted) > 0


def test_adapter_freeze_and_calibrator_options():
    autoencoder = NeuroAutoEncoder(seed=0, out="logit", **DIMS)
    head = ProjHead(16, 12, 8, seed=0)
    adapter = InterleavedDecoderAdapter(autoencoder, head, freeze_proj_head=True, freeze_pretrained_decoder=False,
                                        use_output_calibrator=False)
    assert isinstance(adapter.output_calibrator, nn.Identity)
    assert not any(p.requires_grad for p in head.parameters())
    assert all(p.requires_grad for p in adapter.dec0.parameters())
    assert adapter(torch.randn(2, 16)).shape == (2, 32)


def test_adapter_rejects_a_decoder_without_the_expected_layout():
    short = SimpleNamespace(decoder=nn.Sequential(nn.Linear(8, 32), nn.Sigmoid()))
    with pytest.raises(ValueError):
        InterleavedDecoderAdapter(short, ProjHead(16, 12, 8))
