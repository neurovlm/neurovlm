"""Tests for ALE 3D CNN models."""

import sys
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
THREEDCNN = REPO_ROOT / "experiments" / "3dcnn"
if str(THREEDCNN) not in sys.path:
    sys.path.insert(0, str(THREEDCNN))

from atlas_free_cnn.training.ale_cnn import (
    ALE3DCNNAutoEncoder,
    ALE3DCNNDecoder,
    ALE3DCNNEncoder,
    ALEResNet3DEncoder,
    validate_retained_resnet_architecture,
)


def test_ale_3dcnn_encoder_shape():
    model = ALE3DCNNEncoder(base_channels=4, num_blocks=2, out_dim=384)
    x = torch.randn(2, 1, 13, 17, 19)

    out = model(x)

    assert out.shape == (2, 384)


def test_retained_resnet48_multiscale_attention_shape():
    model = ALEResNet3DEncoder(
        base_channels=4,
        num_stages=4,
        blocks_per_stage=1,
        out_dim=384,
        multi_scale=True,
        global_context="attention",
    )
    x = torch.randn(2, 1, 13, 17, 19)

    out = model(x)

    assert out.shape == (2, 384)


def test_production_resnet_rejects_discarded_variants():
    with pytest.raises(ValueError, match="Only the retained ResNet48"):
        validate_retained_resnet_architecture(
            base_channels=64,
            num_stages=4,
            blocks_per_stage=2,
            multi_scale=True,
            global_context="attention",
        )


def test_ale_3dcnn_decoder_exact_output_shape():
    model = ALE3DCNNDecoder(
        output_shape=(13, 17, 19),
        latent_dim=384,
        base_channels=4,
        num_blocks=2,
    )
    z = torch.randn(2, 384)

    out = model(z)

    assert out.shape == (2, 1, 13, 17, 19)


def test_ale_3dcnn_autoencoder_backward():
    model = ALE3DCNNAutoEncoder(
        output_shape=(13, 17, 19),
        base_channels=4,
        num_blocks=2,
        latent_dim=384,
        dropout=0.0,
    )
    x = torch.rand(2, 1, 13, 17, 19)

    recon = model(x)
    loss = (recon - x).pow(2).mean()
    loss.backward()

    assert recon.shape == x.shape
    assert all(p.grad is not None for p in model.parameters() if p.requires_grad)


def test_ale_resnet_autoencoder_backward():
    model = ALE3DCNNAutoEncoder(
        output_shape=(13, 17, 19),
        base_channels=4,
        num_blocks=3,
        latent_dim=384,
        dropout=0.0,
        encoder_arch="resnet",
        blocks_per_stage=1,
        multi_scale=True,
        global_context="attention",
    )
    x = torch.rand(2, 1, 13, 17, 19)

    recon = model(x)
    loss = (recon - x).pow(2).mean()
    loss.backward()

    assert recon.shape == x.shape
    assert all(p.grad is not None for p in model.parameters() if p.requires_grad)
