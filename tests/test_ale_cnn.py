"""Tests for ALE 3D CNN models."""

import torch

from neurovlm.gnn.ale_cnn import (
    ALE3DCNNAutoEncoder,
    ALE3DCNNDecoder,
    ALE3DCNNEncoder,
    ALEResNet3DEncoder,
)


def test_ale_3dcnn_encoder_shape():
    model = ALE3DCNNEncoder(base_channels=4, num_blocks=2, out_dim=384)
    x = torch.randn(2, 1, 13, 17, 19)

    out = model(x)

    assert out.shape == (2, 384)


def test_ale_resnet_3d_encoder_global_variants_shape():
    model = ALEResNet3DEncoder(
        base_channels=4,
        num_stages=4,
        blocks_per_stage=1,
        out_dim=384,
        use_dilation=True,
        multi_scale=True,
        global_context="attention",
    )
    x = torch.randn(2, 1, 13, 17, 19)

    out = model(x)

    assert out.shape == (2, 384)


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
        global_context="se",
    )
    x = torch.rand(2, 1, 13, 17, 19)

    recon = model(x)
    loss = (recon - x).pow(2).mean()
    loss.backward()

    assert recon.shape == x.shape
    assert all(p.grad is not None for p in model.parameters() if p.requires_grad)
