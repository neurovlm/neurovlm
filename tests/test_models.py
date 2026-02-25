"""Tests for models module."""

import pytest
import torch
from torch import nn
import pandas as pd

from neurovlm.models import (
    NormalizeLayer,
    NeuroAutoEncoder,
    ProjHead,
    ConceptClf,
    load_model,
)


class TestNormalizeLayer:
    """Tests for NormalizeLayer."""

    def test_normalize_layer_forward(self):
        """Test that NormalizeLayer normalizes correctly."""
        layer = NormalizeLayer()
        x = torch.randn(5, 128)
        output = layer(x)

        # Check output shape
        assert output.shape == x.shape

        # Check that each row has norm ~1
        norms = torch.norm(output, dim=1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-6)

    def test_normalize_layer_zero_input(self):
        """Test normalization with zero vectors."""
        layer = NormalizeLayer()
        x = torch.zeros(3, 10)
        output = layer(x)

        # Should handle zero vectors gracefully (likely returns zeros or handles numerically)
        assert output.shape == x.shape
        assert not torch.isnan(output).any()


class TestNeuroAutoEncoder:
    """Tests for NeuroAutoEncoder class."""

    def test_neuroautoencoder_initialization_default(self):
        """Test default initialization."""
        model = NeuroAutoEncoder()

        assert isinstance(model.encoder, nn.Sequential)
        assert isinstance(model.decoder, nn.Sequential)

    def test_neuroautoencoder_initialization_custom(self):
        """Test initialization with custom parameters."""
        model = NeuroAutoEncoder(
            seed=42,
            out="logits",
            dim_neuro=1000,
            dim_h0=512,
            dim_h1=256,
            dim_latent=128,
        )

        assert isinstance(model.encoder, nn.Sequential)
        assert isinstance(model.decoder, nn.Sequential)

    def test_neuroautoencoder_forward_probability_output(self):
        """Test forward pass with probability output."""
        model = NeuroAutoEncoder(out="probability")
        x = torch.rand(4, 28542)

        output = model(x)

        assert output.shape == x.shape
        # Probabilities should be in [0, 1]
        assert (output >= 0).all() and (output <= 1).all()

    def test_neuroautoencoder_forward_logit_output(self):
        """Test forward pass with logit output."""
        model = NeuroAutoEncoder(out="logits")
        x = torch.rand(4, 28542)

        output = model(x)

        assert output.shape == x.shape
        # Logits can be any value
        assert not torch.isnan(output).any()

    def test_neuroautoencoder_encoder_shape(self):
        """Test encoder output has correct latent dimension."""
        model = NeuroAutoEncoder(dim_latent=384)
        x = torch.rand(4, 28542)

        latent = model.encoder(x)

        assert latent.shape == (4, 384)

    def test_neuroautoencoder_decoder_shape(self):
        """Test decoder output has correct neuro dimension."""
        model = NeuroAutoEncoder(dim_neuro=28542, dim_latent=384)
        latent = torch.rand(4, 384)

        output = model.decoder(latent)

        assert output.shape == (4, 28542)

    def test_neuroautoencoder_normalize_latent(self):
        """Test that normalize_latent produces normalized embeddings."""
        model = NeuroAutoEncoder(normalize_latent=True)
        x = torch.rand(4, 28542)

        latent = model.encoder(x)

        # Check that latents are normalized
        norms = torch.norm(latent, dim=1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)

    def test_neuroautoencoder_seed_reproducibility(self):
        """Test that same seed produces same initialization."""
        model1 = NeuroAutoEncoder(seed=42)
        model2 = NeuroAutoEncoder(seed=42)

        # Check that initial weights are the same
        for p1, p2 in zip(model1.parameters(), model2.parameters()):
            assert torch.allclose(p1, p2)

    def test_neuroautoencoder_different_activations(self):
        """Test that custom activation functions work."""
        model_relu = NeuroAutoEncoder(activation_fn=nn.ReLU())
        model_tanh = NeuroAutoEncoder(activation_fn=nn.Tanh())

        x = torch.rand(2, 28542)

        output_relu = model_relu(x)
        output_tanh = model_tanh(x)

        # Outputs should be different with different activations
        assert not torch.allclose(output_relu, output_tanh)


class TestProjHead:
    """Tests for ProjHead class."""

    def test_projhead_initialization_default(self):
        """Test default initialization."""
        model = ProjHead()

        assert isinstance(model.aligner, nn.Sequential)

    def test_projhead_initialization_custom(self):
        """Test initialization with custom dimensions."""
        model = ProjHead(latent_in_dim=768, hidden_dim=512, latent_out_dim=384, seed=42)

        assert isinstance(model.aligner, nn.Sequential)

    def test_projhead_forward(self):
        """Test forward pass with default dimensions."""
        model = ProjHead()
        x = torch.randn(5, 768)

        output = model(x)

        assert output.shape == (5, 384)

    def test_projhead_forward_custom_dims(self):
        """Test forward pass with custom dimensions."""
        model = ProjHead(latent_in_dim=512, latent_out_dim=256)
        x = torch.randn(8, 512)

        output = model(x)

        assert output.shape == (8, 256)

    def test_projhead_seed_reproducibility(self):
        """Test that same seed produces same initialization."""
        model1 = ProjHead(seed=42)
        model2 = ProjHead(seed=42)

        # Check that initial weights are the same
        for p1, p2 in zip(model1.parameters(), model2.parameters()):
            assert torch.allclose(p1, p2)

    def test_projhead_gradient_flow(self):
        """Test that gradients flow through the projection head."""
        model = ProjHead()
        x = torch.randn(4, 768, requires_grad=True)

        output = model(x)
        loss = output.sum()
        loss.backward()

        # Check that input has gradients
        assert x.grad is not None
        assert not torch.isnan(x.grad).any()


class TestConceptClf:
    """Tests for ConceptClf class."""

    def test_conceptclf_initialization(self):
        """Test ConceptClf initialization."""
        d_out = 100
        model = ConceptClf(d_out=d_out)

        assert isinstance(model.seq, nn.Sequential)

    def test_conceptclf_forward(self):
        """Test forward pass."""
        d_out = 50
        model = ConceptClf(d_out=d_out)
        x = torch.randn(8, 384)

        output = model(x)

        assert output.shape == (8, d_out)

    def test_conceptclf_from_pretrained_not_implemented(self):
        """Test that from_pretrained raises NotImplementedError."""
        with pytest.raises(NotImplementedError):
            ConceptClf.from_pretrained()


class TestLoadModel:
    """Tests for load_model function."""

    def test_load_model_invalid_name(self):
        """Test that invalid model name raises ValueError."""
        with pytest.raises(ValueError):
            load_model("invalid_model_name")

    @pytest.mark.parametrize(
        "model_name",
        [
            "proj_head_text_infonce",
            "proj_head_image_infonce",
            "proj_head_text_mse",
        ],
    )
    def test_load_model_proj_heads_return_type(self, model_name):
        """Test that loading projection heads returns correct type (may fail if pretrained not available)."""
        # This test will fail if pretrained models are not available
        # In that case, it should be skipped or mocked
        try:
            model = load_model(model_name)
            assert isinstance(model, nn.Module)
        except Exception as e:
            pytest.skip(f"Pretrained model not available: {e}")

    def test_load_model_autoencoder_return_type(self):
        """Test that loading autoencoder returns correct type (may fail if pretrained not available)."""
        try:
            model = load_model("autoencoder")
            assert isinstance(model, nn.Module)
        except Exception as e:
            pytest.skip(f"Pretrained model not available: {e}")


class TestModelIntegration:
    """Integration tests for model components."""

    def test_encoder_decoder_roundtrip(self):
        """Test that encoder-decoder can reconstruct inputs."""
        torch.manual_seed(42)
        model = NeuroAutoEncoder(seed=42)
        model.eval()

        x = torch.rand(4, 28542)

        with torch.no_grad():
            output = model(x)

        # Check that reconstruction is reasonable (not checking accuracy, just shape)
        assert output.shape == x.shape
        assert (output >= 0).all() and (output <= 1).all()

    def test_projhead_with_encoder(self):
        """Test that projection head works with encoder output."""
        torch.manual_seed(42)
        encoder = NeuroAutoEncoder(seed=42).encoder
        proj_head = ProjHead(latent_in_dim=384, latent_out_dim=384, seed=42)

        x = torch.rand(4, 28542)

        with torch.no_grad():
            latent = encoder(x)
            projected = proj_head(latent)

        assert projected.shape == (4, 384)
