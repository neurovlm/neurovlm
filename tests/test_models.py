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
    Specter,
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

    @pytest.mark.requires_pretrained
    @pytest.mark.parametrize(
        "model_name",
        [
            "proj_head_text_infonce",
            "proj_head_image_infonce",
            "proj_head_text_mse",
        ],
    )
    def test_load_model_proj_heads_return_type(self, model_name, skip_if_no_models):
        """Test that loading projection heads returns correct type."""
        model = load_model(model_name)
        assert isinstance(model, nn.Module)

    @pytest.mark.requires_pretrained
    def test_load_model_autoencoder_return_type(self, skip_if_no_models):
        """Test that loading autoencoder returns correct type."""
        model = load_model("autoencoder")
        assert isinstance(model, nn.Module)


@pytest.mark.requires_specter
class TestSpecter:
    """Tests for Specter text encoder."""

    def test_specter_initialization_default(self, specter_model):
        """Test default Specter initialization."""
        model = specter_model
        assert model.device.type == "cpu"
        assert model.pooling is None  # Default is CLS token

    def test_specter_initialization_with_adapter(self, skip_if_no_models):
        """Test Specter initialization with specific adapter."""
        model = Specter(adapter="adhoc_query")
        assert model.device.type == "cpu"

    def test_specter_initialization_no_adapter(self, skip_if_no_models):
        """Test Specter initialization without adapter."""
        model = Specter(adapter=None)
        assert model.device.type == "cpu"

    @pytest.mark.parametrize("adapter", [
        "adhoc_query",
        "classification",
        "regression",
        "proximity",
        None,  # No adapter
    ])
    def test_specter_forward_with_adapters(self, adapter, skip_if_no_models):
        """Test forward pass with different adapters."""
        model = Specter(adapter=adapter)

        # Test with single string
        text = "Memory and attention in the prefrontal cortex."
        output = model(text)

        assert isinstance(output, torch.Tensor)
        assert output.shape == (1, 768)  # Specter embedding dimension
        assert not torch.isnan(output).any()

    def test_specter_forward_single_string(self, specter_model):
        """Test forward pass with a single string."""
        model = specter_model
        text = "Neural correlates of working memory."
        output = model(text)

        assert output.shape == (1, 768)
        assert not torch.isnan(output).any()

    def test_specter_forward_list_of_strings(self, specter_model):
        """Test forward pass with list of strings."""
        model = specter_model
        texts = [
            "Working memory in prefrontal cortex",
            "Visual attention and perception",
            "Language processing networks"
        ]
        output = model(texts)

        assert output.shape == (3, 768)
        assert not torch.isnan(output).any()

    def test_specter_forward_dataframe(self, specter_model):
        """Test forward pass with DataFrame."""
        model = specter_model
        df = pd.DataFrame({
            'title': ['Neural mechanisms of memory', 'Attention networks'],
            'abstract': ['Study of memory encoding...', 'Investigation of attention...']
        })
        output = model(df)

        assert output.shape == (2, 768)
        assert not torch.isnan(output).any()

    def test_specter_forward_dict(self, specter_model):
        """Test forward pass with dictionary."""
        model = specter_model
        doc = {
            'title': 'Neural correlates of decision making',
            'abstract': 'We investigated the neural mechanisms...'
        }
        output = model(doc)

        assert output.shape == (1, 768)
        assert not torch.isnan(output).any()

    def test_specter_forward_list_of_dicts(self, specter_model):
        """Test forward pass with list of dictionaries."""
        model = specter_model
        docs = [
            {'title': 'Memory study', 'abstract': 'Abstract 1'},
            {'title': 'Attention study', 'summary': 'Abstract 2'}  # Test with 'summary' instead
        ]
        output = model(docs)

        assert output.shape == (2, 768)
        assert not torch.isnan(output).any()

    @pytest.mark.parametrize("pooling", [None, "mean", "max", "mean_max", "attention"])
    def test_specter_pooling_methods(self, pooling, skip_if_no_models):
        """Test different pooling strategies."""
        model = Specter(pooling=pooling)
        text = "Neural processing in the brain."
        output = model(text)

        assert isinstance(output, torch.Tensor)
        # mean_max doubles the dimension
        expected_dim = 1536 if pooling == "mean_max" else 768
        assert output.shape == (1, expected_dim)
        assert not torch.isnan(output).any()

    def test_specter_orthogonalization(self, skip_if_no_models):
        """Test that orthogonalization affects embeddings."""
        # Note: parameter is 'orthgonalize' (with typo) in the codebase
        model_ortho = Specter(orthgonalize=True)
        model_no_ortho = Specter(orthgonalize=False)

        text = "Neural networks in the brain."

        output_ortho = model_ortho(text)
        output_no_ortho = model_no_ortho(text)

        # Outputs should be different when orthogonalization is applied
        assert output_ortho.shape == output_no_ortho.shape
        assert not torch.allclose(output_ortho, output_no_ortho, atol=1e-3)

    def test_specter_device_movement(self, skip_if_no_models):
        """Test moving Specter to different devices."""
        model = Specter(device="cpu")

        # Test CPU
        text = "Neural activity patterns."
        output = model(text)
        assert output.device.type == "cpu"

        # Test CUDA if available
        if torch.cuda.is_available():
            model = model.to("cuda")
            output = model(text)
            assert output.device.type == "cuda"

    def test_specter_batch_consistency(self, specter_model):
        """Test that batch processing gives same results as individual processing."""
        model = specter_model
        texts = ["Text 1", "Text 2", "Text 3"]

        # Batch processing
        batch_output = model(texts)

        # Individual processing
        individual_outputs = torch.stack([model(t) for t in texts]).squeeze(1)

        # Should be identical
        assert torch.allclose(batch_output, individual_outputs, atol=1e-5)


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

    @pytest.mark.requires_specter
    def test_full_pipeline_text_to_latent(self, specter_model):
        """Test full pipeline: Specter -> ProjHead."""
        specter = specter_model
        proj_head = ProjHead(latent_in_dim=768, latent_out_dim=384)

        text = "Neural mechanisms of cognitive control."

        with torch.no_grad():
            # Specter encoding
            text_emb = specter(text)
            assert text_emb.shape == (1, 768)

            # Project to shared latent space
            latent = proj_head(text_emb)
            assert latent.shape == (1, 384)
            assert not torch.isnan(latent).any()

    def test_full_pipeline_neuro_to_latent(self):
        """Test full pipeline: NeuroAutoEncoder encoder -> latent."""
        torch.manual_seed(42)
        autoencoder = NeuroAutoEncoder(seed=42)

        brain_data = torch.rand(2, 28542)

        with torch.no_grad():
            # Encode brain data
            latent = autoencoder.encoder(brain_data)
            assert latent.shape == (2, 384)
            assert not torch.isnan(latent).any()

            # Decode back
            reconstructed = autoencoder.decoder(latent)
            assert reconstructed.shape == (2, 28542)

    def test_concept_classifier_with_encoder(self):
        """Test ConceptClf with encoder output."""
        torch.manual_seed(42)
        autoencoder = NeuroAutoEncoder(seed=42)
        clf = ConceptClf(d_out=100)

        brain_data = torch.rand(4, 28542)

        with torch.no_grad():
            latent = autoencoder.encoder(brain_data)
            concepts = clf(latent)

        assert concepts.shape == (4, 100)
        assert not torch.isnan(concepts).any()


class TestForwardPassesAllModels:
    """Comprehensive forward pass tests for all models."""

    def test_normalize_layer_forward_pass(self):
        """Test NormalizeLayer forward pass."""
        layer = NormalizeLayer()
        x = torch.randn(10, 768)

        output = layer(x)

        assert output.shape == (10, 768)
        assert not torch.isnan(output).any()
        norms = torch.norm(output, dim=1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-6)

    def test_autoencoder_forward_pass(self):
        """Test NeuroAutoEncoder full forward pass."""
        model = NeuroAutoEncoder(seed=42)
        x = torch.rand(5, 28542)

        output = model(x)

        assert output.shape == (5, 28542)
        assert not torch.isnan(output).any()
        assert (output >= 0).all() and (output <= 1).all()

    def test_projhead_text_forward_pass(self):
        """Test ProjHead forward pass with text dimensions."""
        model = ProjHead(latent_in_dim=768, latent_out_dim=384)
        x = torch.randn(8, 768)

        output = model(x)

        assert output.shape == (8, 384)
        assert not torch.isnan(output).any()

    def test_projhead_image_forward_pass(self):
        """Test ProjHead forward pass with image dimensions."""
        model = ProjHead(latent_in_dim=384, latent_out_dim=384)
        x = torch.randn(6, 384)

        output = model(x)

        assert output.shape == (6, 384)
        assert not torch.isnan(output).any()

    def test_concept_clf_forward_pass(self):
        """Test ConceptClf forward pass."""
        model = ConceptClf(d_out=200)
        x = torch.randn(12, 384)

        output = model(x)

        assert output.shape == (12, 200)
        assert not torch.isnan(output).any()

    @pytest.mark.requires_specter
    @pytest.mark.parametrize("adapter", [
        "adhoc_query",
        "classification",
        "regression",
        None,
    ])
    def test_specter_adapters_forward_pass(self, adapter, skip_if_no_models):
        """Test Specter forward passes with all adapters."""
        model = Specter(adapter=adapter)

        # Test various input formats
        test_cases = [
            "Single text input",
            ["Multiple", "text", "inputs"],
            {"title": "Title", "abstract": "Abstract"},
            pd.DataFrame({
                "title": ["Title 1", "Title 2"],
                "abstract": ["Abs 1", "Abs 2"]
            })
        ]

        for test_input in test_cases:
            output = model(test_input)
            assert isinstance(output, torch.Tensor)
            assert output.shape[1] == 768  # Specter dimension
            assert not torch.isnan(output).any()

    @pytest.mark.requires_pretrained
    def test_all_pretrained_models_forward(self, skip_if_no_models):
        """Test forward passes for all pretrained models via load_model."""
        model_names = [
            "proj_head_text_infonce",
            "proj_head_image_infonce",
            "proj_head_text_mse",
            "autoencoder",
        ]

        for model_name in model_names:
            model = load_model(model_name)

            # Determine appropriate input shape
            if "text" in model_name:
                x = torch.randn(3, 768)
                expected_out = (3, 384)
            elif "image" in model_name:
                x = torch.randn(3, 384)
                expected_out = (3, 384)
            elif "autoencoder" in model_name:
                x = torch.rand(3, 28542)
                expected_out = (3, 28542)
            else:
                continue

            with torch.no_grad():
                output = model(x)

            assert output.shape == expected_out
            assert not torch.isnan(output).any()
