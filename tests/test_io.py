"""Tests for io module."""

import tempfile
from pathlib import Path

import pytest
import torch
from torch import nn

from neurovlm.io import save_model, load_model


class SimpleModel(nn.Module):
    """Simple model for testing save/load functionality."""

    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(10, 20)
        self.linear2 = nn.Linear(20, 5)

    def forward(self, x):
        x = torch.relu(self.linear1(x))
        return self.linear2(x)


class TestSaveModel:
    """Tests for save_model function."""

    def test_save_model_from_module(self):
        """Test saving a model instance."""
        model = SimpleModel()

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "model.safetensors"
            save_model(model, str(save_path))

            assert save_path.exists()
            assert save_path.suffix == ".safetensors"

    def test_save_model_preserves_parameters(self):
        """Test that saved model parameters can be loaded back."""
        torch.manual_seed(42)
        model = SimpleModel()
        original_params = {
            name: param.clone() for name, param in model.named_parameters()
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "model.safetensors"
            save_model(model, str(save_path))

            # Load into new model
            new_model = SimpleModel()
            loaded_model = load_model(new_model, str(save_path))

            # Check parameters match
            for name, param in loaded_model.named_parameters():
                assert torch.allclose(param, original_params[name])


class TestLoadModel:
    """Tests for load_model function."""

    def test_load_model_basic(self):
        """Test basic model loading."""
        torch.manual_seed(42)
        model = SimpleModel()

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "model.safetensors"
            save_model(model, str(save_path))

            new_model = SimpleModel()
            loaded_model = load_model(new_model, str(save_path))

            assert isinstance(loaded_model, SimpleModel)

    def test_load_model_eval_mode(self):
        """Test that model is loaded in eval mode by default."""
        model = SimpleModel()

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "model.safetensors"
            save_model(model, str(save_path))

            new_model = SimpleModel()
            loaded_model = load_model(new_model, str(save_path), eval=True)

            assert not loaded_model.training

    def test_load_model_train_mode(self):
        """Test that model can be loaded in train mode."""
        model = SimpleModel()

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "model.safetensors"
            save_model(model, str(save_path))

            new_model = SimpleModel()
            loaded_model = load_model(new_model, str(save_path), eval=False)

            assert loaded_model.training

    def test_load_model_cpu(self):
        """Test loading model on CPU."""
        model = SimpleModel()

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "model.safetensors"
            save_model(model, str(save_path))

            new_model = SimpleModel()
            loaded_model = load_model(new_model, str(save_path), device="cpu")

            # Check all parameters are on CPU
            for param in loaded_model.parameters():
                assert param.device.type == "cpu"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_load_model_cuda(self):
        """Test loading model on CUDA."""
        model = SimpleModel()

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "model.safetensors"
            save_model(model, str(save_path))

            new_model = SimpleModel()
            loaded_model = load_model(new_model, str(save_path), device="cuda")

            # Check all parameters are on CUDA
            for param in loaded_model.parameters():
                assert param.device.type == "cuda"

    def test_load_model_inference_consistency(self):
        """Test that loaded model produces same outputs as original."""
        torch.manual_seed(42)
        model = SimpleModel()
        model.eval()

        # Generate test input
        test_input = torch.randn(2, 10)

        with torch.no_grad():
            original_output = model(test_input)

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "model.safetensors"
            save_model(model, str(save_path))

            new_model = SimpleModel()
            loaded_model = load_model(new_model, str(save_path))

            with torch.no_grad():
                loaded_output = loaded_model(test_input)

            # Outputs should be identical
            assert torch.allclose(original_output, loaded_output)


class TestSaveLoadRoundtrip:
    """Tests for save/load round-trip consistency."""

    def test_roundtrip_preserves_state(self):
        """Test that save->load preserves exact model state."""
        torch.manual_seed(42)
        model = SimpleModel()

        original_state = model.state_dict()

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "model.safetensors"
            save_model(model, str(save_path))

            new_model = SimpleModel()
            loaded_model = load_model(new_model, str(save_path))

            loaded_state = loaded_model.state_dict()

            # Check all keys match
            assert set(original_state.keys()) == set(loaded_state.keys())

            # Check all values match
            for key in original_state.keys():
                assert torch.allclose(original_state[key], loaded_state[key])

    def test_multiple_save_load_cycles(self):
        """Test multiple save/load cycles preserve model state."""
        torch.manual_seed(42)
        model = SimpleModel()

        original_state = {
            name: param.clone() for name, param in model.named_parameters()
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            # Save and load 3 times
            for i in range(3):
                save_path = Path(tmpdir) / f"model_{i}.safetensors"
                save_model(model, str(save_path))

                new_model = SimpleModel()
                model = load_model(new_model, str(save_path))

            # Check parameters still match original
            for name, param in model.named_parameters():
                assert torch.allclose(param, original_state[name])
