"""Pytest configuration and fixtures for neurovlm tests."""

import os
import pytest
import torch
import numpy as np


@pytest.fixture(autouse=True)
def set_random_seeds():
    """Set random seeds for reproducibility in tests."""
    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
        torch.cuda.manual_seed_all(42)


@pytest.fixture
def device():
    """Get the default device for testing."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def skip_if_no_models():
    """Skip test if SKIP_PRETRAINED_TESTS environment variable is set."""
    if os.getenv("SKIP_PRETRAINED_TESTS", "false").lower() == "true":
        pytest.skip("Skipping pretrained model tests (SKIP_PRETRAINED_TESTS=true)")


@pytest.fixture
def specter_model():
    """Fixture to provide Specter model, skip if not available or disabled."""
    from neurovlm.models import Specter

    if os.getenv("SKIP_PRETRAINED_TESTS", "false").lower() == "true":
        pytest.skip("Skipping Specter tests (SKIP_PRETRAINED_TESTS=true)")

    try:
        model = Specter()
        return model
    except Exception as e:
        pytest.skip(f"Specter model not available: {e}")


@pytest.fixture
def pretrained_autoencoder():
    """Fixture to provide pretrained autoencoder, skip if not available or disabled."""
    from neurovlm.models import load_model

    if os.getenv("SKIP_PRETRAINED_TESTS", "false").lower() == "true":
        pytest.skip("Skipping pretrained autoencoder tests (SKIP_PRETRAINED_TESTS=true)")

    try:
        model = load_model("autoencoder")
        return model
    except Exception as e:
        pytest.skip(f"Pretrained autoencoder not available: {e}")


@pytest.fixture
def pretrained_proj_heads():
    """Fixture to provide pretrained projection heads, skip if not available or disabled."""
    from neurovlm.models import load_model

    if os.getenv("SKIP_PRETRAINED_TESTS", "false").lower() == "true":
        pytest.skip("Skipping pretrained projection head tests (SKIP_PRETRAINED_TESTS=true)")

    try:
        text_infonce = load_model("proj_head_text_infonce")
        image_infonce = load_model("proj_head_image_infonce")
        text_mse = load_model("proj_head_text_mse")
        return {
            "text_infonce": text_infonce,
            "image_infonce": image_infonce,
            "text_mse": text_mse,
        }
    except Exception as e:
        pytest.skip(f"Pretrained projection heads not available: {e}")


def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers",
        "requires_data: marks tests that require downloaded data files",
    )
    config.addinivalue_line(
        "markers",
        "requires_pretrained: marks tests that require pretrained models",
    )
    config.addinivalue_line(
        "markers",
        "requires_specter: mark test as requiring Specter model from HuggingFace"
    )
