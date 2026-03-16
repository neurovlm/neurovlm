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
def specter_model():
    """Fixture to provide Specter model"""
    from neurovlm.models import Specter
    model = Specter()
    return model


@pytest.fixture
def pretrained_autoencoder():
    """Fixture to provide pretrained autoencoder"""
    from neurovlm.models import load_model
    model = load_model("autoencoder")
    return model


@pytest.fixture
def pretrained_proj_heads():
    """Fixture to provide pretrained projection heads"""
    from neurovlm.models import load_model
    text_infonce = load_model("proj_head_text_infonce")
    image_infonce = load_model("proj_head_image_infonce")
    text_mse = load_model("proj_head_text_mse")
    return {
        "text_infonce": text_infonce,
        "image_infonce": image_infonce,
        "text_mse": text_mse,
    }

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
