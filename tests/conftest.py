"""Pytest configuration and fixtures for neurovlm tests."""

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
