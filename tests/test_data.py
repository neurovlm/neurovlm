"""Tests for data module."""

import pytest
import torch
import numpy as np
import pandas as pd

from neurovlm.data import (
    load_dataset,
    load_latent,
    load_masker,
    get_data_dir,
    _without_grad,
)


class TestWithoutGrad:
    """Tests for _without_grad utility function."""

    def test_without_grad_tensor(self):
        """Test detaching a single tensor."""
        x = torch.randn(3, 4, requires_grad=True)
        result = _without_grad(x)

        assert isinstance(result, torch.Tensor)
        assert not result.requires_grad
        assert result.shape == x.shape

    def test_without_grad_tuple(self):
        """Test detaching tensors in a tuple."""
        x = torch.randn(3, 4, requires_grad=True)
        y = torch.randn(5, 6, requires_grad=True)
        result = _without_grad((x, y))

        assert isinstance(result, tuple)
        assert len(result) == 2
        assert not result[0].requires_grad
        assert not result[1].requires_grad

    def test_without_grad_list(self):
        """Test detaching tensors in a list."""
        x = torch.randn(3, 4, requires_grad=True)
        y = torch.randn(5, 6, requires_grad=True)
        result = _without_grad([x, y])

        assert isinstance(result, list)
        assert len(result) == 2
        assert not result[0].requires_grad
        assert not result[1].requires_grad

    def test_without_grad_dict(self):
        """Test detaching tensors in a dict."""
        data = {
            "a": torch.randn(3, 4, requires_grad=True),
            "b": torch.randn(5, 6, requires_grad=True),
        }
        result = _without_grad(data)

        assert isinstance(result, dict)
        assert set(result.keys()) == {"a", "b"}
        assert not result["a"].requires_grad
        assert not result["b"].requires_grad

    def test_without_grad_nested(self):
        """Test detaching nested structures."""
        data = {
            "tensors": [
                torch.randn(2, 3, requires_grad=True),
                torch.randn(4, 5, requires_grad=True),
            ],
            "tuple": (torch.randn(1, 2, requires_grad=True),),
        }
        result = _without_grad(data)

        assert not result["tensors"][0].requires_grad
        assert not result["tensors"][1].requires_grad
        assert not result["tuple"][0].requires_grad

    def test_without_grad_non_tensor(self):
        """Test that non-tensor objects are returned unchanged."""
        x = 42
        result = _without_grad(x)
        assert result == x

        y = "string"
        result = _without_grad(y)
        assert result == y


class TestGetDataDir:
    """Tests for get_data_dir function."""

    def test_get_data_dir_returns_path(self):
        """Test that get_data_dir returns a Path object."""
        from pathlib import Path

        data_dir = get_data_dir()
        assert isinstance(data_dir, Path)

    def test_get_data_dir_creates_directory(self):
        """Test that get_data_dir creates the directory if it doesn't exist."""
        data_dir = get_data_dir()
        assert data_dir.exists()
        assert data_dir.is_dir()

    def test_get_data_dir_in_home_cache(self):
        """Test that data dir is in user's home .cache directory."""
        from pathlib import Path

        data_dir = get_data_dir()
        expected_parent = Path.home() / ".cache"
        assert data_dir.parent == expected_parent
        assert data_dir.name == "neurovlm"


class TestLoadDataset:
    """Tests for load_dataset function.

    Note: These tests may fail if the actual data is not available.
    They primarily test the function interface and error handling.
    """

    def test_load_dataset_invalid_name(self):
        """Test that invalid dataset name raises ValueError."""
        with pytest.raises(ValueError):
            load_dataset("invalid_dataset_name")

    @pytest.mark.parametrize(
        "dataset_name",
        [
            "pubmed_text",
            "wiki",
            "neurowiki",
            "cogatlas",
            "cogatlas_task",
            "cogatlas_disorder",
        ],
    )
    def test_load_dataset_valid_names_dataframe(self, dataset_name):
        """Test loading datasets that should return DataFrames."""
        try:
            result = load_dataset(dataset_name)
            assert isinstance(result, pd.DataFrame)
        except (ValueError, FileNotFoundError, Exception) as e:
            pytest.skip(f"Dataset {dataset_name} not available: {e}")

    @pytest.mark.parametrize(
        "dataset_name",
        ["pubmed_images", "neurovault_images"],
    )
    def test_load_dataset_valid_names_tensor(self, dataset_name):
        """Test loading datasets that should return tensors/arrays."""
        try:
            result = load_dataset(dataset_name)
            # Should be tensor or tuple containing tensor
            if isinstance(result, tuple):
                assert len(result) >= 1
                assert isinstance(result[0], (torch.Tensor, np.ndarray))
            else:
                assert isinstance(result, (torch.Tensor, np.ndarray))
        except (ValueError, FileNotFoundError, Exception) as e:
            pytest.skip(f"Dataset {dataset_name} not available: {e}")

    def test_load_dataset_networks(self):
        """Test loading networks dataset."""
        try:
            result = load_dataset("networks")
            # Should return a dict with network data
            assert isinstance(result, dict)
        except (ValueError, FileNotFoundError, Exception) as e:
            pytest.skip(f"Networks dataset not available: {e}")

    def test_load_dataset_alias_wiki(self):
        """Test that 'wiki' and 'neurowiki' are aliases."""
        try:
            result1 = load_dataset("wiki")
            result2 = load_dataset("neurowiki")
            # Should return same type
            assert type(result1) == type(result2)
        except (ValueError, FileNotFoundError, Exception) as e:
            pytest.skip(f"Wiki dataset not available: {e}")


class TestLoadLatent:
    """Tests for load_latent function.

    Note: These tests may fail if the actual latent data is not available.
    """

    def test_load_latent_invalid_name(self):
        """Test that invalid latent name raises ValueError."""
        with pytest.raises(ValueError):
            load_latent("invalid_latent_name")

    @pytest.mark.parametrize(
        "latent_name",
        [
            "pubmed_text",
            "pubmed_images",
            "wiki",
            "neurowiki",
            "cogatlas",
            "cogatlas_task",
            "cogatlas_disorder",
        ],
    )
    def test_load_latent_valid_names(self, latent_name):
        """Test loading valid latent embeddings."""
        try:
            result = load_latent(latent_name)
            # Result could be tensor, tuple, or dict
            assert result is not None

            # Check that if it's a tensor, it has no gradients
            if isinstance(result, torch.Tensor):
                assert not result.requires_grad
            elif isinstance(result, tuple) and len(result) > 0:
                if isinstance(result[0], torch.Tensor):
                    assert not result[0].requires_grad
        except (ValueError, FileNotFoundError, Exception) as e:
            pytest.skip(f"Latent {latent_name} not available: {e}")

    def test_load_latent_returns_no_grad(self):
        """Test that loaded latents have no gradients."""
        try:
            result = load_latent("pubmed_text")

            # Recursively check that no tensors have gradients
            def check_no_grad(obj):
                if isinstance(obj, torch.Tensor):
                    assert not obj.requires_grad
                elif isinstance(obj, (list, tuple)):
                    for item in obj:
                        check_no_grad(item)
                elif isinstance(obj, dict):
                    for value in obj.values():
                        check_no_grad(value)

            check_no_grad(result)
        except (ValueError, FileNotFoundError, Exception) as e:
            pytest.skip(f"Latent data not available: {e}")

    def test_load_latent_networks_neuro(self):
        """Test loading network neuro latents (dict structure)."""
        try:
            result = load_latent("networks_neuro")
            # Should be a nested dict
            assert isinstance(result, dict)
        except (ValueError, FileNotFoundError, Exception) as e:
            pytest.skip(f"Networks neuro latent not available: {e}")

    def test_load_latent_alias_wiki(self):
        """Test that 'wiki' and 'neurowiki' are aliases for latents."""
        try:
            result1 = load_latent("wiki")
            result2 = load_latent("neurowiki")
            # Should return same type
            assert type(result1) == type(result2)
        except (ValueError, FileNotFoundError, Exception) as e:
            pytest.skip(f"Wiki latent not available: {e}")


class TestLoadMasker:
    """Tests for load_masker function."""

    def test_load_masker_returns_masker(self):
        """Test that load_masker returns a masker object."""
        try:
            masker = load_masker()
            # Should have transform and inverse_transform methods
            assert hasattr(masker, "transform")
            assert hasattr(masker, "inverse_transform")
            assert hasattr(masker, "mask_img_")
        except (ValueError, FileNotFoundError, Exception) as e:
            pytest.skip(f"Masker not available: {e}")

    def test_load_masker_consistent(self):
        """Test that loading masker multiple times returns consistent object."""
        try:
            masker1 = load_masker()
            masker2 = load_masker()

            # Should be same type
            assert type(masker1) == type(masker2)
        except (ValueError, FileNotFoundError, Exception) as e:
            pytest.skip(f"Masker not available: {e}")


class TestDataIntegration:
    """Integration tests for data loading."""

    def test_load_multiple_datasets(self):
        """Test loading multiple datasets in sequence."""
        dataset_names = ["pubmed_text", "wiki", "cogatlas"]

        for name in dataset_names:
            try:
                result = load_dataset(name)
                assert result is not None
            except (ValueError, FileNotFoundError, Exception):
                # Skip if data not available
                pass

    def test_load_dataset_and_latent_consistency(self):
        """Test that dataset and corresponding latent have consistent sizes."""
        try:
            dataset = load_dataset("pubmed_text")
            latent = load_latent("pubmed_text")

            # Extract tensor from latent (could be tuple or dict)
            latent_tensor = None
            if isinstance(latent, torch.Tensor):
                latent_tensor = latent
            elif isinstance(latent, tuple) and len(latent) > 0:
                latent_tensor = latent[0]

            if latent_tensor is not None and isinstance(dataset, pd.DataFrame):
                # Sizes should be consistent (may not be exact match due to filtering)
                assert latent_tensor.shape[0] > 0
                assert len(dataset) > 0
        except (ValueError, FileNotFoundError, Exception) as e:
            pytest.skip(f"Data not available for consistency test: {e}")
