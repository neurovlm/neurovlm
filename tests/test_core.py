"""Tests for core module."""

import pytest
import torch
import numpy as np
import pandas as pd
from unittest.mock import Mock, MagicMock, patch

from neurovlm.core import (
    _l2_normalize,
    TextSearchResult,
    BrainSearchResult,
    BrainTopKResult,
    DATASET_ALIASES,
    DATASET_ID_COLUMNS,
    TEXT_EMBED_DIM,
    LATENT_DIM,
    BRAIN_FLAT_DIM,
)


class TestL2Normalize:
    """Tests for _l2_normalize utility function."""

    def test_l2_normalize_basic(self):
        """Test basic L2 normalization."""
        x = torch.randn(5, 128)
        normalized = _l2_normalize(x)

        # Check shape preserved
        assert normalized.shape == x.shape

        # Check each row has norm 1
        norms = torch.norm(normalized, dim=1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-6)

    def test_l2_normalize_already_normalized(self):
        """Test normalizing already normalized vectors."""
        x = torch.randn(3, 64)
        x = x / x.norm(dim=1, keepdim=True)

        normalized = _l2_normalize(x)

        # Should remain normalized
        norms = torch.norm(normalized, dim=1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-6)

    def test_l2_normalize_numerical_stability(self):
        """Test normalization handles small vectors gracefully."""
        x = torch.ones(4, 10) * 1e-15
        normalized = _l2_normalize(x)

        # Should not produce NaNs or Infs
        assert not torch.isnan(normalized).any()
        assert not torch.isinf(normalized).any()


class TestTextSearchResult:
    """Tests for TextSearchResult class."""

    @pytest.fixture
    def sample_text_result(self):
        """Create a sample TextSearchResult for testing."""
        scores_by_dataset = {
            "pubmed": torch.rand(10, 2),  # 10 documents, 2 queries
            "wiki": torch.rand(5, 2),  # 5 documents, 2 queries
        }
        metadata_by_dataset = {
            "pubmed": pd.DataFrame(
                {
                    "title": [f"Paper {i}" for i in range(10)],
                    "description": [f"Abstract {i}" for i in range(10)],
                }
            ),
            "wiki": pd.DataFrame(
                {
                    "title": [f"Wiki {i}" for i in range(5)],
                    "description": [f"Summary {i}" for i in range(5)],
                }
            ),
        }
        query_embeddings = torch.randn(2, TEXT_EMBED_DIM)
        return TextSearchResult(
            scores_by_dataset=scores_by_dataset,
            metadata_by_dataset=metadata_by_dataset,
            query_embeddings=query_embeddings,
            retrieval_space="shared",
        )

    def test_text_search_result_initialization(self, sample_text_result):
        """Test TextSearchResult can be initialized."""
        assert isinstance(sample_text_result, TextSearchResult)
        assert len(sample_text_result.scores_by_dataset) == 2
        assert "pubmed" in sample_text_result.scores_by_dataset
        assert "wiki" in sample_text_result.scores_by_dataset

    def test_text_search_result_top_k(self, sample_text_result):
        """Test top_k method returns correct number of results."""
        result_df = sample_text_result.top_k(k=3)

        assert isinstance(result_df, pd.DataFrame)
        # Should have results from both datasets (up to k per dataset)
        assert len(result_df) > 0
        assert "dataset" in result_df.columns
        assert "title" in result_df.columns
        assert "description" in result_df.columns
        assert "cosine_similarity" in result_df.columns

    def test_text_search_result_top_k_single_query(self, sample_text_result):
        """Test top_k with specific query index from multi-query result."""
        result_df = sample_text_result.top_k(k=5, query_index=0)

        assert isinstance(result_df, pd.DataFrame)
        assert len(result_df) > 0
        # When original result has multiple queries, query_index column is included
        # even when filtering to a single query
        assert "query_index" in result_df.columns
        # All rows should be for query_index=0
        assert (result_df["query_index"] == 0).all()

    def test_text_search_result_top_k_truly_single_query(self):
        """Test top_k when original result has only one query."""
        # Create result with only 1 query
        scores_by_dataset = {
            "pubmed": torch.rand(10, 1),  # 10 documents, 1 query
        }
        metadata_by_dataset = {
            "pubmed": pd.DataFrame(
                {
                    "title": [f"Paper {i}" for i in range(10)],
                    "description": [f"Abstract {i}" for i in range(10)],
                }
            ),
        }
        query_embeddings = torch.randn(1, TEXT_EMBED_DIM)
        single_query_result = TextSearchResult(
            scores_by_dataset=scores_by_dataset,
            metadata_by_dataset=metadata_by_dataset,
            query_embeddings=query_embeddings,
            retrieval_space="shared",
        )

        result_df = single_query_result.top_k(k=5)

        assert isinstance(result_df, pd.DataFrame)
        assert len(result_df) > 0
        # With truly single query, no query_index column
        assert "query_index" not in result_df.columns

    def test_text_search_result_top_k_single_dataset(self, sample_text_result):
        """Test top_k with specific dataset."""
        result_df = sample_text_result.top_k(k=5, dataset="pubmed")

        assert isinstance(result_df, pd.DataFrame)
        # All results should be from pubmed
        assert (result_df["dataset"] == "pubmed").all()

    def test_text_search_result_top_k_invalid_k(self, sample_text_result):
        """Test that k <= 0 raises ValueError."""
        with pytest.raises(ValueError, match="k must be > 0"):
            sample_text_result.top_k(k=0)

        with pytest.raises(ValueError, match="k must be > 0"):
            sample_text_result.top_k(k=-1)

    def test_text_search_result_top_k_invalid_dataset(self, sample_text_result):
        """Test that invalid dataset raises ValueError."""
        with pytest.raises(ValueError, match="Unknown dataset"):
            sample_text_result.top_k(k=5, dataset="invalid_dataset")

    def test_text_search_result_df_property(self, sample_text_result):
        """Test df property returns full ranked results."""
        df = sample_text_result.df

        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0

    def test_text_search_result_format(self, sample_text_result):
        """Test format method returns string."""
        result_str = sample_text_result.format(k=3)

        assert isinstance(result_str, str)
        assert len(result_str) > 0

    def test_text_search_result_plot_raises(self, sample_text_result):
        """Test that plot method raises ValueError."""
        with pytest.raises(ValueError, match="plot is only available for brain"):
            sample_text_result.plot()

    def test_text_search_result_resolve_query_indices(self):
        """Test _resolve_query_indices static method."""
        # Test with None (all queries)
        indices = TextSearchResult._resolve_query_indices(None, 5)
        assert indices == [0, 1, 2, 3, 4]

        # Test with specific index
        indices = TextSearchResult._resolve_query_indices(2, 5)
        assert indices == [2]

        # Test out of range
        with pytest.raises(IndexError):
            TextSearchResult._resolve_query_indices(10, 5)


class TestBrainTopKResult:
    """Tests for BrainTopKResult class."""

    @pytest.fixture
    def sample_brain_topk(self):
        """Create a sample BrainTopKResult for testing."""
        table = pd.DataFrame(
            {
                "dataset": ["pubmed"] * 5,
                "dataset_index": [0, 1, 2, 3, 4],
                "title": [f"Study {i}" for i in range(5)],
                "description": [f"Desc {i}" for i in range(5)],
                "cosine_similarity": [0.9, 0.8, 0.7, 0.6, 0.5],
            }
        )
        table.attrs["brain_indices"] = [0, 1, 2, 3, 4]
        table.attrs["dataset_names"] = ["pubmed"] * 5
        table.attrs["dataset_indices"] = [0, 1, 2, 3, 4]

        parent_mock = Mock(spec=BrainSearchResult)
        return BrainTopKResult(table=table, parent=parent_mock)

    def test_brain_topk_result_initialization(self, sample_brain_topk):
        """Test BrainTopKResult initialization."""
        assert isinstance(sample_brain_topk, BrainTopKResult)
        assert isinstance(sample_brain_topk.table, pd.DataFrame)

    def test_brain_topk_result_len(self, sample_brain_topk):
        """Test __len__ returns table length."""
        assert len(sample_brain_topk) == 5

    def test_brain_topk_result_repr(self, sample_brain_topk):
        """Test __repr__ returns table representation."""
        repr_str = repr(sample_brain_topk)
        assert isinstance(repr_str, str)
        assert "Study" in repr_str  # Should contain some table content

    def test_brain_topk_result_getitem(self, sample_brain_topk):
        """Test __getitem__ delegates to table."""
        result = sample_brain_topk["title"]
        assert isinstance(result, pd.Series)
        assert len(result) == 5

    def test_brain_topk_result_to_pandas(self, sample_brain_topk):
        """Test to_pandas returns the underlying DataFrame."""
        df = sample_brain_topk.to_pandas()
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 5


class TestBrainSearchResult:
    """Tests for BrainSearchResult class."""

    @pytest.fixture
    def sample_brain_result(self):
        """Create a sample BrainSearchResult for testing."""
        scores = torch.rand(20, 2)  # 20 brain maps, 2 queries
        metadata = pd.DataFrame(
            {
                "dataset": ["pubmed"] * 20,
                "_dataset_index": list(range(20)),
                "title": [f"Study {i}" for i in range(20)],
                "description": [f"Description {i}" for i in range(20)],
            }
        )
        latents = torch.randn(20, LATENT_DIM)
        query_embeddings = torch.randn(2, LATENT_DIM)

        return BrainSearchResult(
            scores=scores,
            metadata=metadata,
            latents=latents,
            query_embeddings=query_embeddings,
            retrieval_space="infonce",
            masker=Mock(),
            decoder=Mock(),
        )

    def test_brain_search_result_initialization(self, sample_brain_result):
        """Test BrainSearchResult initialization."""
        assert isinstance(sample_brain_result, BrainSearchResult)
        assert sample_brain_result.scores is not None
        assert sample_brain_result.metadata is not None

    def test_brain_search_result_top_k(self, sample_brain_result):
        """Test top_k method returns BrainTopKResult."""
        result = sample_brain_result.top_k(k=5)

        assert isinstance(result, BrainTopKResult)
        assert len(result.table) > 0
        assert "cosine_similarity" in result.table.columns

    def test_brain_search_result_top_k_single_query(self, sample_brain_result):
        """Test top_k with specific query index."""
        result = sample_brain_result.top_k(k=3, query_index=0)

        assert isinstance(result, BrainTopKResult)
        assert len(result.table) > 0

    def test_brain_search_result_top_k_invalid_k(self, sample_brain_result):
        """Test that k <= 0 raises ValueError."""
        with pytest.raises(ValueError, match="k must be > 0"):
            sample_brain_result.top_k(k=0)

    def test_brain_search_result_df_property(self, sample_brain_result):
        """Test df property returns DataFrame."""
        df = sample_brain_result.df

        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0

    def test_brain_search_result_generation_mode(self):
        """Test BrainSearchResult in generation mode (MSE)."""
        generated_flatmaps = torch.rand(5, BRAIN_FLAT_DIM)
        latents = torch.randn(5, LATENT_DIM)
        query_embeddings = torch.randn(5, LATENT_DIM)

        result = BrainSearchResult(
            scores=None,
            metadata=None,
            latents=latents,
            query_embeddings=query_embeddings,
            retrieval_space="mse",
            generated_flatmaps=generated_flatmaps,
            masker=Mock(),
        )

        assert result.scores is None
        assert result.metadata is None
        assert result.generated_flatmaps is not None

        # top_k should raise error in generation mode
        with pytest.raises(
            ValueError, match="top_k is only available for retrieval outputs"
        ):
            result.top_k(k=3)

    def test_brain_search_result_images_property(self):
        """Test images property for generation mode."""
        generated_flatmaps = torch.rand(3, BRAIN_FLAT_DIM)
        latents = torch.randn(3, LATENT_DIM)
        query_embeddings = torch.randn(3, LATENT_DIM)

        # Mock masker that returns a simple object
        masker_mock = Mock()
        masker_mock.inverse_transform.return_value = Mock()

        result = BrainSearchResult(
            scores=None,
            metadata=None,
            latents=latents,
            query_embeddings=query_embeddings,
            retrieval_space="mse",
            generated_flatmaps=generated_flatmaps,
            masker=masker_mock,
        )

        # Should not raise error
        images = result.images
        assert images is not None


class TestConstants:
    """Tests for module constants."""

    def test_dataset_aliases_consistency(self):
        """Test that DATASET_ALIASES maps to valid datasets."""
        for alias, canonical in DATASET_ALIASES.items():
            assert isinstance(alias, str)
            assert isinstance(canonical, str)

    def test_dataset_id_columns_consistency(self):
        """Test that DATASET_ID_COLUMNS has valid keys."""
        for dataset, id_col in DATASET_ID_COLUMNS.items():
            assert isinstance(dataset, str)
            assert isinstance(id_col, str)

    def test_dimension_constants(self):
        """Test that dimension constants are positive integers."""
        assert TEXT_EMBED_DIM > 0
        assert LATENT_DIM > 0
        assert BRAIN_FLAT_DIM > 0


class TestUtilityFunctions:
    """Tests for utility functions in core module."""

    def test_normalize_record_id_integers(self):
        """Test _normalize_record_id with integer inputs."""
        from neurovlm.core import NeuroVLM

        assert NeuroVLM._normalize_record_id(42) == 42
        assert NeuroVLM._normalize_record_id(np.int64(42)) == 42

    def test_normalize_record_id_strings(self):
        """Test _normalize_record_id with string inputs."""
        from neurovlm.core import NeuroVLM

        assert NeuroVLM._normalize_record_id("123") == 123
        assert NeuroVLM._normalize_record_id("abc") == "abc"
        assert NeuroVLM._normalize_record_id("  ") is None

    def test_normalize_record_id_none(self):
        """Test _normalize_record_id with None."""
        from neurovlm.core import NeuroVLM

        assert NeuroVLM._normalize_record_id(None) is None
        assert NeuroVLM._normalize_record_id(np.nan) is None

    def test_clean_text(self):
        """Test _clean_text method."""
        from neurovlm.core import NeuroVLM

        assert NeuroVLM._clean_text("  hello   world  ") == "hello world"
        assert NeuroVLM._clean_text("hello\nworld") == "hello world"
        assert NeuroVLM._clean_text("") == ""

    def test_is_text_payload(self):
        """Test _is_text_payload method."""
        from neurovlm.core import NeuroVLM

        assert NeuroVLM._is_text_payload("hello") is True
        assert NeuroVLM._is_text_payload({"title": "test"}) is True
        assert NeuroVLM._is_text_payload(pd.DataFrame()) is True
        assert NeuroVLM._is_text_payload(["text1", "text2"]) is True
        assert NeuroVLM._is_text_payload([{"title": "test"}]) is True

        assert NeuroVLM._is_text_payload(torch.randn(10, 768)) is False
        assert NeuroVLM._is_text_payload(np.random.randn(10, 768)) is False
        assert NeuroVLM._is_text_payload([1, 2, 3]) is False

    def test_as_2d_tensor(self):
        """Test _as_2d_tensor method."""
        from neurovlm.core import NeuroVLM

        # 1D tensor
        x_1d = torch.randn(10)
        result = NeuroVLM._as_2d_tensor(x_1d)
        assert result.shape == (1, 10)

        # 2D tensor
        x_2d = torch.randn(5, 10)
        result = NeuroVLM._as_2d_tensor(x_2d)
        assert result.shape == (5, 10)

        # NumPy array
        x_np = np.random.randn(3, 10)
        result = NeuroVLM._as_2d_tensor(x_np)
        assert result.shape == (3, 10)
        assert isinstance(result, torch.Tensor)

        # 3D tensor should raise error
        x_3d = torch.randn(2, 3, 4)
        with pytest.raises(ValueError, match="Expected 1D/2D input"):
            NeuroVLM._as_2d_tensor(x_3d)

    def test_validate_head(self):
        """Test _validate_head method."""
        from neurovlm.core import NeuroVLM

        # Valid heads should not raise
        NeuroVLM._validate_head("mse")
        NeuroVLM._validate_head("infonce")

        # Invalid head should raise
        with pytest.raises(ValueError, match="head must be either"):
            NeuroVLM._validate_head("invalid")

    def test_canonicalize_brain_dataset(self):
        """Test _canonicalize_brain_dataset method."""
        from neurovlm.core import NeuroVLM

        assert NeuroVLM._canonicalize_brain_dataset("pubmed") == "neuro"
        assert NeuroVLM._canonicalize_brain_dataset("neuro") == "neuro"
        assert NeuroVLM._canonicalize_brain_dataset("networks") == "networks"
        assert NeuroVLM._canonicalize_brain_dataset("neurovault") == "neurovault"

        with pytest.raises(ValueError):
            NeuroVLM._canonicalize_brain_dataset("invalid")
