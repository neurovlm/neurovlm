"""Tests for metrics module."""

import numpy as np
import pytest
import torch

from neurovlm.metrics import (
    compute_metrics,
    dice,
    dice_top_k,
    recall_at_k,
    recall_curve,
    bernoulli_bce,
    bits_per_pixel,
    compute_ae_performance,
)


class TestDice:
    """Tests for dice score computation."""

    def test_dice_perfect_match(self):
        """Test dice score with perfect overlap."""
        img_a = np.array([1, 1, 0, 0])
        img_b = np.array([1, 1, 0, 0])
        score = dice(img_a, img_b)
        assert score == 1.0

    def test_dice_no_overlap(self):
        """Test dice score with no overlap."""
        img_a = np.array([1, 1, 0, 0])
        img_b = np.array([0, 0, 1, 1])
        score = dice(img_a, img_b)
        assert score == 0.0

    def test_dice_partial_overlap(self):
        """Test dice score with partial overlap."""
        img_a = np.array([1, 1, 1, 0])
        img_b = np.array([1, 1, 0, 0])
        score = dice(img_a, img_b)
        expected = 2.0 * 2 / (3 + 2)  # 2 intersection, 3+2 total
        assert np.isclose(score, expected)

    def test_dice_empty_images(self):
        """Test dice score with empty images (edge case)."""
        img_a = np.array([0, 0, 0, 0])
        img_b = np.array([0, 0, 0, 0])
        score = dice(img_a, img_b)
        assert score == 1.0  # Default for empty images


class TestDiceTopK:
    """Tests for dice_top_k function."""

    def test_dice_top_k_perfect(self):
        """Test dice_top_k with perfect prediction."""
        y_true = np.array([1, 1, 0, 0, 0])
        y_prob = np.array([0.9, 0.8, 0.2, 0.1, 0.05])
        score = dice_top_k(y_true, y_prob, k=2)
        assert score == 1.0

    def test_dice_top_k_no_overlap(self):
        """Test dice_top_k with no overlap."""
        y_true = np.array([1, 1, 0, 0, 0])
        y_prob = np.array([0.1, 0.05, 0.9, 0.8, 0.7])
        score = dice_top_k(y_true, y_prob, k=2)
        assert score == 0.0

    def test_dice_top_k_auto_k(self):
        """Test dice_top_k with automatic k selection."""
        y_true = np.array([1, 1, 1, 0, 0])
        y_prob = np.array([0.9, 0.8, 0.7, 0.2, 0.1])
        score = dice_top_k(y_true, y_prob, k=None)
        assert score == 1.0


class TestRecallAtK:
    """Tests for recall_at_k metric."""

    def test_recall_at_k_perfect(self):
        """Test recall at k with perfect similarity matrix."""
        n = 5
        cos_sim = torch.eye(n)
        score = recall_at_k(cos_sim, k=1)
        assert score == 1.0

    def test_recall_at_k_random(self):
        """Test recall at k with random similarity."""
        torch.manual_seed(42)
        n = 10
        cos_sim = torch.randn(n, n)
        score = recall_at_k(cos_sim, k=5)
        assert 0.0 <= score <= 1.0

    def test_recall_at_k_all(self):
        """Test recall at k=n should always be 1.0."""
        n = 5
        cos_sim = torch.randn(n, n)
        score = recall_at_k(cos_sim, k=n)
        assert score == 1.0


class TestRecallCurve:
    """Tests for recall curve computation."""

    def test_recall_curve_perfect_alignment(self):
        """Test recall curve with perfect text-image alignment."""
        n = 10
        d = 128
        torch.manual_seed(42)
        latent = torch.randn(n, d)
        t_to_i, i_to_t = recall_curve(latent, latent, step=1)

        # Perfect alignment should give perfect recall
        assert t_to_i[-1].item() == 1.0
        assert i_to_t[-1].item() == 1.0

    def test_recall_curve_shape(self):
        """Test recall curve output shapes."""
        n = 20
        d = 64
        torch.manual_seed(42)
        latent_text = torch.randn(n, d)
        latent_image = torch.randn(n, d)

        step = 5
        t_to_i, i_to_t = recall_curve(latent_text, latent_image, step=step)

        expected_len = len(range(0, n, step))
        assert len(t_to_i) == expected_len
        assert len(i_to_t) == expected_len


class TestBernoulliBCE:
    """Tests for bernoulli BCE computation."""

    def test_bernoulli_bce_perfect_prediction(self):
        """Test BCE with perfect predictions."""
        y = np.array([1.0, 0.0, 1.0, 0.0])
        p = np.array([1.0, 0.0, 1.0, 0.0])
        loss = bernoulli_bce(y, p)
        # Should be very close to zero
        assert np.all(loss < 1e-5)

    def test_bernoulli_bce_worst_prediction(self):
        """Test BCE with worst predictions."""
        y = np.array([1.0, 0.0, 1.0, 0.0])
        p = np.array([0.0, 1.0, 0.0, 1.0])
        loss = bernoulli_bce(y, p)
        # Should be large
        assert np.all(loss > 10)

    def test_bernoulli_bce_shape(self):
        """Test BCE output shape matches input."""
        y = np.random.rand(10, 100)
        p = np.random.rand(10, 100)
        loss = bernoulli_bce(y, p)
        assert loss.shape == y.shape


class TestBitsPerPixel:
    """Tests for bits per pixel computation."""

    def test_bits_per_pixel_perfect_prediction(self):
        """Test BPP with perfect predictions."""
        y_true = np.array([[1.0, 0.0, 1.0, 0.0]])
        logits = np.array([[10.0, -10.0, 10.0, -10.0]])  # High confidence

        result = bits_per_pixel(y_true, logits, baseline="global")

        # Perfect predictions should have low BPP
        assert result["bpp_model_per_image"][0] < 0.1
        # Should improve over baseline
        assert result["delta_bpp_per_image"][0] > 0

    def test_bits_per_pixel_baseline_options(self):
        """Test different baseline options."""
        y_true = np.random.rand(5, 100)
        logits = np.random.randn(5, 100)

        result_global = bits_per_pixel(y_true, logits, baseline="global")
        result_pixel = bits_per_pixel(y_true, logits, baseline="per_pixel")

        assert "bpp_model_per_image" in result_global
        assert "bpp_model_per_image" in result_pixel
        assert result_global["baseline"] == "global"
        assert result_pixel["baseline"] == "per_pixel"

    def test_bits_per_pixel_output_keys(self):
        """Test that all expected keys are present in output."""
        y_true = np.random.rand(3, 50)
        logits = np.random.randn(3, 50)

        result = bits_per_pixel(y_true, logits)

        expected_keys = {
            "bpp_model_per_image",
            "bpp_base_per_image",
            "delta_bpp_per_image",
            "delta_bpp_per_pixel",
            "baseline",
        }
        assert set(result.keys()) == expected_keys


class TestComputeMetrics:
    """Tests for compute_metrics function."""

    def test_compute_metrics_identical_tensors(self):
        """Test metrics with identical original and reconstructed."""
        original = torch.rand(100)
        reconstructed = original.clone()

        mse, ssim, dice_scores = compute_metrics(
            original, reconstructed, thresholds=(0.5,)
        )

        # MSE should be 0, SSIM and Dice should be 1
        assert np.isclose(mse[0], 0.0)
        assert np.isclose(ssim[0], 1.0)
        assert np.isclose(dice_scores[0], 1.0)

    def test_compute_metrics_multiple_thresholds(self):
        """Test metrics with multiple thresholds."""
        original = torch.rand(100)
        reconstructed = torch.rand(100)

        thresholds = (0.1, 0.5, 0.9)
        mse, ssim, dice_scores = compute_metrics(
            original, reconstructed, thresholds=thresholds
        )

        assert len(mse) == len(thresholds)
        assert len(ssim) == len(thresholds)
        assert len(dice_scores) == len(thresholds)

    def test_compute_metrics_with_percentile(self):
        """Test metrics with percentile thresholding."""
        original = torch.rand(100)
        reconstructed = torch.rand(100)

        mse, ssim, dice_scores = compute_metrics(
            original, reconstructed, thresholds=(90, 95, 99), percentile=True
        )

        assert len(mse) == 3
        assert len(ssim) == 3
        assert len(dice_scores) == 3


class TestComputeAEPerformance:
    """Tests for compute_ae_performance function."""

    def test_compute_ae_performance_output_types(self):
        """Test that compute_ae_performance returns correct types."""
        torch.manual_seed(42)
        X = torch.rand(10, 100)
        X_re = torch.randn(10, 100)  # Logits

        fpr, tpr, pct, roc_auc = compute_ae_performance(X, X_re)

        assert isinstance(fpr, np.ndarray)
        assert isinstance(tpr, np.ndarray)
        assert isinstance(pct, np.ndarray)
        assert isinstance(roc_auc, float)
        assert 0.0 <= roc_auc <= 1.0

    def test_compute_ae_performance_perfect_reconstruction(self):
        """Test with perfect reconstruction."""
        torch.manual_seed(42)
        X = (torch.rand(10, 100) > 0.5).float()
        # Perfect logits: large positive for 1, large negative for 0
        X_re = torch.where(X > 0.5, torch.tensor(10.0), torch.tensor(-10.0))

        fpr, tpr, pct, roc_auc = compute_ae_performance(X, X_re)

        # Should have near-perfect AUC
        assert roc_auc > 0.99
        # Percentage improvement should be positive
        assert np.mean(pct) > 0
