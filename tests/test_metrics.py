"""Tests for metrics module."""

import numpy as np
import pytest
import torch

from neurovlm.metrics import (
    bleu,
    compute_metrics,
    dice,
    dice_top_k,
    pearson_correlation,
    psnr,
    recall_at_k,
    recall_curve,
    rouge,
    bernoulli_bce,
    bits_per_pixel,
    compute_ae_performance,
)


class TestBleu:
    """Tests for BLEU score computation."""

    def test_bleu_identical(self):
        """Identical hypothesis and reference should score near 1."""
        ref = ["the quick brown fox jumps over the lazy dog"]
        hyp = "the quick brown fox jumps over the lazy dog"
        score = bleu(ref, hyp)
        assert score > 0.99

    def test_bleu_no_overlap(self):
        """Completely different hypothesis should score near 0."""
        ref = ["the quick brown fox"]
        hyp = "lorem ipsum dolor sit"
        score = bleu(ref, hyp)
        assert score < 0.05

    def test_bleu_partial_overlap(self):
        """Partial overlap should give an intermediate score."""
        ref = ["working memory activates the prefrontal cortex"]
        hyp = "working memory involves the prefrontal cortex region"
        score = bleu(ref, hyp)
        assert 0.0 < score < 1.0

    def test_bleu_multiple_references(self):
        """Multiple references should not raise and return a valid score."""
        refs = [
            "the prefrontal cortex supports working memory",
            "working memory relies on prefrontal cortex activity",
        ]
        hyp = "the prefrontal cortex is involved in working memory"
        score = bleu(refs, hyp)
        assert 0.0 <= score <= 1.0

    def test_bleu_unigram(self):
        """BLEU-1 (n=1) should return a valid float."""
        score = bleu(["memory and attention"], "memory and learning", n=1)
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0

    def test_bleu_returns_float(self):
        """Return type should always be float."""
        score = bleu(["some reference text"], "some hypothesis text")
        assert isinstance(score, float)


class TestRouge:
    """Tests for ROUGE score computation."""

    def test_rouge_identical(self):
        """Identical texts should give perfect F-measures."""
        text = "working memory activates the prefrontal cortex"
        scores = rouge(text, text)
        assert scores["rouge1"]["fmeasure"] > 0.99
        assert scores["rouge2"]["fmeasure"] > 0.99
        assert scores["rougeL"]["fmeasure"] > 0.99

    def test_rouge_no_overlap(self):
        """Completely different texts should give near-zero scores."""
        scores = rouge("the quick brown fox", "lorem ipsum dolor sit")
        assert scores["rouge1"]["fmeasure"] < 0.05

    def test_rouge_keys(self):
        """Output must contain rouge1, rouge2, rougeL with the right sub-keys."""
        scores = rouge("reference text here", "hypothesis text here")
        for key in ("rouge1", "rouge2", "rougeL"):
            assert key in scores
            assert set(scores[key].keys()) == {"precision", "recall", "fmeasure"}

    def test_rouge_values_in_range(self):
        """All precision/recall/fmeasure values must be in [0, 1]."""
        scores = rouge(
            "the prefrontal cortex supports working memory",
            "working memory involves prefrontal regions",
        )
        for key in ("rouge1", "rouge2", "rougeL"):
            for metric in ("precision", "recall", "fmeasure"):
                val = scores[key][metric]
                assert 0.0 <= val <= 1.0, f"{key}.{metric} = {val} out of range"

    def test_rouge_partial_overlap(self):
        """Partial overlap should give intermediate scores for rouge1."""
        scores = rouge(
            "memory and attention are cognitive functions",
            "attention and perception involve brain networks",
        )
        assert 0.0 < scores["rouge1"]["fmeasure"] < 1.0


class TestPearsonCorrelation:
    """Tests for pearson_correlation."""

    def test_perfect_positive(self):
        """Identical arrays should give r = 1.0."""
        arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        assert np.isclose(pearson_correlation(arr, arr), 1.0)

    def test_perfect_negative(self):
        """Perfectly anti-correlated arrays should give r = -1.0."""
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([3.0, 2.0, 1.0])
        assert np.isclose(pearson_correlation(a, b), -1.0)

    def test_zero_correlation(self):
        """Orthogonal arrays should give r near 0."""
        a = np.array([1.0, -1.0, 1.0, -1.0])
        b = np.array([1.0, 1.0, -1.0, -1.0])
        r = pearson_correlation(a, b)
        assert np.isclose(r, 0.0, atol=1e-10)

    def test_constant_array_returns_zero(self):
        """Constant array (zero variance) should return 0.0 without error."""
        a = np.ones(10)
        b = np.random.rand(10)
        assert pearson_correlation(a, b) == 0.0

    def test_2d_input_flattened(self):
        """2D arrays should be flattened before computing r."""
        rng = np.random.default_rng(0)
        a = rng.random((5, 10))
        r = pearson_correlation(a, a)
        assert np.isclose(r, 1.0)

    def test_output_in_range(self):
        """Correlation should always be in [-1, 1]."""
        rng = np.random.default_rng(42)
        a = rng.random(100)
        b = rng.random(100)
        r = pearson_correlation(a, b)
        assert -1.0 <= r <= 1.0

    def test_torch_tensor_input(self):
        """torch.Tensor inputs should be accepted via numpy conversion."""
        a = torch.rand(50)
        b = torch.rand(50)
        r = pearson_correlation(a.numpy(), b.numpy())
        assert isinstance(r, float)


class TestPsnr:
    """Tests for psnr (Peak Signal-to-Noise Ratio)."""

    def test_identical_images_infinite(self):
        """Identical arrays should give infinite (or very large) PSNR."""
        arr = np.random.rand(100)
        result = psnr(arr, arr)
        assert result == float("inf") or result > 100.0

    def test_perfect_reconstruction_high_psnr(self):
        """Near-perfect reconstruction should give high PSNR."""
        rng = np.random.default_rng(0)
        y_true = rng.random(1000)
        noise = rng.random(1000) * 1e-6
        result = psnr(y_true, y_true + noise)
        assert result > 100.0

    def test_poor_reconstruction_low_psnr(self):
        """Very noisy prediction should give lower PSNR."""
        rng = np.random.default_rng(1)
        y_true = rng.random(1000)
        y_pred = rng.random(1000)
        result = psnr(y_true, y_pred)
        assert result < 20.0

    def test_returns_float(self):
        """Return type should be float."""
        a = np.random.rand(50)
        b = np.random.rand(50)
        assert isinstance(psnr(a, b), float)

    def test_data_range_parameter(self):
        """Different data_range values should produce different PSNR."""
        rng = np.random.default_rng(2)
        y_true = rng.random(100)
        y_pred = y_true + rng.random(100) * 0.1
        p1 = psnr(y_true, y_pred, data_range=1.0)
        p2 = psnr(y_true, y_pred, data_range=2.0)
        assert not np.isclose(p1, p2)


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
        assert np.isclose(score, 1.0, atol=1e-6)

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
        assert np.isclose(score, 1.0, atol=1e-6)


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
