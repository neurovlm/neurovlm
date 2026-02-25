"""Tests for loss module."""

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from neurovlm.loss import FocalLoss, FocalWithLogitsLoss, InfoNCELoss, TruncatedLoss


class TestFocalLoss:
    """Tests for FocalLoss class."""

    def test_focal_loss_initialization(self):
        """Test FocalLoss can be initialized with different parameters."""
        loss_fn = FocalLoss(alpha=1.0, gamma=2.0)
        assert loss_fn.alpha == 1.0
        assert loss_fn.gamma == 2.0

    def test_focal_loss_forward(self):
        """Test forward pass returns scalar loss."""
        loss_fn = FocalLoss()
        inputs = torch.randn(10, 5)
        targets = torch.randint(0, 2, (10, 5)).float()

        loss = loss_fn(inputs, targets)

        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0  # Scalar
        assert loss.item() >= 0

    def test_focal_loss_reduces_with_gamma_zero(self):
        """Test that gamma=0 recovers standard BCE."""
        torch.manual_seed(42)
        inputs = torch.randn(10, 5)
        targets = torch.randint(0, 2, (10, 5)).float()

        focal_loss = FocalLoss(alpha=1.0, gamma=0.0)
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets)

        focal_output = focal_loss(inputs, targets)

        # Should be approximately equal to BCE when gamma=0
        assert torch.isclose(focal_output, bce_loss, rtol=1e-5)

    def test_focal_loss_perfect_predictions(self):
        """Test focal loss with perfect predictions."""
        loss_fn = FocalLoss()
        targets = torch.tensor([[1.0, 0.0, 1.0]])
        # High confidence correct predictions
        inputs = torch.tensor([[10.0, -10.0, 10.0]])

        loss = loss_fn(inputs, targets)
        # Should be very small
        assert loss.item() < 0.01


class TestFocalWithLogitsLoss:
    """Tests for FocalWithLogitsLoss class."""

    def test_focal_with_logits_initialization(self):
        """Test initialization with various parameters."""
        loss_fn = FocalWithLogitsLoss(gamma=2.0)
        assert loss_fn.gamma == 2.0
        assert loss_fn.pos_weight is None

        pos_weight = torch.tensor([1.5, 2.0])
        loss_fn_weighted = FocalWithLogitsLoss(gamma=2.0, pos_weight=pos_weight)
        assert torch.equal(loss_fn_weighted.pos_weight, pos_weight)

    def test_focal_with_logits_forward(self):
        """Test forward pass returns valid loss."""
        loss_fn = FocalWithLogitsLoss(gamma=2.0)
        logits = torch.randn(10, 5)
        targets = torch.randint(0, 2, (10, 5)).float()

        loss = loss_fn(logits, targets)

        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0
        assert loss.item() >= 0

    def test_focal_with_logits_gamma_zero(self):
        """Test that gamma=0 recovers standard BCE."""
        torch.manual_seed(42)
        logits = torch.randn(10, 5)
        targets = torch.randint(0, 2, (10, 5)).float()

        focal_loss = FocalWithLogitsLoss(gamma=0.0)
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets)

        focal_output = focal_loss(logits, targets)

        assert torch.isclose(focal_output, bce_loss, rtol=1e-4)

    def test_focal_with_logits_pos_weight(self):
        """Test that pos_weight affects the loss."""
        torch.manual_seed(42)
        logits = torch.randn(5, 3)
        targets = torch.randint(0, 2, (5, 3)).float()

        loss_no_weight = FocalWithLogitsLoss(gamma=2.0)(logits, targets)

        pos_weight = torch.tensor([2.0, 1.0, 3.0])
        loss_with_weight = FocalWithLogitsLoss(gamma=2.0, pos_weight=pos_weight)(
            logits, targets
        )

        # Losses should be different
        assert not torch.isclose(loss_no_weight, loss_with_weight)


class TestInfoNCELoss:
    """Tests for InfoNCELoss class."""

    def test_infonce_loss_initialization(self):
        """Test InfoNCELoss initialization."""
        loss_fn = InfoNCELoss(temperature=0.07)
        assert loss_fn.temperature == 0.07

    def test_infonce_loss_forward(self):
        """Test forward pass with matching batch."""
        loss_fn = InfoNCELoss()
        batch_size = 8
        dim = 128

        torch.manual_seed(42)
        image = torch.randn(batch_size, dim)
        text = torch.randn(batch_size, dim)

        loss = loss_fn(image, text)

        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0
        assert loss.item() >= 0

    def test_infonce_loss_perfect_alignment(self):
        """Test loss with perfectly aligned embeddings."""
        loss_fn = InfoNCELoss()
        batch_size = 4
        dim = 64

        # Same embeddings for image and text
        embeddings = torch.randn(batch_size, dim)
        loss = loss_fn(embeddings, embeddings)

        # Should be close to zero for perfect alignment
        assert loss.item() < 1.0

    def test_infonce_loss_symmetry(self):
        """Test that loss is symmetric (image->text == text->image)."""
        loss_fn = InfoNCELoss()
        torch.manual_seed(42)

        image = torch.randn(5, 128)
        text = torch.randn(5, 128)

        loss1 = loss_fn(image, text)
        loss2 = loss_fn(text, image)

        # Should be equal due to symmetrization
        assert torch.isclose(loss1, loss2)

    def test_infonce_loss_temperature_effect(self):
        """Test that temperature affects the loss magnitude."""
        torch.manual_seed(42)
        image = torch.randn(8, 64)
        text = torch.randn(8, 64)

        loss_low_temp = InfoNCELoss(temperature=0.01)(image, text)
        loss_high_temp = InfoNCELoss(temperature=1.0)(image, text)

        # Different temperatures should give different losses
        assert not torch.isclose(loss_low_temp, loss_high_temp)


class TestTruncatedLoss:
    """Tests for TruncatedLoss class."""

    def test_truncated_loss_initialization(self):
        """Test TruncatedLoss initialization."""
        loss_fn = TruncatedLoss(percentile=0.8, base_loss="l1")
        assert loss_fn.percentile == 0.8
        assert loss_fn.base_loss == "l1"

    def test_truncated_loss_l1_forward(self):
        """Test forward pass with L1 base loss."""
        loss_fn = TruncatedLoss(percentile=0.7, base_loss="l1")
        predicted = torch.randn(10, 100)
        target = torch.randn(10, 100)

        loss = loss_fn(predicted, target)

        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0
        assert loss.item() >= 0

    def test_truncated_loss_mse_forward(self):
        """Test forward pass with MSE base loss."""
        loss_fn = TruncatedLoss(percentile=0.7, base_loss="mse")
        predicted = torch.randn(10, 100)
        target = torch.randn(10, 100)

        loss = loss_fn(predicted, target)

        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0
        assert loss.item() >= 0

    def test_truncated_loss_keeps_easy_examples(self):
        """Test that truncated loss only keeps easy examples."""
        torch.manual_seed(42)
        loss_fn = TruncatedLoss(percentile=0.5, base_loss="mse")

        # Create targets with some easy and some hard examples
        target = torch.zeros(10, 50)
        # Half are perfect predictions (easy)
        predicted = torch.zeros(10, 50)
        predicted[5:] = torch.randn(5, 50) * 10  # Other half are bad (hard)

        loss = loss_fn(predicted, target)

        # Loss should be less than full MSE since hard examples are excluded
        full_mse = F.mse_loss(predicted, target)
        assert loss.item() < full_mse.item()

    def test_truncated_loss_percentile_100(self):
        """Test that percentile=1.0 keeps all examples."""
        torch.manual_seed(42)
        loss_truncated = TruncatedLoss(percentile=1.0, base_loss="mse")

        predicted = torch.randn(10, 50)
        target = torch.randn(10, 50)

        loss_trunc = loss_truncated(predicted, target)
        loss_full = F.mse_loss(predicted, target)

        # Should be approximately equal
        assert torch.isclose(loss_trunc, loss_full, rtol=1e-5)

    @pytest.mark.parametrize("base_loss", ["l1", "mse"])
    def test_truncated_loss_base_loss_types(self, base_loss):
        """Test both L1 and MSE base losses work."""
        loss_fn = TruncatedLoss(percentile=0.8, base_loss=base_loss)
        predicted = torch.randn(10, 50)
        target = torch.randn(10, 50)

        loss = loss_fn(predicted, target)

        assert isinstance(loss, torch.Tensor)
        assert loss.item() >= 0
