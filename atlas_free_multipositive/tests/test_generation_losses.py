import torch

from atlas_free_multipositive.training.generation_losses import (
    GenerationLossConfig,
    combined_generation_loss,
    hard_topk_dice,
    latent_alignment_loss,
    topk_overlap_loss,
)


def test_generation_loss_returns_finite_scalar():
    pred = torch.randn(2, 1, 5, 5, 5, requires_grad=True)
    target = torch.zeros(2, 1, 5, 5, 5)
    target[:, :, 2, 2, 2] = 1.0

    loss, parts = combined_generation_loss(pred, target, config=GenerationLossConfig())
    loss.backward()

    assert torch.isfinite(loss)
    assert "topk_overlap" in parts
    assert pred.grad is not None


def test_topk_dice_and_overlap_on_sparse_toy_tensor():
    target = torch.zeros(1, 1, 4, 4, 4)
    pred = torch.zeros_like(target)
    target[..., 1, 1, 1] = 1.0
    pred[..., 1, 1, 1] = 1.0

    dice = hard_topk_dice(pred, target, k_percent=0.05)
    loss = topk_overlap_loss(pred, target, k_percents=[0.05])

    assert float(dice.item()) > 0.99
    assert torch.isfinite(loss)


def test_latent_alignment_loss_finite():
    text_z = torch.randn(3, 8)
    brain_z = torch.randn(3, 8)

    loss = latent_alignment_loss(text_z, brain_z)

    assert torch.isfinite(loss)

