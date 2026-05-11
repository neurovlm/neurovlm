import torch

from atlas_free_multipositive.training.losses import multi_positive_infonce


def test_multi_positive_infonce_backward():
    brain = torch.randn(3, 8, requires_grad=True)
    text = torch.randn(6, 8, requires_grad=True)
    mask = torch.zeros(3, 6, dtype=torch.bool)
    mask[0, [0, 1]] = True
    mask[1, [2]] = True
    mask[2, [3, 4, 5]] = True
    weights = mask.float()

    loss = multi_positive_infonce(brain, text, mask, weights, temperature=0.1)
    loss.backward()

    assert torch.isfinite(loss)
    assert brain.grad is not None
    assert text.grad is not None


def test_multi_positive_infonce_ignores_rows_without_positives():
    brain = torch.randn(2, 4, requires_grad=True)
    text = torch.randn(2, 4, requires_grad=True)
    mask = torch.tensor([[True, False], [False, False]])
    weights = mask.float()

    loss = multi_positive_infonce(brain, text, mask, weights)

    assert torch.isfinite(loss)

