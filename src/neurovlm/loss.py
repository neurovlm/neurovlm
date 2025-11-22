"""Non-standard torch loss functions."""
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * bce_loss
        return focal_loss.mean()

class InfoNCELoss(torch.nn.Module):
    """Compute InfoNCE loss between image and text embeddings."""
    def __init__(self,  temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, image, text):
        # Normalize embeddings
        image = F.normalize(image, dim=1)
        text = F.normalize(text, dim=1)

        # Compute similarity matrix: (batch_size, batch_size)
        logits = torch.matmul(image, text.T) / self.temperature

        # Labels are indices of the correct pairs
        batch_size = image.size(0)
        labels = torch.arange(batch_size, device=image.device)

        # Cross-entropy loss in both directions (symmetrized)
        loss_i2t = F.cross_entropy(logits, labels)
        loss_t2i = F.cross_entropy(logits.T, labels)

        return (loss_i2t + loss_t2i) / 2

class TruncatedLoss(nn.Module):
    def __init__(self, percentile=0.8, base_loss="l1"):
        super().__init__()
        self.percentile = percentile
        self.base_loss = base_loss

    def forward(self, predicted, target):
        # either smooth l1, here it's huber, or mse loss
        if self.base_loss == "l1":
            loss_per_sample = F.smooth_l1_loss(predicted, target, reduction='none', beta=1.).mean(dim=1)
        else:
            loss_per_sample = F.mse_loss(predicted, target, reduction='none').mean(dim=1)

        # keep the easiest x% of examples
        threshold = torch.quantile(loss_per_sample, self.percentile)
        easy_mask = loss_per_sample <= threshold

        if easy_mask.sum() > 0:
            return loss_per_sample[easy_mask].mean()
        else:
            return loss_per_sample.mean()
