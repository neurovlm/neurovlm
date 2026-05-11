"""Weighted multi-positive contrastive losses."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def multi_positive_infonce(
    brain_emb: torch.Tensor,
    text_emb: torch.Tensor,
    pos_mask: torch.Tensor,
    pos_weights: torch.Tensor,
    temperature: float = 0.07,
) -> torch.Tensor:
    brain_emb = F.normalize(brain_emb, dim=1, eps=1e-8)
    text_emb = F.normalize(text_emb, dim=1, eps=1e-8)
    logits = brain_emb @ text_emb.T
    logits = logits / temperature
    pos_mask = pos_mask.to(device=logits.device, dtype=torch.bool)
    pos_weights = pos_weights.to(device=logits.device, dtype=logits.dtype)

    log_den = torch.logsumexp(logits, dim=1)
    positive_logits = logits.masked_fill(~pos_mask, -torch.inf)
    weighted_positive_logits = positive_logits + torch.log(torch.clamp(pos_weights, min=1e-8))
    log_num = torch.logsumexp(weighted_positive_logits, dim=1)
    valid = pos_mask.any(dim=1)
    if not bool(valid.any()):
        return logits.sum() * 0.0
    return -(log_num[valid] - log_den[valid]).mean()


class MultiPositiveInfoNCELoss(torch.nn.Module):
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, brain_emb, text_emb, pos_mask, pos_weights):
        return multi_positive_infonce(brain_emb, text_emb, pos_mask, pos_weights, self.temperature)

