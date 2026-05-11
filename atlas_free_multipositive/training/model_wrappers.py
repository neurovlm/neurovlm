"""Model wrappers reused by the smoke-test trainer."""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch import nn

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


class TextProjectionHead(nn.Module):
    def __init__(self, in_dim: int = 768, hidden_dim: int = 512, out_dim: int = 384):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        return self.net(x)


def build_brain_encoder(out_dim: int = 384):
    from neurovlm.gnn.ale_cnn import ALE3DCNNEncoder

    return ALE3DCNNEncoder(base_channels=8, num_blocks=2, out_dim=out_dim, dropout=0.1)


@torch.no_grad()
def encode_texts_specter(texts: list[str], *, device: str = "cpu", batch_size: int = 16) -> torch.Tensor:
    try:
        from neurovlm.models import Specter
    except ModuleNotFoundError as exc:
        if exc.name == "adapters":
            raise ModuleNotFoundError(
                "SPECTER2 text embedding requires the `adapters` package in the "
                "active notebook kernel. Install the project dependencies in this "
                "environment, or switch the notebook kernel to the repo `.conda` "
                "environment where the package is available."
            ) from exc
        raise

    specter = Specter(device=device)
    chunks = []
    for start in range(0, len(texts), batch_size):
        chunks.append(specter(texts[start : start + batch_size]).detach().cpu())
    return torch.cat(chunks, dim=0)
