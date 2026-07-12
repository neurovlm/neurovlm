"""Small projection heads used by the ALE 3D CNN experiments."""

from __future__ import annotations

from torch import Tensor, nn


class TextProjHead(nn.Module):
    """Project SPECTER text embeddings into the brain-model latent space."""

    def __init__(
        self,
        in_dim: int = 768,
        hidden_dim: int = 512,
        out_dim: int = 384,
    ):
        super().__init__()
        self.aligner = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.aligner(x)
