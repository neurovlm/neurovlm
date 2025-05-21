# Pytorch models
from typing import Optional
import torch
from torch import nn

class NeuroAutoEncoder(nn.Module):
    """Autoencoder for neuro-vectors.

    Attributes
    ----------
    encoder : torch.nn.Sequential
        Encoder network.
    decoder : torch.nn.Sequential
        Decoder network.
    """
    def __init__(self, seed: Optional[int]=None, out: Optional[str]="probability"):
        """Define network.

        Parameters
        ----------
        seed : int, optional, default: None
            Random seed for weight initialization.
        out : {"probability", "logits"}, optional, default: "probability"
            Whether the models returns logits or probabilities. If logits are returned,
            use BCEWithLogitsLoss. If probabilities are return, use BCELoss.
        """
        super().__init__()

        if seed is not None:
            torch.manual_seed(seed)

        # Layer dimensions
        h0 = 1024
        h1 = 512
        latent = 384
        neuro_dim = 28_542

        # Networks
        self.encoder = nn.Sequential(
            nn.Linear(neuro_dim, h0),
            nn.ReLU(),
            nn.Linear(h0, h1),
            nn.Linear(h1, latent),
        )

        decoder_seq = [
            nn.Linear(latent, h1),
            nn.ReLU(),
            nn.Linear(h1, h0),
            nn.ReLU(),
            nn.Linear(h0, neuro_dim)
        ]

        assert "prob" in out or "logit" in out

        if "prob" in out:
            decoder_seq.append(nn.Sigmoid())
        # else returns logits

        self.decoder = nn.Sequential(*decoder_seq)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        X : 2d torch.tensor
            Batched neuro-tensors.

        Returns
        -------
        torch.tensor
            Probability of neuro-activation.
        """
        return self.decoder(self.encoder(X))

class TextAligner(nn.Module):
    """Align latent text-tensor to latent neuro-tensor.

    Attributes
    ----------
    aligner : torch.nn.Sequential
        Non-linear mapping from 384 to 384.
    """
    def __init__(
        self,
        latent_text_dim: Optional[int]=384,
        hidden_dim: Optional[int]=384,
        latent_neuro_dim: Optional[int]=384,
        seed: Optional[int]=None
    ):
        """Define network.

        Parameters
        ----------
        latent_dim : int, optional, default: 384
            Input and output size.
        seed : int, optional, default: None
            Random seed for weight initialization.
        """
        super().__init__()
        if seed is not None:
            torch.manual_seed(seed)
        self.aligner = nn.Sequential(
            nn.Linear(latent_text_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_neuro_dim),
        )
    def forward(self, X: torch.tensor) -> torch.tensor:
        """Forward pass.

        Parameters
        ----------
        X : 2d torch.tensor
            Batched text-tensors.

        Returns
        -------
        torch.tensor
            Aligned text-tensor.
        """
        return self.aligner(X)
