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
    def __init__(
            self,
            seed: Optional[int]=None,
            out: Optional[str]="probability",
            dim_neuro: Optional[int]= 28_542,
            dim_h0: Optional[int]=1024,
            dim_h1: Optional[int]=512,
            dim_latent: Optional[int]=384,
        ):
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

        # Networks
        self.encoder = nn.Sequential(
            nn.Linear(dim_neuro, dim_h0),
            nn.ReLU(),
            nn.Linear(dim_h0, dim_h1),
            nn.ReLU(),
            nn.Linear(dim_h1, dim_latent),
        )

        decoder_seq = [
            nn.Linear(dim_latent, dim_h1),
            nn.ReLU(),
            nn.Linear(dim_h1, dim_h0),
            nn.ReLU(),
            nn.Linear(dim_h0, dim_neuro),
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
