# Pytorch models

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
    def __init__(self):

        super().__init__()

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

        self.decoder = nn.Sequential(
            nn.Linear(latent, h1),
            nn.ReLU(),
            nn.Linear(h1, h0),
            nn.ReLU(),
            nn.Linear(h0, neuro_dim),
            nn.Sigmoid() # probabilistic output
        )

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

class Aligner(nn.Module):
    """Align latent text-tensor to latent neuro-tensor.

    Attributes
    ----------
    aligner : torch.nn.Sequential
        Non-linear mapping from 384 to 384.
    """
    def __init__(self):
        super().__init__()
        self.aligner = nn.Sequential(
            nn.Linear(384, 384),
            nn.ReLU(),
            nn.Linear(384, 384),
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
