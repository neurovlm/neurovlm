# Pytorch models
from typing import Optional
import torch
from torch import nn
from adapters import AutoAdapterModel
from transformers import AutoTokenizer
from transformers.utils.logging import disable_progress_bar
disable_progress_bar()

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

class Specter:
    """Wrapper for Specter model."""
    def __init__(self, model="allenai/specter2_aug2023refresh", adapter="adhoc_query", orthgonalize=True):
        """Initialize.

        Parameters
        ----------
        model : {"allenai/specter2_aug2023refresh", allenai/specter2"}
            Base model.
        adapter : {"adhoc_query", "classification", "regression", "proximity"}
            Adapter to attach to the model, for specific use cases.
        """
        tokenizer = AutoTokenizer.from_pretrained(f'{model}_base')
        self.sep_token = tokenizer.sep_token
        self.tokenizer = lambda text : tokenizer(
            text,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
            return_token_type_ids=False,
        )

        self.specter = AutoAdapterModel.from_pretrained(f'{model}_base')
        self.specter.load_adapter(f"{model}_{adapter}", source="hf", load_as="specter2", set_active=True)

        if orthgonalize:
            self.ref = self.specter(**self.tokenizer("")).last_hidden_state[:, 0]
            self.ref= self.ref / self.ref.norm()
            self.f_transform = self.orthogonalize
        else:
            self.f_transform = lambda i : i

    def __call__(self, X):
        """Pass text through the model.

        Parameters
        ----------
        X : list of str
            Text to encode.

        Returns
        -------
        embeddings : torch.tensor
            Latent text encodings.
        """
        text = [i['title'] + self.sep_token + (i.get('abstract') or '') for i in X]
        tokens  = self.tokenizer(text)
        embeddings = self.specter(**tokens)
        embeddings = self.f_transform(embeddings.last_hidden_state[:, 0])
        return embeddings

    def orthogonalize(self, embedding):
        proj = (embedding @ self.ref.T) * self.ref
        return embedding - proj
