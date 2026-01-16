# Pytorch models
from typing import Optional
import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F
from adapters import AutoAdapterModel
from transformers import AutoTokenizer, AutoModel
from transformers.utils.logging import disable_progress_bar
disable_progress_bar()

class NormalizeLayer(nn.Module):
    def forward(self, x):
        return F.normalize(x, dim=1)

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
        dim_neuro: Optional[int]=28_542,
        dim_h0: Optional[int]=1024,
        dim_h1: Optional[int]=512,
        dim_latent: Optional[int]=384,
        activation_fn: Optional[callable]=None,
        normalize_latent: Optional[bool]=False
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

        if activation_fn is None:
            activation_fn = nn.ReLU()

        # Networks
        self.encoder = nn.Sequential(
            nn.Linear(dim_neuro, dim_h0),
            activation_fn,
            nn.Linear(dim_h0, dim_h1),
            activation_fn,
            nn.Linear(dim_h1, dim_latent),
        )

        if normalize_latent:
            self.encoder.append(NormalizeLayer())

        decoder_seq = [
            nn.Linear(dim_latent, dim_h1),
            activation_fn,
            nn.Linear(dim_h1, dim_h0),
            activation_fn,
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

class ProjHead(nn.Module):
    """Align latent tensors.

    Attributes
    ----------
    aligner : torch.nn.Sequential
        Non-linear mapping from 384 to 384.
    """

    def __init__(
        self,
        latent_in_dim: Optional[int]=768,
        hidden_dim: Optional[int]=512,
        latent_out_dim: Optional[int]=384,
        seed: Optional[int]=None
    ):
        """Define network.

        Parameters
        ----------
        latent_in_dim : int, optional, default: 384
            Input size.
        hidden_dim : int, optional, default: 512
            Hidden layer size.
        latent_out_dim : int, optional, default: 384
            Output size.
        seed : int, optional, default: None
            Random seed for weight initialization.
        """
        super().__init__()
        if seed is not None:
            torch.manual_seed(seed)

        self.aligner = nn.Sequential(
            nn.Linear(latent_in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_out_dim),
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
    def __init__(self, model="allenai/specter2_aug2023refresh", adapter="adhoc_query",
                 orthgonalize=True, pooling=None, device="cpu"):
        """Initialize.

        Parameters
        ----------
        model : {"allenai/specter2_aug2023refresh", allenai/specter2"}
            Base model.
        adapter : {"adhoc_query", "classification", "regression", "proximity"}
            Adapter to attach to the model, for specific use cases.
        """
        self.device = torch.device(device)
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
        self.pooling = pooling

        if adapter is None:
            # no adapter
            self.specter = AutoModel.from_pretrained(f'{model}_base')
        elif "/" in adapter:
            # custom adapters, e.g. neurospecter trained by Jerjes
            self.specter = AutoModel.from_pretrained(f'{model}_base')
            self.specter.load_adapter(adapter)
        else:
            # specter2 supported adapters: proximity, adhoc_query, regression, classification
            # Map the proximity adapter explicitly to the HF id "allenai/specter2".
            # Other adapters follow the naming pattern "{model}_{adapter}".
            if adapter == "proximity":
                adapter_id = "allenai/specter2"
            else:
                adapter_id = f"{model}_{adapter}"

            self.specter = AutoAdapterModel.from_pretrained(f'{model}_base')
            self.specter.load_adapter(adapter_id, source="hf", load_as="specter2", set_active=True)

        self.specter = self.specter.to(self.device).eval()

        if orthgonalize:
            with torch.inference_mode():
                tokens = {k: v.to(self.device) for k, v in self.tokenizer("").items()}
                self.ref = self.pool(
                    self.specter(**tokens).last_hidden_state,
                    tokens["attention_mask"],
                    method=self.pooling
                )
                self.ref= self.ref / self.ref.norm()
            self.f_transform = self.orthogonalize
        else:
            self.f_transform = lambda i : i

    def __call__(self, X: pd.DataFrame | dict | list | str) -> torch.Tensor:
        """Pass text through the model.

        Parameters
        ----------
        X : DataFrame | dict | list[str] | list[dict] | str
            Text to encode. Accepts:
            - pandas DataFrame with columns 'title' and 'abstract' (or 'summary').
            - dict with keys 'title' and optional 'abstract'/'summary'.
            - list of strings or list of dicts as above.
            - a single string.

        Returns
        -------
        embeddings : torch.tensor
            Latent text encodings.
        """
        if isinstance(X, pd.DataFrame):
            abs_col = (
                'abstract' if 'abstract' in X.columns
                else ('summary' if 'summary' in X.columns else None)
            )
            titles = X['title'].fillna('').astype(str).tolist() if 'title' in X.columns else [''] * len(X)
            abstracts = (
                X[abs_col].fillna('').astype(str).tolist() if abs_col is not None else [''] * len(X)
            )
            text = [t + self.sep_token + a for t, a in zip(titles, abstracts)]
        elif isinstance(X, dict):
            title = X.get('title') or ''
            abstract = X.get('abstract') or X.get('summary') or ''
            text = [f"{title}{self.sep_token}{abstract}"]
        elif isinstance(X, (list, tuple)):
            if len(X) > 0 and isinstance(X[0], dict):
                text = [
                    (d.get('title') or '') + self.sep_token + (d.get('abstract') or d.get('summary') or '')
                    for d in X
                ]
            else:
                text = list(X)
        else:
            text = [str(X)]

        tokens ={k: v.to(self.device) for k, v in self.tokenizer(text).items()}
        with torch.inference_mode():
            embeddings = self.pool(
                self.specter(**tokens).last_hidden_state,
                tokens["attention_mask"],
                method=self.pooling
            )
            embeddings = self.f_transform(embeddings)
        return embeddings

    def orthogonalize(self, embedding: torch.Tensor) -> torch.Tensor:
        proj = (embedding @ self.ref.T) * self.ref
        return embedding - proj

    def pool(self, hidden: torch.Tensor, attention_mask: torch.Tensor, method: Optional[str]=None) -> torch.Tensor:
        """Pool embedding matrix."""

        mask = attention_mask.unsqueeze(-1)

        if method == "mean": # mean pooling
            emb = (hidden * mask).sum(dim=1) / mask.sum(dim=1)
        elif method == "max": # max pooling
            hidden_masked = hidden.masked_fill(mask == 0, -1e9)
            emb = hidden_masked.max(dim=1).values
        elif method == "mean_max": # mean + max
            mean_emb = (hidden * mask).sum(dim=1) / mask.sum(dim=1)
            hidden_masked = hidden.masked_fill(mask == 0, -1e9)
            max_emb = hidden_masked.max(dim=1).values
            emb = torch.cat([mean_emb, max_emb], dim=-1)
        elif method == "attention": # attention pooling (self-weighted)
            query = (hidden * mask).sum(dim=1, keepdim=True) / mask.sum(dim=1, keepdim=True)
            scores = torch.matmul(hidden, query.transpose(1, 2)).squeeze(-1)
            scores = scores.masked_fill(attention_mask == 0, -1e9)
            weights = F.softmax(scores, dim=-1).unsqueeze(-1)
            emb = (hidden * weights).sum(dim=1)
        else:
            emb = hidden[:, 0]

        return emb

class ConceptClf(nn.Module):
    """Predict concepts from latent neuro embeddings."""
    def __init__(self, d_out):
        super().__init__()
        self.seq = nn.Sequential(
            nn.Linear(384, 768),
            nn.ReLU(),
            nn.Linear(768, 1526),
            nn.ReLU(),
            nn.Linear(1526, d_out)
        )
    def forward(self, X: torch.tensor):
        return self.seq(X)
        