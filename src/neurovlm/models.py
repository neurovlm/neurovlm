# Pytorch models
from typing import Optional
import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F

from neurovlm.model_registry import (
    ModelDomain,
    ModelFamily,
    ModelLoader,
    ModelSpec,
    ModelTask,
    ModelVariant,
    resolve_model_spec,
)

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

    @staticmethod
    def from_pretrained() -> nn.Module:
        """Load pretrained autoencoder."""
        from neurovlm.retrieval_resources import _load_autoencoder
        return _load_autoencoder()

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

    @staticmethod
    def from_pretrained(model_name: str) -> nn.Module:
        """Load pretrained autoencoder.

        Parameters
        ----------
        model_name : str, {"text_infonce", "image_infonce", "text_mse"}
        """
        match model_name:
            # Contrastive models
            case "text_infonce":
                from neurovlm.retrieval_resources import _proj_head_text_infonce
                return _proj_head_text_infonce()
            case "image_infonce":
                from neurovlm.retrieval_resources import _proj_head_image_infonce
                return _proj_head_image_infonce()
            # MSE text-to-brain model
            case "text_mse":
                from neurovlm.retrieval_resources import _proj_head_text_mse
                return _proj_head_text_mse()

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
        import os
        # Prevent transformers from importing TensorFlow/Flax - on macOS this loads a
        # 662MB TF dylib that takes ~10 minutes to initialize. Must be set before import.
        os.environ.setdefault("USE_TF", "0")
        os.environ.setdefault("USE_FLAX", "0")

        from adapters import AutoAdapterModel
        from transformers import AutoTokenizer, AutoModel
        from transformers.utils.logging import disable_progress_bar
        disable_progress_bar()

        self.device = torch.device(device)
        # Prefer an existing Hugging Face cache without making a network HEAD
        # request on every comparison run. Fall back to the normal download
        # path when the tokenizer is not cached yet.
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                f'{model}_base', local_files_only=True
            )
        except Exception:
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

            # Try local cache first to avoid network hangs (e.g. slow HuggingFace connection).
            # Falls back to downloading only if the model is not cached.
            try:
                self.specter = AutoAdapterModel.from_pretrained(f'{model}_base', local_files_only=True)
                self.specter.load_adapter(adapter_id, source="hf", load_as="specter2", set_active=True, local_files_only=True)
            except Exception:
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
                self.ref = self.ref / self.ref.norm()
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

    @staticmethod
    def from_pretrained() -> nn.Module:
        """Load pretrained Specter - an alias to init to keep api consistent."""
        return Specter()

    def to(self, device):
        """Move model to device."""
        self.device = device
        self.specter = self.specter.to(device).eval()
        self.ref = self.ref.to(device)
        return self


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

    @staticmethod
    def from_pretrained() -> nn.Module:
        """Load pretrained model - an alias to init to keep api consistent."""
        raise NotImplementedError


# Unified interface for all packaged models
def _load_resolved_model(spec: ModelSpec):
    """Dispatch a resolved model specification to its resource loader."""

    if spec.loader is ModelLoader.CNN_AUTOENCODER:
        from neurovlm.retrieval_resources import _load_cnn_autoencoder

        return _load_cnn_autoencoder(spec.loader_variant)
    if spec.loader is ModelLoader.CNN_CONTRASTIVE:
        from neurovlm.retrieval_resources import _load_cnn_contrastive

        return _load_cnn_contrastive(spec.loader_variant)
    if spec.loader is ModelLoader.CNN_TEXT_TO_BRAIN:
        from neurovlm.retrieval_resources import _load_cnn_text_to_brain

        return _load_cnn_text_to_brain(spec.loader_variant)
    if spec.loader is ModelLoader.MLP_TEXT_INFONCE:
        return ProjHead().from_pretrained("text_infonce")
    if spec.loader is ModelLoader.MLP_IMAGE_INFONCE:
        return ProjHead().from_pretrained("image_infonce")
    if spec.loader is ModelLoader.MLP_TEXT_MSE:
        return ProjHead().from_pretrained("text_mse")
    if spec.loader is ModelLoader.MLP_AUTOENCODER:
        return NeuroAutoEncoder.from_pretrained()
    if spec.loader is ModelLoader.MLP_SPECTER:
        return Specter()
    if spec.loader is ModelLoader.MLP_NEURO_QFORMER:
        from neurovlm.retrieval_resources import _load_neuro_qformer

        return _load_neuro_qformer()
    if spec.loader is ModelLoader.MLP_NEURO_ADAPTER:
        from neurovlm.retrieval_resources import _load_neuro_adapter

        return _load_neuro_adapter()
    raise RuntimeError(f"Model registry contains unsupported loader {spec.loader!r}")


def load_model(
    name: str | None = None,
    *,
    family: ModelFamily | str = ModelFamily.MLP,
    task: ModelTask | str | None = None,
    domain: ModelDomain | str | None = None,
    variant: ModelVariant | str | None = None,
):
    """Load a packaged model by legacy name or structured fields.

    Parameters
    ----------
    name : str, optional
        Existing packaged-model name.  Every name supported before the
        structured API remains an alias to the same checkpoint.
    family : {"mlp", "cnn"}, default: "mlp"
        Architecture family.  MLP remains the global default.
    task : str, optional
        Structured task identifier. Required when ``name`` is omitted.
    domain : {"pubmed", "nilearn", "neurovault"}, optional
        Required for CNN contrastive/text-to-brain models and for fine-tuned
        CNN autoencoders.
    variant : str, optional
        CNN models default to ``mixed_baseline``. Domain-specialized CNN
        checkpoints require the explicit value ``finetuned``.

    Returns
    -------
    model

    Examples
    --------
    Existing calls continue to work:

    >>> model = load_model("autoencoder")

    Structured CNN selection defaults to the mixed baseline:

    >>> model = load_model(family="cnn", task="contrastive", domain="nilearn")
    """
    spec = resolve_model_spec(
        name,
        family=family,
        task=task,
        domain=domain,
        variant=variant,
    )
    return _load_resolved_model(spec)
