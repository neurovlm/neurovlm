"""QFormer implementation for BLIP-2 like brain-to-text generation."""

import torch
from torch import nn
import torch.nn.functional as F


class QFormer(nn.Module):

    def __init__(
        self,
        image_dim,
        semantic_dim,
        lm_dim,
        num_queries=32,
        hidden_dim=512,
        num_heads=8,
        num_layers=6,
        dropout=0.05
    ):
        super().__init__()
        self.num_queries = num_queries
        self.raw_proj = nn.Sequential(
            nn.LayerNorm(image_dim),
            nn.Linear(image_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.semantic_proj = nn.Sequential(
            nn.LayerNorm(semantic_dim),
            nn.Linear(semantic_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.query_tokens = nn.Parameter(torch.randn(1, num_queries, hidden_dim) * 0.02)
        layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerDecoder(layer, num_layers=num_layers)
        self.to_lm = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, lm_dim),
        )
        self.align_head = nn.Sequential(
            nn.LayerNorm(lm_dim),
            nn.Linear(lm_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, semantic_dim),
        )

    def forward(self, raw_images, semantic_images):
        batch = raw_images.size(0)
        dtype = self.query_tokens.dtype
        raw_images = raw_images.to(dtype=dtype)
        semantic_images = semantic_images.to(dtype=dtype)
        raw_mem = self.raw_proj(raw_images)
        sem_mem = self.semantic_proj(semantic_images)
        mem = torch.stack([raw_mem, sem_mem], dim=1)
        q = self.query_tokens.expand(batch, -1, -1)
        out = self.transformer(q, mem)
        return self.to_lm(out)

    def semantic_from_visual(self, visual_tokens):
        x = visual_tokens.to(dtype=self.query_tokens.dtype).mean(dim=1)
        return F.normalize(self.align_head(x).float(), dim=1)


def _canonical_basis_name(basis: str | None) -> str:
    basis = "all" if basis is None else str(basis).lower()
    return {
        "networks": "network",
        "regions": "region",
        "functions": "function",
    }.get(basis, basis)


class CanonicalProjection(nn.Module):
    """Frozen soft projection onto canonical semantic embeddings."""

    bank_names = ("all", "network", "region", "function")

    def __init__(
        self,
        canonical_banks: dict[str, torch.Tensor] | None = None,
        *,
        projection_temp: float = 0.05,
        enabled: bool = True,
        semantic_dim: int = 384,
    ):
        super().__init__()
        self.projection_temp = float(projection_temp)
        self.enabled = bool(enabled)
        self.semantic_dim = int(semantic_dim)

        canonical_banks = canonical_banks or {}
        for name in self.bank_names:
            bank = canonical_banks.get(name)
            if bank is None:
                bank = torch.empty(0, self.semantic_dim, dtype=torch.float32)
            bank = torch.as_tensor(bank, dtype=torch.float32)
            if bank.ndim != 2:
                raise ValueError(f"Canonical bank {name!r} must be 2-D, got shape {tuple(bank.shape)}")
            if bank.shape[1] != self.semantic_dim:
                raise ValueError(
                    f"Canonical bank {name!r} has dim {bank.shape[1]}, expected {self.semantic_dim}"
                )
            self.register_buffer(f"{name}_bank", F.normalize(bank, dim=1), persistent=True)

    def get_bank(self, basis: str | None = "all") -> torch.Tensor:
        basis = _canonical_basis_name(basis)
        if basis not in self.bank_names:
            valid = ", ".join(self.bank_names)
            raise ValueError(f"Unknown canonical basis {basis!r}; valid options: {valid}")
        bank = getattr(self, f"{basis}_bank")
        if bank.numel() == 0:
            raise ValueError(f"Canonical basis {basis!r} is empty")
        return bank

    def forward(
        self,
        x: torch.Tensor,
        *,
        basis: str | None = "all",
        temperature: float | None = None,
        enabled: bool | None = None,
        return_weights: bool = False,
    ):
        enabled = self.enabled if enabled is None else bool(enabled)
        x = F.normalize(x.float(), dim=1)
        if not enabled:
            return (x, None) if return_weights else x

        bank = self.get_bank(basis).to(device=x.device, dtype=torch.float32)
        temperature = self.projection_temp if temperature is None else float(temperature)
        logits = x @ bank.T
        weights = torch.softmax(logits / temperature, dim=-1)
        z = F.normalize(weights @ bank, dim=1)
        if return_weights:
            return z, weights
        return z


class NeuroQFormer(nn.Module):
    """QFormer plus frozen image-semantic projection and canonical projection.

    The wrapped module accepts raw autoencoder latents at inference. If semantic
    inputs are not supplied, it computes image semantics with ``proj_head_image``
    and optionally projects them onto a frozen canonical embedding bank before
    calling the underlying QFormer.
    """

    def __init__(
        self,
        qformer: QFormer | None = None,
        proj_head_image: nn.Module | None = None,
        canonical_banks: dict[str, torch.Tensor] | None = None,
        *,
        image_dim: int = 384,
        semantic_dim: int = 384,
        lm_dim: int = 1024,
        num_queries: int = 32,
        hidden_dim: int = 512,
        num_heads: int = 8,
        num_layers: int = 6,
        dropout: float = 0.05,
        raw_image_input_mode: str = "raw",
        projection_temp: float = 0.05,
        use_canonical_projection: bool = True,
        freeze_projection: bool = True,
    ):
        super().__init__()
        if raw_image_input_mode not in {"raw", "projected", "zero"}:
            raise ValueError("raw_image_input_mode must be 'raw', 'projected', or 'zero'")

        self.qformer = qformer or QFormer(
            image_dim=image_dim,
            semantic_dim=semantic_dim,
            lm_dim=lm_dim,
            num_queries=num_queries,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout=dropout,
        )
        self.proj_head_image = proj_head_image or nn.Identity()
        self.canonical_projection = CanonicalProjection(
            canonical_banks,
            projection_temp=projection_temp,
            enabled=use_canonical_projection,
            semantic_dim=semantic_dim,
        )
        self.raw_image_input_mode = raw_image_input_mode

        if freeze_projection:
            for module in (self.proj_head_image, self.canonical_projection):
                for param in module.parameters():
                    param.requires_grad = False

    @property
    def projection_temp(self) -> float:
        return self.canonical_projection.projection_temp

    @projection_temp.setter
    def projection_temp(self, value: float) -> None:
        self.canonical_projection.projection_temp = float(value)

    @property
    def use_canonical_projection(self) -> bool:
        return self.canonical_projection.enabled

    @use_canonical_projection.setter
    def use_canonical_projection(self, value: bool) -> None:
        self.canonical_projection.enabled = bool(value)

    def project_semantic(
        self,
        raw_images: torch.Tensor,
        *,
        basis: str | None = "all",
        projection_temp: float | None = None,
        use_canonical_projection: bool | None = None,
        return_weights: bool = False,
    ):
        image_sem = F.normalize(self.proj_head_image(raw_images.float()), dim=1)
        return self.canonical_projection(
            image_sem,
            basis=basis,
            temperature=projection_temp,
            enabled=use_canonical_projection,
            return_weights=return_weights,
        )

    def forward(
        self,
        raw_images: torch.Tensor,
        semantic_images: torch.Tensor | None = None,
        *,
        basis: str | None = "all",
        projection_temp: float | None = None,
        use_canonical_projection: bool | None = None,
        raw_image_input_mode: str | None = None,
        return_semantic: bool = False,
    ):
        raw_images = torch.as_tensor(raw_images)
        if raw_images.ndim == 1:
            raw_images = raw_images.reshape(1, -1)

        if semantic_images is None:
            semantic_images = self.project_semantic(
                raw_images,
                basis=basis,
                projection_temp=projection_temp,
                use_canonical_projection=use_canonical_projection,
            )
        else:
            semantic_images = torch.as_tensor(semantic_images)
            if semantic_images.ndim == 1:
                semantic_images = semantic_images.reshape(1, -1)

        mode = self.raw_image_input_mode if raw_image_input_mode is None else raw_image_input_mode
        if mode == "raw":
            raw_input = raw_images
        elif mode == "projected":
            raw_input = semantic_images
        elif mode == "zero":
            raw_input = torch.zeros_like(raw_images)
        else:
            raise ValueError("raw_image_input_mode must be 'raw', 'projected', or 'zero'")

        tokens = self.qformer(raw_input, semantic_images)
        if return_semantic:
            return tokens, semantic_images
        return tokens

    @classmethod
    def from_state_dict_payload(
        cls,
        payload: dict,
        *,
        map_location: str | torch.device = "cpu",
    ) -> "NeuroQFormer":
        """Instantiate from a saved payload or raw state dict."""
        state = payload.get("state_dict", payload)
        config = payload.get("config", {})

        def _bank(name: str) -> torch.Tensor:
            key = f"canonical_projection.{name}_bank"
            return state.get(key, torch.empty(0, config.get("semantic_dim", 384)))

        proj_head_image = None
        if any(str(key).startswith("proj_head_image.") for key in state):
            from neurovlm.models import ProjHead

            proj_head_image = ProjHead(384, 384, 384)

        canonical_banks = {name: _bank(name) for name in CanonicalProjection.bank_names}
        model = cls(
            proj_head_image=proj_head_image,
            canonical_banks=canonical_banks,
            image_dim=config.get("image_dim", 384),
            semantic_dim=config.get("semantic_dim", 384),
            lm_dim=config.get("lm_dim", 1024),
            num_queries=config.get("num_queries", 32),
            hidden_dim=config.get("hidden_dim", 512),
            num_heads=config.get("num_heads", 8),
            num_layers=config.get("num_layers", 6),
            dropout=config.get("dropout", 0.05),
            raw_image_input_mode=config.get("raw_image_input_mode", "raw"),
            projection_temp=config.get("projection_temp", 0.05),
            use_canonical_projection=config.get("use_canonical_projection", True),
        )
        model.load_state_dict(state, strict=True)
        return model.to(map_location).eval()
