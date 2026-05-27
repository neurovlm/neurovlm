"""Model wrappers reused by the smoke-test trainer."""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch import nn

REPO_ROOT = Path(__file__).resolve().parents[3]
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


class TextToBrainProjectionHead(nn.Module):
    """SPECTER/SPECTER2 embedding -> AE decoder latent projection."""

    def __init__(
        self,
        in_dim: int = 768,
        hidden_dim: int = 512,
        out_dim: int = 384,
        *,
        depth: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        if depth not in {2, 3}:
            raise ValueError("depth must be 2 or 3")
        layers: list[nn.Module] = [nn.Linear(in_dim, hidden_dim), nn.ReLU()]
        if dropout:
            layers.append(nn.Dropout(dropout))
        if depth == 3:
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.ReLU()])
            if dropout:
                layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(hidden_dim, out_dim))
        # The 2-layer, dropout=0 case intentionally matches neurovlm.models.ProjHead
        # key names enough for shape-compatible pretrained text_infonce initialization.
        self.aligner = nn.Sequential(*layers)

    def forward(self, x):
        return self.aligner(x)


def build_text_projection(init: str = "random", *, device: str | torch.device = "cpu") -> nn.Module:
    """Build a text projection variant for CNN latent experiments.

    ``pretrained_text_infonce`` uses the original NeuroVLM MLP-latent text
    projection as an initialization variant. It is intentionally not assumed to
    be aligned with the CNN latent manifold.
    """

    if init in {"random", "scratch"}:
        from neurovlm.models import ProjHead

        return ProjHead(latent_in_dim=768, hidden_dim=512, latent_out_dim=384).to(device)
    if init in {"pretrained_text_infonce", "text_infonce"}:
        from neurovlm.models import ProjHead

        return ProjHead.from_pretrained("text_infonce").to(device)
    raise ValueError("init must be 'random' or 'pretrained_text_infonce'")


def build_text_to_brain_projection(
    init: str = "random",
    *,
    device: str | torch.device = "cpu",
    in_dim: int = 768,
    hidden_dim: int = 512,
    out_dim: int = 384,
    depth: int = 2,
    dropout: float = 0.1,
) -> nn.Module:
    model = TextToBrainProjectionHead(
        in_dim=in_dim,
        hidden_dim=hidden_dim,
        out_dim=out_dim,
        depth=depth,
        dropout=dropout,
    ).to(device)
    if init in {"random", "scratch"}:
        return model
    if init in {"pretrained_text_infonce", "text_infonce"}:
        from neurovlm.models import ProjHead

        pretrained = ProjHead.from_pretrained("text_infonce").state_dict()
        current = model.state_dict()
        compatible = {
            key: value
            for key, value in pretrained.items()
            if key in current and tuple(value.shape) == tuple(current[key].shape)
        }
        current.update(compatible)
        model.load_state_dict(current)
        model.pretrained_init_summary = {  # type: ignore[attr-defined]
            "loaded_tensors": len(compatible),
            "checkpoint_tensors": len(pretrained),
        }
        return model
    raise ValueError("init must be 'random' or 'pretrained_text_infonce'")


def build_brain_encoder(
    out_dim: int = 384,
    *,
    encoder_arch: str = "plain",
    base_channels: int = 48,
    num_blocks: int = 4,
    blocks_per_stage: int = 2,
    dropout: float = 0.1,
    use_dilation: bool = False,
    multi_scale: bool = False,
    global_context: str = "none",
):
    from neurovlm.gnn.ale_cnn import ALE3DCNNEncoder, ALEResNet3DEncoder

    if encoder_arch == "plain":
        return ALE3DCNNEncoder(
            base_channels=base_channels,
            num_blocks=num_blocks,
            out_dim=out_dim,
            dropout=dropout,
        )
    if encoder_arch == "resnet":
        return ALEResNet3DEncoder(
            base_channels=base_channels,
            num_stages=num_blocks,
            blocks_per_stage=blocks_per_stage,
            out_dim=out_dim,
            dropout=dropout,
            use_dilation=use_dilation,
            multi_scale=multi_scale,
            global_context=global_context,
        )
    raise ValueError("encoder_arch must be 'plain' or 'resnet'")


def build_cnn_autoencoder(
    output_shape: tuple[int, int, int],
    *,
    latent_dim: int = 384,
    base_channels: int = 48,
    num_blocks: int = 4,
    dropout: float = 0.1,
    norm: str = "group",
    pooling: str = "max",
    encoder_arch: str = "plain",
    blocks_per_stage: int = 2,
    use_dilation: bool = False,
    multi_scale: bool = False,
    global_context: str = "none",
):
    from neurovlm.gnn.ale_cnn import ALE3DCNNAutoEncoder

    return ALE3DCNNAutoEncoder(
        output_shape=output_shape,
        base_channels=base_channels,
        num_blocks=num_blocks,
        latent_dim=latent_dim,
        dropout=dropout,
        norm=norm,
        pooling=pooling,
        encoder_arch=encoder_arch,
        blocks_per_stage=blocks_per_stage,
        use_dilation=use_dilation,
        multi_scale=multi_scale,
        global_context=global_context,
    )


def load_autoencoder_checkpoint(model: nn.Module, checkpoint_path: str | Path, *, strict: bool = True) -> dict:
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = payload.get("model") or payload.get("autoencoder") or payload.get("state_dict")
    if state is None:
        raise KeyError("Checkpoint must contain 'model', 'autoencoder', or 'state_dict'")
    model.load_state_dict(state, strict=strict)
    return payload


def load_text_projection_checkpoint(model: nn.Module, checkpoint_path: str | Path, *, strict: bool = True) -> dict:
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = payload.get("text_projector") or payload.get("text_projection") or payload.get("state_dict")
    if state is None:
        raise KeyError("Checkpoint must contain 'text_projector', 'text_projection', or 'state_dict'")
    model.load_state_dict(state, strict=strict)
    return payload


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
