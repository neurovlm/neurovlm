"""Loads the final run's models (``--model-dir``, default scripts/outputs/runs/final) for run_correctness.py.

Copied from ../a2_granularity_tradeoff/_review_common.py so this directory imports nothing from scripts/. Files the
run lacks fall back to the released models; NeuroQwen3-0.6B is always the frozen release.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[5]
DEFAULT_MODEL_DIR = ROOT / "scripts" / "outputs" / "runs" / "final"
RUN_FILES = {"autoencoder": "autoencoder.pt", "qformer_canonical": "neuro_qformer_canonical/model.safetensors",
             "canonical_semantic_banks": "canonical_semantic_banks.pt", "pmids_split": "pmids_split.pt"}
_MODEL_DIR: dict = {"path": None, "sources": {}}


def set_model_dir(path) -> None:
    _MODEL_DIR["path"] = Path(path).resolve()
    if not _MODEL_DIR["path"].is_dir():
        raise FileNotFoundError(f"--model-dir {path} is not a directory")


def run_file(name: str) -> Path | None:
    p = _MODEL_DIR["path"] / RUN_FILES[name]
    return p if p.exists() else None


def _rel(p) -> str:
    p = Path(p).resolve()
    try:
        return str(p.relative_to(ROOT))
    except ValueError:
        return str(p)


def note_source(name: str, source) -> None:
    _MODEL_DIR["sources"][name] = source if isinstance(source, str) else _rel(source)


def model_sources() -> dict:
    """Which checkpoint every loaded component came from."""
    return dict(_MODEL_DIR["sources"])


def component(name: str, device):
    """A whole-module ``.pt`` from the run directory, else the released model."""
    import torch

    p = run_file(name)
    if p is None:
        from _hf_pins import load_model

        print(f"[model-dir] {_MODEL_DIR['path']} has no {RUN_FILES[name]}; released weights are used for {name}")
        note_source(name, f"released:{name}")
        return load_model(name).to(device).eval()
    note_source(name, p)
    return torch.load(p, map_location="cpu", weights_only=False).to(device).eval()


def _define_mlp_projector():
    """``ResidualMLPProjector`` of scripts/models/_common_models.py, verbatim."""
    import torch
    import torch.nn.functional as F
    from torch import nn

    class ResidualBlock(nn.Module):
        def __init__(self, width: int, expansion: int = 2, dropout: float = 0.05):
            super().__init__()
            self.norm = nn.LayerNorm(width)
            self.fc1 = nn.Linear(width, width * expansion)
            self.act = nn.GELU()
            self.drop = nn.Dropout(dropout)
            self.fc2 = nn.Linear(width * expansion, width)

        def forward(self, x):
            return x + self.fc2(self.drop(self.act(self.fc1(self.norm(x)))))

    class ResidualMLPProjector(nn.Module):
        def __init__(self, image_dim=384, semantic_dim=384, lm_dim=1024, num_queries=32, *, width=704, depth=2, expansion=2,
                     dropout=0.05, head="tokens", align_hidden=512, **_ignored):
            super().__init__()
            self.image_dim, self.semantic_dim, self.lm_dim, self.num_queries = int(image_dim), int(semantic_dim), int(lm_dim), int(num_queries)
            self.width, self.depth, self.expansion, self.head_kind = int(width), int(depth), int(expansion), head
            self.dropout, self.align_hidden = float(dropout), int(align_hidden)
            self.norm_raw, self.norm_sem = nn.LayerNorm(image_dim), nn.LayerNorm(semantic_dim)
            self.in_proj = nn.Linear(image_dim + semantic_dim, width)
            self.blocks = nn.ModuleList([ResidualBlock(width, expansion, dropout) for _ in range(depth)])
            self.out_norm = nn.LayerNorm(width)
            self.head = nn.Linear(width, num_queries * lm_dim if head == "tokens" else lm_dim, bias=False)
            self.query_pos = nn.Parameter(torch.zeros(1, num_queries, lm_dim))
            self.token_norm = nn.LayerNorm(lm_dim)
            self.align_head = nn.Sequential(nn.LayerNorm(lm_dim), nn.Linear(lm_dim, align_hidden), nn.GELU(), nn.Linear(align_hidden, semantic_dim))

        def forward(self, raw_images, semantic_images):
            dtype = self.in_proj.weight.dtype
            h = self.in_proj(torch.cat([self.norm_raw(raw_images.to(dtype)), self.norm_sem(semantic_images.to(dtype))], dim=-1))
            for block in self.blocks:
                h = block(h)
            t = self.head(self.out_norm(h))
            if self.head_kind == "tokens":
                t = t.view(t.size(0), self.num_queries, self.lm_dim)
            else:
                t = t.unsqueeze(1).expand(-1, self.num_queries, -1)
            return self.token_norm(t + self.query_pos)

        def semantic_from_visual(self, visual_tokens):
            x = visual_tokens.to(dtype=self.in_proj.weight.dtype).mean(dim=1)
            return F.normalize(self.align_head(x).float(), dim=1)

    return ResidualMLPProjector


def load_projector(device):
    """The run's stage-2 (canonical) brain-to-text projector as a library ``NeuroQFormer``, else the released one.

    ``config.json["projector"]`` picks the module (``mlp`` or ``qformer``); the run's canonical_semantic_banks.pt
    replaces the packaged banks."""
    import torch
    import torch.nn.functional as F

    key = "qformer_canonical"
    weights = run_file(key)
    if weights is None:
        from _hf_pins import released_qformer

        note_source(key, "released:neurovlm/NeuroQformer")
        return released_qformer("canonical", device)
    from safetensors.torch import load_file
    from neurovlm.models.qformer import CanonicalProjection, NeuroQFormer

    cfg = json.loads((weights.parent / "config.json").read_text())
    state = dict(load_file(str(weights)))
    note_bank = ""
    if run_file("canonical_semantic_banks") is not None:
        banks = torch.load(run_file("canonical_semantic_banks"), map_location="cpu", weights_only=False)
        same = True
        for b in ("all", "network", "region", "function"):
            k = f"canonical_projection.{b}_bank"
            new = F.normalize(torch.as_tensor(banks[b]).float(), dim=1)
            old = state.get(k)
            same &= old is not None and old.shape == new.shape and float((old.float() - new).abs().max()) == 0.0
            state[k] = new
        note_bank = " (canonical_semantic_banks.pt " + ("identical to the packaged banks)" if same else "replaced the packaged banks)")
    kind = cfg.get("projector", "qformer")
    block = dict(cfg.get("qformer", {}))
    if kind == "qformer":
        model = NeuroQFormer.from_state_dict_payload({"state_dict": state, "config": block, "canonical_metadata": cfg.get("canonical_metadata", {})},
                                                     map_location=device)
    elif kind == "mlp":
        from neurovlm.models.base import ProjHead

        pcfg = {k: v for k, v in (cfg.get("projector_config") or {}).items() if k not in ("class", "params")}
        projector = _define_mlp_projector()(**pcfg)
        first, last = state["proj_head_image.aligner.0.weight"], state["proj_head_image.aligner.2.weight"]
        head = ProjHead(int(first.shape[1]), int(first.shape[0]), int(last.shape[0]))
        sem_dim = int(pcfg.get("semantic_dim", 384))
        banks = {n: state.get(f"canonical_projection.{n}_bank", torch.empty(0, sem_dim)) for n in CanonicalProjection.bank_names}
        model = NeuroQFormer(qformer=projector, proj_head_image=head, canonical_banks=banks, semantic_dim=sem_dim,
                             projection_temp=block.get("projection_temp", 0.05), canonical_basis=block.get("canonical_basis", "all"),
                             use_canonical_projection=block.get("use_canonical_projection", True))
        model.load_state_dict(state, strict=True)
    else:
        raise ValueError(f"unknown projector {kind!r} in {weights.parent / 'config.json'}")
    note_source(key, f"{_rel(weights.parent)} (projector: {kind}){note_bank}")
    return model.to(device).eval()


def make_nvlm(device, ae, projector):
    """``NeuroVLM`` client that captions with the given autoencoder and canonical projector (NeuroQwen3 at its pinned
    revision)."""
    from neurovlm import NeuroVLM
    from _hf_pins import hf_cache_pinned, pin_neuroqwen

    hf_cache_pinned()  # anything the client still lazy-loads comes from the pinned HF cache
    nvlm = NeuroVLM(device=str(device))
    nvlm._autoencoder = ae
    nvlm._neuro_qformer, nvlm._neuro_qformer_variant = projector, "canonical"
    pin_neuroqwen(nvlm)
    note_source("neuroqwen", "released:neurovlm/NeuroQwen3-0.6B (frozen LM)")
    return nvlm


def nvlm_generate(nvlm, X, **kw) -> list[str]:
    out = nvlm.generate_text(X, qformer_variant="canonical", **kw)
    return [out] if isinstance(out, str) else list(out)
