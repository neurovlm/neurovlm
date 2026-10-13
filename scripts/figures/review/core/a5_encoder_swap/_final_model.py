"""Trained models of ``--model-dir`` (default scripts/outputs/runs/final, the paper model) or, with ``--released``, the
released Hugging Face weights at the revisions pinned in scripts/hf_manifest.json (_hf_pins.py). Copied from
../a2_granularity_tradeoff/_review_common.py so this directory imports nothing from scripts/; a file the run lacks falls back
to the released one. SPECTER2 and NeuroQwen3-0.6B are frozen: always released.

Generated maps follow the paper's calibration rule (scripts/README.md, "Map calibration"): each sigmoid map is multiplied
by the model's global factor (fitted on the validation split, <model dir>/map_calibration.json) and clipped at 1 before it
is drawn, encoded or scored. Percentile Dice does not change; anything encoded does.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[5]
DEFAULT_MODEL_DIR = REPO / "scripts" / "outputs" / "runs" / "final"
RUN_FILES = {
    "autoencoder": "autoencoder.pt", "proj_head_text_mse": "proj_head_text_mse.pt", "proj_head_text_infonce": "proj_head_text_infonce.pt",
    "proj_head_image_infonce": "proj_head_image_infonce.pt", "neuro_adapter": "adapter_anat_full.pt",
    "qformer_canonical": "neuro_qformer_canonical/model.safetensors", "qformer_pubmed": "neuro_qformer_pubmed/model.safetensors",
    "canonical_semantic_banks": "canonical_semantic_banks.pt", "pmids_split": "pmids_split.pt", "latent_neuro": "latent_neuro.pt",
}
MAP_CALIBRATION = "global"  # "max" = each map divided by its own maximum (the previous rule)
_STATE: dict = {"path": None, "sources": {}}


def add_model_args(parser) -> None:
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR, help="Run directory with the trained models (default: runs/final).")
    parser.add_argument("--released", action="store_true", help="Use the released Hugging Face weights instead of --model-dir.")


def set_model_dir(path) -> Path | None:
    """Use the run directory ``path`` (None = released weights)."""
    _STATE["path"] = None if path is None else Path(path).resolve()
    _STATE.pop("map_calibration", None)
    if _STATE["path"] is not None and not _STATE["path"].is_dir():
        raise FileNotFoundError(f"--model-dir {path} is not a directory (train it or pass --released)")
    return _STATE["path"]


def from_args(args) -> Path | None:
    return set_model_dir(None if getattr(args, "released", False) else args.model_dir)


def model_dir() -> Path | None:
    return _STATE["path"]


def model_tag() -> str:
    """'released' or the run directory's name (part of cache file names)."""
    return "released" if _STATE["path"] is None else _STATE["path"].name


def run_file(name: str) -> Path | None:
    d = _STATE["path"]
    if d is None:
        return None
    p = d / RUN_FILES[name]
    return p if p.exists() else None


def rel(p) -> str:
    p = Path(p).resolve()
    try:
        return str(p.relative_to(REPO))
    except ValueError:
        return str(p)


def note_source(name: str, source) -> None:
    _STATE["sources"][name] = source if isinstance(source, str) else rel(source)


def model_sources() -> dict:
    """Checkpoint of every loaded component."""
    return dict(_STATE["sources"])


def model_info() -> dict:
    """Model directory, component sources and map calibration, for the outputs' config / metrics files."""
    return {"model_dir": rel(_STATE["path"]) if _STATE["path"] else "released", "model_sources": model_sources(),
            "map_calibration": MAP_CALIBRATION}


def data_dir() -> Path:
    from neurovlm.data import data_dir as _dd

    return Path(_dd)


def released(name: str, device):
    """Released ``neurovlm`` model in eval mode, at its pinned revision."""
    from _hf_pins import load_model

    model = load_model(name)
    if hasattr(model, "to"):
        model = model.to(device)
    if hasattr(model, "eval"):
        model.eval()
    elif hasattr(model, "specter"):
        model.specter.eval()
    return model


def component(name: str, device):
    """Trained component of the run (whole-module .pt) or the released one: autoencoder, proj_head_text_mse (runs/final:
    text-to-brain head v2), proj_head_text_infonce, proj_head_image_infonce, neuro_adapter (anatomical head) or specter."""
    import torch

    p = run_file(name) if name in RUN_FILES else None
    if p is None:
        if _STATE["path"] is not None and name in RUN_FILES:
            print(f"[model-dir] {_STATE['path']} has no {RUN_FILES[name]}; using the released {name}")
        note_source(name, "released:allenai/specter2_aug2023refresh + adhoc_query (frozen)" if name == "specter" else f"released:{name}")
        return released(name, device)
    obj = torch.load(p, map_location="cpu", weights_only=False)
    if isinstance(obj, dict):  # raw state dict -> released architecture
        from _hf_pins import load_model

        m = load_model(name)
        m.load_state_dict(obj.get("state_dict", obj), strict=True)
        obj = m
    note_source(name, p)
    return obj.to(device).eval()


def pubmed_latents():
    """Autoencoder latents [N, 384] (CPU) and int64 pmids: the run's latent_neuro.pt, else the released latents."""
    import torch

    p = run_file("latent_neuro")
    if p is not None:
        payload = torch.load(p, map_location="cpu", weights_only=False)
        note_source("latent_neuro", p)
        return payload["latent"].float().clone(), np.asarray(payload["pmid"]).astype(np.int64)
    from _hf_pins import load_latent

    lat, pm = load_latent("pubmed_images")
    note_source("latent_neuro", "released:HF pubmed_images latents (released autoencoder)")
    return torch.as_tensor(lat).float().cpu().clone(), np.asarray(pm).astype(np.int64)


def split_pmids() -> dict[str, np.ndarray]:
    """Train / val / test PMIDs: the run's pmids_split.pt, else the split flags of the pinned HF publications.parquet."""
    import torch

    p = run_file("pmids_split")
    if p is None:
        from _hf_pins import hf_split_pmids

        return hf_split_pmids()
    split = torch.load(p, map_location="cpu", weights_only=False)
    return {k: np.asarray(split[k]).astype(np.int64) for k in ("train", "val", "test")}


def map_factor(kind: str = "head") -> float:
    """Global calibration factor of the text-to-brain head ("head") or anatomical head ("adapter"): <model dir>/
    map_calibration.json, or scripts/outputs/released_map_calibration.json for the released weights. Both are written by
    scripts/validate_models.py (scripts/models/_common_models.map_calibration)."""
    import hashlib

    if "map_calibration" not in _STATE:
        d = _STATE["path"]
        path = d / "map_calibration.json" if d is not None else REPO / "scripts" / "outputs" / "released_map_calibration.json"
        if not path.exists():
            raise FileNotFoundError(f"{path} is missing: run scripts/validate_models.py on this model first")
        cal = json.loads(path.read_text())
        for src in cal["source"]["models"].values():  # the factors belong to the model files they were fitted on
            if d is not None and not src.startswith("released:"):
                fname, sha = src.split(":")
                if hashlib.sha256((d / fname).read_bytes()).hexdigest() != sha:
                    raise RuntimeError(f"{path} was fitted on another {fname}; rerun scripts/validate_models.py")
        _STATE["map_calibration"] = cal
    return float(_STATE["map_calibration"][kind])


def calibrate_maps(maps, rule: str = MAP_CALIBRATION, *, factor: float | None = None, kind: str = "head", eps: float = 1e-8):
    """Calibrate maps (last axis = voxels): "global" = times ``factor`` (default map_factor(kind)), clipped at 1; "max" =
    divided by their own maximum; "none" = unchanged. Torch or numpy in, same type out. As
    scripts/models/_common_models.calibrate_maps."""
    if rule == "none":
        return maps
    if rule == "global":
        factor = map_factor(kind) if factor is None else factor
        if isinstance(maps, np.ndarray):
            return np.minimum(maps * np.float32(factor), 1).astype(np.float32, copy=False)
        return (maps * factor).clamp(max=1.0)
    if rule != "max":
        raise ValueError(f"unknown map calibration {rule!r}")
    if isinstance(maps, np.ndarray):
        peak = np.maximum(maps.max(axis=-1, keepdims=True), eps)
        return (maps / peak).astype(maps.dtype if maps.dtype.kind == "f" else np.float32, copy=False)
    return maps / maps.amax(dim=-1, keepdim=True).clamp_min(eps)


def generate_maps(text_emb, head, decoder, device, *, batch_size: int = 256):
    """Calibrated text-to-brain maps [N, 28542] (CPU float32): SPECTER2 (L2-normalised) -> head -> decoder -> sigmoid ->
    calibrate_maps (the final head's factor)."""
    import torch
    import torch.nn.functional as F

    out = []
    with torch.no_grad():
        for s in range(0, len(text_emb), batch_size):
            e = F.normalize(torch.as_tensor(text_emb[s:s + batch_size]).float().to(device), dim=1)
            out.append(calibrate_maps(torch.sigmoid(decoder(head(e)).float())).cpu())
    return torch.cat(out)


def _define_mlp_projector():
    """``ResidualMLPProjector`` of scripts/models/_common_models.py, verbatim (the final run's projectors)."""
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


def load_projector(variant: str, device):
    """Stage-1 (``pubmed``) or stage-2 (``canonical``) brain-to-text projector of the run, wrapped in ``NeuroQFormer``, else the
    released Q-Former. For the canonical variant the run's canonical_semantic_banks.pt replaces the packaged banks."""
    import torch
    import torch.nn.functional as F

    key = f"qformer_{variant}"
    weights = run_file(key)
    if weights is None:
        from _hf_pins import released_qformer

        note_source(key, f"released:neurovlm/{'NeuroQformer' if variant == 'canonical' else 'NeuroQformer-PubMed'}")
        return released_qformer(variant, device)
    from safetensors.torch import load_file
    from neurovlm.models.qformer import CanonicalProjection, NeuroQFormer

    cfg = json.loads((weights.parent / "config.json").read_text())
    state = dict(load_file(str(weights)))
    note_bank = ""
    if variant == "canonical" and run_file("canonical_semantic_banks") is not None:
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
    note_source(key, f"{rel(weights.parent)} (projector: {kind}){note_bank}")
    return model.to(device).eval()


def make_nvlm(device, *, ae, text_head, image_head, mse_head, specter=None, projector=None, variant: str = "canonical"):
    """``NeuroVLM`` client that uses the given autoencoder, heads, SPECTER2 and brain-to-text projector (with NeuroQwen3 at
    its pinned revision)."""
    from neurovlm import NeuroVLM
    from _hf_pins import hf_cache_pinned, pin_neuroqwen

    hf_cache_pinned()  # anything the client still lazy-loads comes from the pinned HF cache
    nvlm = NeuroVLM(device=str(device))
    nvlm._autoencoder = ae
    nvlm._proj_head_text_infonce, nvlm._proj_head_image_infonce, nvlm._proj_head_text_mse = text_head, image_head, mse_head
    if specter is not None:
        nvlm._specter = specter
    if projector is not None:
        nvlm._neuro_qformer, nvlm._neuro_qformer_variant = projector, variant
        pin_neuroqwen(nvlm)
        note_source("neuroqwen", "released:neurovlm/NeuroQwen3-0.6B (frozen LM)")
    return nvlm
