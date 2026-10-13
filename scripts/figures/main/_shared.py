"""Helpers shared by the paper-figure scripts: CLI flags, panel caches, model loading (run dir or release),
brain maps, text layout, stats and the colour palette."""
from __future__ import annotations

import gc
import json
import os
import re
import sys
from functools import cache
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # scripts/
from common import (  # noqa: E402,F401  (some names are only re-exported for the fig scripts)
    DEFAULT_OUT_ROOT,
    FIG_FULL_W,
    FIG_HALF_W,
    FIG_LETTER_MAX_H,
    FIG_LETTER_W,
    FIG_MAX_H,
    PAPER_FIG_DIR,
    REPO_ROOT,
    apply_neurovault_exclusions,
    apply_paper_style,
    apply_pubmed_exclusions,
    base_parser,
    data_dir,
    finalize_args,
    get_logger,
    hf_manifest,
    hf_pinned_dir,
    hf_split_pmids,
    read_json,
    save_fig,
    select_device,
    setup_torch,
    timed,
    write_json,
)

if str(Path(__file__).resolve().parents[2] / "models") not in sys.path:
    sys.path.append(str(Path(__file__).resolve().parents[2] / "models"))  # appended, so it never shadows a caller's module
from _common_models import (  # noqa: E402
    MAP_CALIBRATION_DEFAULT, MAP_CALIBRATION_RULES, calibrate_maps, hf_pinned, install_title_constraint,
    latin_script_suppress_tokens, map_calibration, neuro_projector_from_state, released_model, released_qformer,
    strip_steering_tags, title_constraint_note, title_vocabulary,
)

FIG_OUT = Path(__file__).resolve().parent  # the final figures live next to their scripts
CACHE_DIR = FIG_OUT / "cache"
DEFAULT_RUN = DEFAULT_OUT_ROOT / "runs" / "final"  # the paper model
# panel caches that hold generated captions: their names carry the title mode ("_titles-vocab" by default; none with
# --free-titles, the decoding of the earlier caches)
CAPTION_CACHES = ("fig2_panels", "fig3_pool", "fig4_b2t", "figS4_panels")
_TITLES = {"free": False}  # --free-titles of the running figure script (finalize); make_nvlm's default
# output stem -> file name in the manuscript's figure folder (--paper)
PAPER_NAMES = {"fig2_network_examples": "networks_examples", "fig3_newgen": "newgen",
               "fig4_generative_metrics": "text_to_brain_metrics_pubmed", "fig5_perf_quant": "perf_quant",
               "fig6_autoencoder_qual": "autoencoder_qual", "figS2_difumo_v_ale": "difumo_v_ale",
               "figS3_recall_comparison": "recall_comparison", "figS4_clip_ica": "clip_ica_hcp_ukb"}

# palette: slots 1-5 of the dataviz reference palette (CVD-safe in this order); brain maps one hue; nulls grey
BLUE, ORANGE, AQUA, YELLOW, VIOLET = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#4a3aa7"
SERIES = [BLUE, ORANGE, AQUA, YELLOW, VIOLET]
MATCH_COLOR = BLUE
NULL_COLOR = "#a9a7a1"
CHANCE_COLOR = "black"
STATUS_GOOD, STATUS_WARN, STATUS_BAD = "#0ca30c", "#eda100", "#d03b3b"
BRAIN_CMAP = "Reds"


# CLI
def fig_parser(description: str):
    """base_parser plus the flags every figure script shares."""
    p = base_parser(description, stage="figures")
    p.set_defaults(out=None)  # set in finalize()
    p.add_argument("--paper", action="store_true", help="Also copy the PNG into paper/figures/.")
    p.add_argument("--released", action="store_true", help="Use the released weights (output: released/).")
    p.add_argument("--model-dir", type=Path, default=DEFAULT_RUN,
                   help="Run directory of scripts/train_pipeline.py; missing checkpoints fall back to the released ones.")
    p.add_argument("--cache-dir", type=Path, default=CACHE_DIR, help="Panel-data cache.")
    p.add_argument("--n-jobs", type=int, default=min(8, os.cpu_count() or 1), help="CPU workers.")
    p.add_argument("--font-size", type=float, default=7.0, help="Smallest text size in pt.")
    p.add_argument("--no-exclusions", action="store_true", help="Keep the leakage-audit items (comparison only).")
    p.add_argument("--map-calibration", choices=MAP_CALIBRATION_RULES, default=MAP_CALIBRATION_DEFAULT,
                   help="Generated maps: global = the model's validation-fitted factor, clipped at 1; max = divide by own "
                        "maximum; none = raw sigmoid.")
    p.add_argument("--rescale-max", dest="map_calibration", action="store_const", const="max",
                   help="Divide each generated map by its own maximum (= --map-calibration max, the previous rule).")
    p.add_argument("--free-titles", action="store_true",
                   help="Decode free [NETWORK] / [REGION] caption titles (default: only the canonical names of the stage-14 banks).")
    return p


def default_out(args) -> Path:
    """figures/main/ for runs/final, released/ or <run>/ otherwise; previews get their own sub-folder."""
    out = FIG_OUT
    if args.released:
        out = out / "released"
    elif Path(args.model_dir).resolve() != DEFAULT_RUN.resolve():
        out = out / Path(args.model_dir).name
    if args.max_samples:
        out = out / f"preview_n{args.max_samples}"
    return out


def finalize(args):
    if args.released:
        args.model_dir = None
    elif not Path(args.model_dir).is_dir():
        raise FileNotFoundError(f"--model-dir {args.model_dir} is not a directory (train it with scripts/train_pipeline.py "
                                "or pass --released)")
    out_given = args.out is not None
    args.out = args.out or FIG_OUT
    args = finalize_args(args)  # resolves --smoke into --max-samples
    if not out_given:
        args.out = default_out(args)
        args.out.mkdir(parents=True, exist_ok=True)
    args.cache_dir = Path(args.cache_dir)
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    _TITLES["free"] = bool(getattr(args, "free_titles", False))
    if args.model_dir is not None:
        args.model_dir = Path(args.model_dir).resolve()
        # decodes the digest in the cache names
        write_json(args.cache_dir / f"model_sources{cache_tag(args)}.json",
                   {"model_dir": rel_path(args.model_dir), "digest": source_hash(args.model_dir),
                    "components": resolve_sources(args.model_dir)})
    apply_paper_style(args.font_size + 1.0)  # ticks/legends = font_size, labels +1, titles +2
    setup_torch(args.seed)
    return args


def banner(log, args, stem: str) -> None:
    log.info("figure=%s out=%s cache=%s max_samples=%s model_dir=%s force=%s",
             stem, args.out, args.cache_dir, args.max_samples, args.model_dir, args.force)


# caches
def cache_tag(args, *, excl: bool = False) -> str:
    """Cache-name suffix: '_excl' (exclusions on; only for panels that depend on the evaluated items),
    '_n<max-samples>' and '_m-<run>-<digest>' (see source_hash)."""
    tag = ""
    if excl and excl_on(args):
        tag += "_excl"
    if args.max_samples:
        tag += f"_n{args.max_samples}"
    if args.model_dir:
        tag += f"_m-{Path(args.model_dir).name}-{source_hash(Path(args.model_dir))}"
    return tag


def cache_file(args, stem: str, ext: str = "pt", *, tagged: bool = True, excl: bool = False) -> Path:
    if stem.startswith(CAPTION_CACHES) and not getattr(args, "free_titles", False):
        stem += "_titles-vocab"
    return Path(args.cache_dir) / f"{stem}{cache_tag(args, excl=excl) if tagged else ''}.{ext}"


def map_cal_tag(args) -> str:
    """'_cal-<rule>' for panels that hold generated maps."""
    return f"_cal-{args.map_calibration}"


def map_factor(args, kind: str = "head", device="cpu") -> float | None:
    """Global calibration factor of the --model-dir (or released) text-to-brain head ("head") or anatomical head ("adapter");
    None unless --map-calibration global."""
    if args.map_calibration != "global":
        return None
    return map_calibration(args.model_dir, device=device)[kind]


def generate_brain_maps(text_emb, models: dict, *, kind: str = "head", calibration: str = MAP_CALIBRATION_DEFAULT,
                        factor: float | None = None, batch_size: int = 512):
    """Calibrated generated maps (CPU) for SPECTER2 embeddings [N, 768]. kind="head": sigmoid(decoder(proj_head_text_mse(e)));
    kind="adapter": sigmoid(neuro_adapter(e)), the anatomical head. ``factor``: map_factor(args, kind)."""
    import torch
    import torch.nn.functional as F

    out = []
    with torch.no_grad():
        for s in range(0, len(text_emb), batch_size):
            e = F.normalize(torch.as_tensor(text_emb[s:s + batch_size]).float(), dim=1)
            if kind == "adapter":
                adapter = models["neuro_adapter"]
                p = torch.sigmoid(adapter(e.to(next(adapter.parameters()).device)))
            else:
                head = models["proj_head_text_mse"]
                p = torch.sigmoid(models["autoencoder"].decoder(head(e.to(next(head.parameters()).device))))
            out.append(calibrate_maps(p.float(), calibration, factor=factor).cpu())
    return torch.cat(out)


# evaluation exclusions: counts per key, saved with the panel caches and written to <stem>_text.json
_EXCLUSION_COUNTS: dict = {}


def excl_on(args) -> bool:
    return not args.no_exclusions


def _note(args, key: str, keep: np.ndarray) -> None:
    _EXCLUSION_COUNTS.setdefault("enabled", excl_on(args))
    _EXCLUSION_COUNTS[key] = {"before": int(len(keep)), "dropped": int((~keep).sum()), "kept": int(keep.sum())}


def pubmed_keep(args, pmids, key: str, log=None) -> np.ndarray:
    """Keep-mask over PubMed pmids (leakage-audit PMIDs dropped unless --no-exclusions)."""
    keep = apply_pubmed_exclusions(pmids, enabled=excl_on(args), logger=log, label=key)
    _note(args, key, keep)
    return keep


def neurovault_keep(args, dois=None, key: str = "neurovault", log=None, **ids) -> np.ndarray:
    """Keep-mask over NeuroVault rows (papers that are PubMed training papers dropped)."""
    keep = apply_neurovault_exclusions(dois, enabled=excl_on(args), logger=log, label=key, **ids)
    _note(args, key, keep)
    return keep


def load_pt(path: Path):
    import torch

    obj = torch.load(path, map_location="cpu", weights_only=False)
    # cache hit: restore the recorded exclusion counts and checkpoint sources
    if isinstance(obj, dict) and isinstance(obj.get("eval_exclusions"), dict):
        for k, v in obj["eval_exclusions"].items():
            _EXCLUSION_COUNTS.setdefault(k, v)
    if isinstance(obj, dict) and isinstance(obj.get("model_sources"), dict):
        for k, v in obj["model_sources"].items():
            _MODEL_SOURCES.setdefault(k, v)
    return obj


def save_pt(path: Path, obj) -> Path:
    import torch

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(obj, dict) and _EXCLUSION_COUNTS:
        obj = {**obj, "eval_exclusions": dict(_EXCLUSION_COUNTS)}
    if isinstance(obj, dict) and _MODEL_SOURCES:
        obj = {**obj, "model_sources": dict(_MODEL_SOURCES)}
    torch.save(obj, path)
    return path


def free_gpu() -> None:
    import torch

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# models: every trained component is looked up in --model-dir first, else the released checkpoint is used (with a
# warning). SPECTER2 and NeuroQwen3 are frozen and always come from the release.
RUN_FILES: dict[str, tuple[str, ...]] = {
    "autoencoder": ("autoencoder.pt", "autoencoder.safetensors"),
    "proj_head_text_mse": ("proj_head_text_mse.pt", "proj_head_text_mse.safetensors"),
    "proj_head_text_infonce": ("proj_head_text_infonce.pt", "proj_head_text_infonce.safetensors"),
    "proj_head_image_infonce": ("proj_head_image_infonce.pt", "proj_head_image_infonce.safetensors"),
    "neuro_adapter": ("adapter_anat_full.pt", "adapter_anat_full.safetensors"),
    "qformer_canonical": ("neuro_qformer_canonical/model.safetensors",),
    "canonical_semantic_banks": ("canonical_semantic_banks.pt",),
    "qformer_pubmed": ("neuro_qformer_pubmed/model.safetensors",),
    "pmids_split": ("pmids_split.pt",),
    "latent_neuro": ("latent_neuro.pt",),
}
FOLD_PARTS = ("autoencoder", "proj_head_text_infonce", "proj_head_image_infonce")  # Fig. 5a CV checkpoints
OVERRIDABLE = ("autoencoder", "proj_head_text_mse", "proj_head_text_infonce", "proj_head_image_infonce", "neuro_adapter")
RELEASED = {
    "autoencoder": "released:neurovlm/NeuroAutoEncoder",
    "proj_head_text_mse": "released:neurovlm/ProjectionHeads/proj_head_text_mse",
    "proj_head_text_infonce": "released:neurovlm/ProjectionHeads/proj_head_text_infonce",
    "proj_head_image_infonce": "released:neurovlm/ProjectionHeads/proj_head_image_infonce",
    "neuro_adapter": "released:neurovlm/NeuroAdapter",
    "qformer_canonical": "released:neurovlm/NeuroQformer",
    "canonical_semantic_banks": "released:neurovlm/NeuroQformer (banks packaged in model.safetensors)",
    "qformer_pubmed": "released:neurovlm/NeuroQformer-PubMed",
    "specter": "released:allenai/specter2_aug2023refresh + adhoc_query adapter (frozen; not retrained by the pipeline)",
    "neuroqwen": "released:neurovlm/NeuroQwen3-0.6B (frozen LM; not retrained by the pipeline)",
    "latent_neuro": "released:HF pubmed_images latents (released autoencoder)",
    "pmids_split": "released:HF neuro_image_papers/publications.parquet split flags (pinned)",
}
_MODEL_SOURCES: dict = {}  # component -> checkpoint it came from, written to <stem>_text.json


def rel_path(path: Path) -> str:
    """Repo-relative path (else ~-abbreviated)."""
    path = Path(path).resolve()
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        home = str(Path.home())
        return "~" + str(path)[len(home):] if str(path).startswith(home) else str(path)


def run_file(model_dir: Path | None, component: str) -> Path | None:
    """First existing RUN_FILES[component] file in model_dir (None without --model-dir)."""
    if model_dir is None:
        return None
    for name in RUN_FILES[component]:
        cand = Path(model_dir) / name
        if cand.exists():
            return cand
    return None


def fold_files(model_dir: Path | None) -> dict[int, dict[str, Path]] | None:
    """{fold: {part: <dir>/<part>_kfold-<fold>.pt}} when all 15 files exist in model_dir."""
    if model_dir is None:
        return None
    files = {i: {k: Path(model_dir) / f"{k}_kfold-{i}.pt" for k in FOLD_PARTS} for i in range(5)}
    return files if all(p.exists() for d in files.values() for p in d.values()) else None


def resolve_sources(model_dir: Path | None) -> dict[str, str]:
    """Where every component comes from for this --model-dir (run file or released label)."""
    out = {}
    for comp in RUN_FILES:
        p = run_file(model_dir, comp)
        out[comp] = rel_path(p) if p is not None else RELEASED.get(comp, f"data_dir/{RUN_FILES[comp][0]}")
    folds = "{" + ",".join(FOLD_PARTS) + "}_kfold-{0..4}.pt"
    out["cv_folds"] = f"{rel_path(model_dir)}/{folds}" if fold_files(model_dir) else f"data_dir/{folds}"
    out["specter"], out["neuroqwen"] = RELEASED["specter"], RELEASED["neuroqwen"]
    return out


def source_hash(model_dir: Path | None) -> str:
    """8-hex digest over (component, file, size, mtime) of every run-dir file used; part of the cache names."""
    import hashlib

    if model_dir is None:
        return ""
    items = []
    for comp in RUN_FILES:
        p = run_file(model_dir, comp)
        if p is not None:
            st = p.stat()
            items.append((comp, p.name, st.st_size, st.st_mtime_ns))
        else:
            items.append((comp, "released"))
    folds = fold_files(model_dir)
    for i, d in sorted((folds or {}).items()):
        for k, p in d.items():
            st = p.stat()
            items.append((f"fold{i}:{k}", p.name, st.st_size, st.st_mtime_ns))
    if not folds:
        items.append(("cv_folds", "data_dir"))
    return hashlib.sha1(json.dumps(items).encode()).hexdigest()[:8]


def note_source(component: str, source) -> None:
    _MODEL_SOURCES[component] = source if isinstance(source, str) else rel_path(source)


# neurovlm loaders without a revision argument (load_dataset, load_latent, load_masker, load_model("specter") and the
# NeuroVLM client's lazy loaders) read the HF cache first; scripts/data/fetch_hf.py points the cache refs at the pinned
# revisions, checked here before any of them runs
SPECTER_REPOS = ("allenai/specter2_aug2023refresh_base", "allenai/specter2_aug2023refresh_adhoc_query")
LIBRARY_REPOS = ("neurovlm/neuro_image_papers", "neurovlm/embedded_text", "neurovlm/cognitive_atlas", "neurovlm/NeuroAutoEncoder",
                 "neurovlm/ProjectionHeads", "neurovlm/NeuroAdapter", "neurovlm/NeuroQformer", "neurovlm/NeuroQformer-PubMed",
                 "neurovlm/NeuroQwen3-0.6B", *SPECTER_REPOS)


@cache
def hf_cache_pinned(*repos: str, group: str = "figures") -> None:
    """Raise unless the HF cache serves every manifest file of ``repos`` (default LIBRARY_REPOS) in ``group`` at its
    pinned revision: refs/<ref> set to it (scripts/data/fetch_hf.py) and the file present."""
    from huggingface_hub import try_to_load_from_cache
    from huggingface_hub.constants import HF_HUB_CACHE

    bad = []
    for repo in repos or LIBRARY_REPOS:
        rows = [e for e in hf_manifest()["files"] if e["repo"] == repo and e.get("hf_cache") and group in e["groups"]]
        if not rows:
            bad.append(f"{repo}: no {group} entry in scripts/hf_manifest.json")
        for e in rows:
            ref = Path(HF_HUB_CACHE) / f"{e['repo_type']}s--{repo.replace('/', '--')}" / "refs" / e["ref"]
            cur = ref.read_text().strip() if ref.exists() else None
            if cur != e["revision"]:
                bad.append(f"{repo}: HF cache refs/{e['ref']} is {cur}, pinned {e['revision'][:8]}")
            elif not isinstance(try_to_load_from_cache(repo, e["path"], revision=e["revision"], repo_type=e["repo_type"]), str):
                bad.append(f"{repo}/{e['path']}@{e['revision'][:8]} is not in the HF cache")
    if bad:
        raise RuntimeError(f"the HF cache is not at the pinned revisions (run scripts/data/fetch_hf.py --only {group}):\n  "
                           + "\n  ".join(sorted(set(bad))))


def load_dataset(name: str):
    """neurovlm.data.load_dataset after hf_cache_pinned()."""
    from neurovlm.data import load_dataset as _load

    hf_cache_pinned()
    return _load(name)


def load_latent(name: str):
    """neurovlm.data.load_latent after hf_cache_pinned()."""
    from neurovlm.data import load_latent as _load

    hf_cache_pinned()
    return _load(name)


def load_masker():
    """neurovlm.data.load_masker (NeuroAutoEncoder mask.npz) after hf_cache_pinned()."""
    from neurovlm.data import load_masker as _load

    hf_cache_pinned()
    return _load()


def load_component(name: str, model_dir: Path | None = None, log=None):
    """One trained component from the run dir (whole-module .pt, state-dict .pt or .safetensors), else the released one
    (at the revision pinned in scripts/hf_manifest.json)."""
    import torch

    path = run_file(model_dir, name) if name in OVERRIDABLE else None
    if path is None:
        if model_dir is not None and name in OVERRIDABLE and log:
            log.warning("--model-dir %s has no %s; released weights are used", model_dir, "/".join(RUN_FILES[name]))
        if name == "specter":  # neurovlm.models.load_model takes no revision; it reads the pinned cache
            hf_cache_pinned(*SPECTER_REPOS)
        module = released_model(name)
        note_source(name, RELEASED.get(name, f"released:{name}"))
        return module
    if path.suffix == ".safetensors":
        from safetensors.torch import load_file

        module = released_model(name)
        module.load_state_dict(load_file(str(path)))
    else:
        module = torch.load(path, map_location="cpu", weights_only=False)
        if not isinstance(module, torch.nn.Module):  # state dict -> released architecture
            state = module
            module = released_model(name)
            module.load_state_dict(state.get("state_dict", state.get("model_state_dict", state)))
    note_source(name, path)
    if log:
        log.info("model: %s <- %s", name, path)
    return module


def load_models(names: Sequence[str], device, model_dir: Path | None = None, log=None) -> dict:
    """load_component for each name, on device, in eval mode."""
    out = {}
    for name in names:
        m = load_component(name, model_dir, log)
        if hasattr(m, "to"):
            m = m.to(device)
        if hasattr(m, "eval"):
            m.eval()
        out[name] = m
    return out


def load_qformer(variant: str, device, model_dir: Path | None = None, log=None):
    """Stage 1/2 projector (variant canonical | pubmed): the run's neuro_qformer_<variant>/ package when present
    (Q-Former or MLP, from config.json["projector"]), else the released Q-Former. For the canonical variant the run's
    canonical_semantic_banks.pt replaces the packaged banks."""
    key = f"qformer_{variant}"
    weights = run_file(model_dir, key)
    if weights is None:
        if model_dir is not None and log:
            log.warning("--model-dir %s has no %s; the released Q-Former is used", model_dir, RUN_FILES[key][0])
        note_source(key, RELEASED[key])
        if variant == "canonical":
            note_source("canonical_semantic_banks", RELEASED["canonical_semantic_banks"])
        return released_qformer(variant, device)
    import torch
    import torch.nn.functional as F
    from safetensors.torch import load_file

    cfg = read_json(weights.parent / "config.json")
    state = load_file(str(weights))
    if variant == "canonical":
        banks_path = run_file(model_dir, "canonical_semantic_banks")
        if banks_path is not None:
            banks = torch.load(banks_path, map_location="cpu", weights_only=False)
            diffs = {}
            for name in ("all", "network", "region", "function"):
                k = f"canonical_projection.{name}_bank"
                new = F.normalize(torch.as_tensor(banks[name]).float(), dim=1)
                old = state.get(k)
                diffs[name] = None if old is None or old.shape != new.shape else float((old.float() - new).abs().max())
                state[k] = new
            same = all(v is not None and v == 0.0 for v in diffs.values())
            note_source("canonical_semantic_banks", f"{rel_path(banks_path)} (" + (
                "identical to the banks packaged in the Q-Former" if same else f"replaced packaged banks; max |diff| {diffs}") + ")")
            if not same and log:
                log.warning("canonical_semantic_banks.pt differs from the packaged banks (%s); the .pt banks are used", diffs)
        else:
            note_source("canonical_semantic_banks", f"{rel_path(weights)} (packaged banks)")
    kind = cfg.get("projector", "qformer")
    model = neuro_projector_from_state(state, cfg, device=device).eval()
    note_source(key, weights.parent if kind == "qformer" else f"{rel_path(weights.parent)} (projector: {kind})")
    if log:
        log.info("model: %s <- %s (projector: %s)", key, weights.parent, kind)
    return model


def make_nvlm(device, model_dir: Path | None = None, log=None, *, generator: bool = True, adapter: bool = False,
              qformer_variant: str = "canonical", free_titles: bool | None = None):
    """NeuroVLM client whose trained modules come from model_dir (released ones without it). Released modules and the LM
    are set here at their pinned revisions; what the client still lazy-loads (SPECTER2, the masker) comes from the HF
    cache, checked to be at the pinned revisions first, so it never reads the HF main branch. Captions use Latin-script
    tokens only, and a [NETWORK] / [REGION] title is one of the canonical names of model_dir's stage-14 banks
    (title_vocabulary) unless ``free_titles`` (default: the script's --free-titles)."""
    from neurovlm import NeuroVLM

    hf_cache_pinned()
    nvlm = NeuroVLM(device=str(device))
    for name, attr in (("autoencoder", "_autoencoder"), ("proj_head_text_mse", "_proj_head_text_mse"),
                       ("proj_head_text_infonce", "_proj_head_text_infonce"),
                       ("proj_head_image_infonce", "_proj_head_image_infonce")):
        setattr(nvlm, attr, load_component(name, model_dir, log).to(nvlm.device).eval())
    if adapter:
        nvlm._neuro_adapter = load_component("neuro_adapter", model_dir, log).to(nvlm.device).eval()
    note_source("specter", RELEASED["specter"])
    if generator:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from neurovlm.resources.loaders import NEURO_QWEN_REPO_ID

        nvlm._neuro_qformer = load_qformer(qformer_variant, nvlm.device, model_dir, log)
        nvlm._neuro_qformer_variant = qformer_variant
        # as NeuroVLM's lazy loader, at the pinned revision
        rev = hf_pinned(NEURO_QWEN_REPO_ID)
        tok = AutoTokenizer.from_pretrained(NEURO_QWEN_REPO_ID, revision=rev, use_fast=True)
        if tok.pad_token_id is None:
            tok.pad_token = tok.eos_token
        dtype = torch.float32 if nvlm.device.type == "cpu" else torch.bfloat16
        lm = AutoModelForCausalLM.from_pretrained(NEURO_QWEN_REPO_ID, revision=rev, torch_dtype=dtype, device_map=None).to(nvlm.device).eval()
        if getattr(lm.config, "pad_token_id", None) is None:
            lm.config.pad_token_id = tok.pad_token_id
        lm.generation_config.suppress_tokens = list(latin_script_suppress_tokens(tok))  # Latin-script captions only
        free = _TITLES["free"] if free_titles is None else free_titles
        install_title_constraint(lm, tok, None if free else title_vocabulary(model_dir, device=nvlm.device, logger=log)["titles"])
        nvlm._neuro_qwen_tokenizer, nvlm._neuro_qwen_model = tok, lm
        note_source("neuroqwen", RELEASED["neuroqwen"])
        note_source("caption_titles", title_constraint_note(lm))
    return nvlm


# PubMed split
def pubmed_split(args) -> tuple[dict[str, np.ndarray], str]:
    """({"train"|"val"|"test": int64 PMIDs}, source): the run's pmids_split.pt when present, else the split flags of the
    pinned HF publications.parquet (set-identical to runs/final, as validate_models.py). Never data_dir."""
    p = run_file(args.model_dir, "pmids_split")
    if p is None:
        split = hf_split_pmids()
        source = f"hf:neurovlm/neuro_image_papers/publications.parquet@{hf_pinned('neurovlm/neuro_image_papers')[:8]} (split flags)"
    else:
        import torch

        raw = torch.load(p, map_location="cpu", weights_only=False)
        split, source = {k: np.asarray(raw[k]).astype(np.int64) for k in ("train", "val", "test")}, rel_path(p)
    note_source("pmids_split", source)
    return split, source


def pubmed_test_mask(args, df, log=None) -> np.ndarray:
    """Test mask over a PubMed table: the HF test flag, or the test pmids of the run's pmids_split.pt."""
    import pandas as pd

    flag = df["test"].fillna(False).astype(bool).to_numpy()
    p = run_file(args.model_dir, "pmids_split")
    if p is None:
        _MODEL_SOURCES.setdefault("pmids_split", "HF pubmed_text test flag")
        return flag
    import torch

    test = np.asarray(torch.load(p, map_location="cpu", weights_only=False)["test"])
    mask = pd.Series(df["pmid"]).isin(test).to_numpy()
    same = set(df.loc[mask, "pmid"].tolist()) == set(df.loc[flag, "pmid"].tolist())
    note_source("pmids_split", f"{rel_path(p)} (test: {len(set(test.tolist()))} pmids; "
                               f"{'identical to' if same else 'DIFFERS from'} the HF pubmed_text test flag)")
    if log:
        (log.info if same else log.warning)("PubMed test split from %s (%s the HF test flag)", p, "identical to" if same else "differs from")
    return mask


def encode_texts(specter, texts: Sequence[str], *, batch_size: int = 32):
    """L2-normalised SPECTER2 embeddings, CPU float [N, 768]."""
    import torch
    import torch.nn.functional as F

    chunks = []
    with torch.no_grad():
        for s in range(0, len(texts), batch_size):
            e = specter([str(t) for t in texts[s:s + batch_size]]).float()
            chunks.append(F.normalize(e, dim=1).detach().cpu())
    return torch.cat(chunks, dim=0)


def to_device_batches(fn, X, device, *, batch_size: int = 1024):
    """fn applied to row chunks of X on device; CPU result."""
    import torch

    outs = []
    with torch.no_grad():
        for s in range(0, len(X), batch_size):
            xb = torch.as_tensor(np.asarray(X[s:s + batch_size]) if not torch.is_tensor(X) else X[s:s + batch_size])
            outs.append(fn(xb.float().to(device)).detach().float().cpu())
    return torch.cat(outs, dim=0)


# brain maps
@cache
def mni_template():
    from nilearn.datasets import load_mni152_template

    return load_mni152_template(resolution=2)


def draw_brain(img, ax, *, cut_coords=None, threshold=1e-2, vmin=0.0, vmax=None, display_mode: str = "ortho"):
    """plot_stat_map into ax: one hue, no colorbar/cross/annotations; vmax=None scales to the map's maximum."""
    from nilearn.plotting import plot_stat_map

    ax.set_axis_off()
    return plot_stat_map(
        img, bg_img=mni_template(), axes=ax, display_mode=display_mode, cut_coords=cut_coords,
        cmap=BRAIN_CMAP, threshold=threshold, vmin=vmin, vmax=vmax, colorbar=False, draw_cross=False, annotate=False,
        black_bg=False, symmetric_cbar=False,
    )


# width / height of the sagittal (x), coronal (y) and axial (z) views of the MNI152 2 mm template (198 x 234 x 190 mm)
_MNI_MM = {"x": 198.0, "y": 234.0, "z": 190.0}
_VIEW_AXES = {"x": ("y", "z"), "y": ("x", "z"), "z": ("x", "y")}  # in-plane axes (horizontal, vertical) per view


def brain_stack_height(width_in: float, gap_in: float = 0.02) -> float:
    """Height of draw_brain_stack for a column width_in wide."""
    scale = width_in / _MNI_MM["y"]
    return sum(_MNI_MM[_VIEW_AXES[m][1]] * scale for m in "xyz") + 2 * gap_in


def draw_brain_stack(fig, img, x_in: float, y_top_in: float, width_in: float, *, cut_coords=None, gap_in: float = 0.02,
                     **kw):
    """Sagittal, coronal and axial views stacked top to bottom in a column (inches from the figure's bottom left),
    one mm-to-inch scale for all three as in nilearn's ortho display. Returns the three axes."""
    fw, fh = fig.get_size_inches()
    if cut_coords is None:
        cut_coords = cut_coords_for(img) or [0.0, 0.0, 0.0]
    scale = width_in / _MNI_MM["y"]
    axes, y = [], y_top_in
    for mode, c in zip("xyz", cut_coords):
        w = _MNI_MM[_VIEW_AXES[mode][0]] * scale
        h = _MNI_MM[_VIEW_AXES[mode][1]] * scale
        y -= h
        ax = fig.add_axes([(x_in + (width_in - w) / 2) / fw, y / fh, w / fw, h / fh])
        draw_brain(img, ax, cut_coords=[float(c)], display_mode=mode, **kw)
        axes.append(ax)
        y -= gap_in
    return axes


def cut_coords_for(img, activation_threshold=None):
    from nilearn.plotting import find_xyz_cut_coords

    try:
        return [float(c) for c in find_xyz_cut_coords(img, activation_threshold=activation_threshold)]
    except Exception:  # empty maps
        return None

# the template's brain (MNI152 2 mm, voxels > 0, +-2 mm) per axis, mm: draw_brain_views crops every view to it
BRAIN_MM = {"x": (-74.0, 74.0), "y": (-108.0, 74.0), "z": (-74.0, 84.0)}


def brain_views_mm(views: str = "xyz", *, vertical: bool = False) -> tuple[float, float]:
    """(width, height) in mm of draw_brain_views' views side by side (stacked when vertical), gaps left out."""
    ext = {a: hi - lo for a, (lo, hi) in BRAIN_MM.items()}
    ws = [ext[_VIEW_AXES[m][0]] for m in views]
    hs = [ext[_VIEW_AXES[m][1]] for m in views]
    return (max(ws), sum(hs)) if vertical else (sum(ws), max(hs))


def draw_brain_views(fig, img, box_in, *, views: str = "xyz", cut_coords=None, vertical: bool = False,
                     gap_in: float = 0.02, **kw):
    """Views of img (x sagittal, y coronal, z axial) side by side, or stacked when vertical, each cropped to the
    template's brain at one mm-to-inch scale, as large as fits and centred in box_in = (x, y, width, height), inches
    from the figure's bottom left. cut_coords = (x, y, z) in mm (find_xyz_cut_coords when None). Returns the axes."""
    fw, fh = fig.get_size_inches()
    x0, y0, bw, bh = box_in
    w_mm, h_mm = brain_views_mm(views, vertical=vertical)
    gaps = gap_in * (len(views) - 1)
    s = min((bw - (0.0 if vertical else gaps)) / w_mm, (bh - (gaps if vertical else 0.0)) / h_mm)  # inches per mm
    tot_w, tot_h = w_mm * s + (0.0 if vertical else gaps), h_mm * s + (gaps if vertical else 0.0)
    x, y = x0 + (bw - tot_w) / 2, y0 + (bh + tot_h) / 2  # the block's top left
    if cut_coords is None:
        cut_coords = cut_coords_for(img) or [0.0, 0.0, 0.0]
    cuts = dict(zip("xyz", cut_coords))
    axes = []
    for m in views:
        hor, ver = _VIEW_AXES[m]
        w, h = (BRAIN_MM[hor][1] - BRAIN_MM[hor][0]) * s, (BRAIN_MM[ver][1] - BRAIN_MM[ver][0]) * s
        if vertical:
            rect = [x + (tot_w - w) / 2, y - h, w, h]
            y -= h + gap_in
        else:
            rect = [x, y - tot_h / 2 - h / 2, w, h]
            x += w + gap_in
        ax = fig.add_axes([rect[0] / fw, rect[1] / fh, rect[2] / fw, rect[3] / fh])
        display = draw_brain(img, ax, cut_coords=[float(cuts[m])], display_mode=m, **kw)
        for cut in display.axes.values():
            cut.ax.set_xlim(*BRAIN_MM[hor])
            cut.ax.set_ylim(*BRAIN_MM[ver])
        axes.append(ax)
    return axes


def best_view(imgs, cut_coords, threshold: float = 1e-2, views: str = "xyz") -> str:
    """The view whose slice through cut_coords (mm) holds the most voxels above threshold in the sparsest of imgs
    (one grid), ties broken by the total, so every map shows activation."""
    import nibabel as nib

    ijk = np.rint(nib.affines.apply_affine(np.linalg.inv(imgs[0].affine), cut_coords)).astype(int)
    scores = {}
    for m in views:
        a = "xyz".index(m)
        k = [int(np.clip(ijk[a], 0, img.shape[a] - 1)) for img in imgs]
        n = [int((np.take(np.asarray(img.dataobj), i, axis=a) > threshold).sum()) for img, i in zip(imgs, k)]
        scores[m] = (min(n), sum(n))
    return max(views, key=lambda m: scores[m])


def load_network_list(*, include_only: Iterable[str] | None = None):
    """load_dataset("networks") as [(atlas, name, Nifti1Image), ...] in HF order; the ICA atlases only if listed."""
    import nibabel as nib

    nets = load_dataset("networks")
    out = []
    for atlas in nets:
        keep = atlas in include_only if include_only is not None else atlas not in ("UKBICA", "HCPICA")
        if not keep:
            continue
        for name, payload in nets[atlas].items():
            out.append((atlas, name, nib.Nifti1Image(payload["array"], affine=payload["affine"])))
    return out


# Figs. 2 and 3 draw every brain map (generated, atlas and true alike) at its top 5 % of in-mask voxels, about the extent
# of the maps in the submitted Fig. 3. A rank cut does not depend on a map's scale: half the maximum (the previous rule)
# kept up to a sixth of the brain on calibrated maps, whose global factor (x19 for the final head) clips the broad base
# of a peaked map at 1. Binary atlas maps keep all their voxels (ties at the cut are drawn, zeros never).
MAP_DISPLAY_TOP = 0.05
MAP_DISPLAY_RULE = ("the top 5 % of in-mask voxels of each map (ties at the cut drawn, zeros never; generated, atlas and "
                    "true maps alike; no cluster threshold)")


def display_vector(vec, top: float = MAP_DISPLAY_TOP) -> tuple[np.ndarray, float]:
    """The map with every voxel outside its top ``top`` fraction (and every zero) set to 0, and a threshold just below
    its smallest drawn value (plot_stat_map hides the values at or below its threshold)."""
    vec = np.asarray(vec, dtype=np.float32)
    keep = (vec >= np.quantile(vec, 1.0 - top)) & (vec > 0)
    if not keep.any():
        return np.zeros_like(vec), 0.0
    return np.where(keep, vec, 0.0).astype(np.float32), float(np.nextafter(vec[keep].min(), np.float32(0)))


def display_vmin(threshold: float, vmax: float) -> float:
    """Colour-scale floor that puts a drawn map's lowest value a third up the colour map, so every map spans the same
    visible range (light at its cut, dark at its maximum) whatever its cut."""
    return threshold - 0.5 * (vmax - threshold)


def mask_vector(masker, img) -> np.ndarray:
    """masker.transform as a 1-D float32 vector."""
    arr = np.asarray(masker.transform(img), dtype="float32")
    if arr.ndim == 2:
        arr = arr[0]
    return arr.reshape(-1)


def network_binary_vector(img, masker) -> np.ndarray:
    """Notebook 13: binary maps -> nearest resample; continuous maps -> binarised at the 95th percentile."""
    import nibabel as nib
    from nilearn.image import resample_img

    arr = img.get_fdata()
    if len(np.unique(arr)) == 2:
        res = resample_img(img, masker.affine_, interpolation="nearest", force_resample=True, copy_header=True)
    else:
        res = resample_img(img, masker.affine_, force_resample=True, copy_header=True)
        a = res.get_fdata()
        a[a < 0] = 0.0
        thr = np.percentile(a.flatten(), 95)
        a = (a >= thr).astype("float32")
        res = nib.Nifti1Image(a, affine=masker.affine_)
    return mask_vector(masker, res)


# text
PREFIX_RE = re.compile(r"(?i)\[(network|region|function)\]\s*")  # [NETWORK]/[Region]/... tokens of a generation


def split_generated(text: str) -> tuple[str | None, str]:
    """'[NETWORK]Title\\nbody' -> ('Title', 'body'); (None, text) without a line break. Tags are dropped as in
    validate_models.py (_common_models.strip_steering_tags)."""
    raw = strip_steering_tags(str(text)).strip()
    return tuple(raw.split("\n", 1)) if "\n" in raw else (None, raw)


_RENDERER = None


def text_width_in(s: str, font_pt: float, weight: str = "normal") -> float:
    """Rendered width of s in inches, measured at the 300 dpi output resolution."""
    global _RENDERER
    from matplotlib.backends.backend_agg import RendererAgg
    from matplotlib.font_manager import FontProperties

    if _RENDERER is None:
        _RENDERER = RendererAgg(16, 16, 300.0)
    w, _, _ = _RENDERER.get_text_width_height_descent(s, FontProperties(size=font_pt, weight=weight), ismath=False)
    return w / 300.0


def wrap_measured(text: str, width_in: float, font_pt: float, *, weight: str = "normal",
                  hyphens: bool = False) -> list[str]:
    """Greedy word wrap by measured width; words wider than the column are hyphen-broken; hyphens=True also breaks
    a word after its own hyphens ('self-' | 'benefitting'). Never truncates."""

    def fits(s: str) -> bool:
        return text_width_in(s, font_pt, weight) <= width_in

    lines: list[str] = []
    for para in str(text).replace("\r", "").split("\n"):
        words = para.split()
        if not words:
            continue
        # (piece, joins the previous piece without a space)
        pieces = ([(q, j > 0) for w in words for j, q in enumerate(re.findall(r"[^-]*-+|[^-]+", w))] if hyphens
                  else [(w, False) for w in words])
        cur = ""
        for w, joined in pieces:
            cand = (cur + w if joined else f"{cur} {w}") if cur else w
            if fits(cand):
                cur = cand
                continue
            if cur:
                lines.append(cur)
            while not fits(w):  # a single word wider than the column
                k = len(w) - 1
                while k > 1 and not fits(w[:k] + "-"):
                    k -= 1
                lines.append(w[:k] + "-")
                w = w[k:]
            cur = w
        if cur:
            lines.append(cur)
    return lines


def text_lines(title: str | None, body: str | None, *, font_pt: float, width_in: float) -> int:
    """Lines text_cell needs for the bold title + body."""
    n = len(wrap_measured(title, width_in, font_pt, weight="bold")) if title else 0
    return n + (len(wrap_measured(body, width_in, font_pt)) if body else 0)


def text_cell(ax, title: str | None, body: str | None, *, font_pt: float, width_in: float,
              line_spacing: float = 1.15, color: str = "black") -> int:
    """Complete bold title + wrapped body, top-aligned in ax (size the cell with text_lines). Returns the line count."""
    ax.set_axis_off()
    fig = ax.figure
    ax_h_in = ax.get_position().height * fig.get_figheight()
    lh = font_pt / 72.0 * line_spacing / max(ax_h_in, 1e-6)  # line height in axes fraction
    y = 1.0
    tl = wrap_measured(title, width_in, font_pt, weight="bold") if title else []
    if tl:
        ax.text(0, y, "\n".join(tl), transform=ax.transAxes, va="top", ha="left", fontsize=font_pt,
                fontweight="bold", color=color, linespacing=line_spacing)
        y -= lh * len(tl)
    bl = wrap_measured(body, width_in, font_pt) if body else []
    if bl:
        ax.text(0, y, "\n".join(bl), transform=ax.transAxes, va="top", ha="left", fontsize=font_pt,
                color=color, linespacing=line_spacing)
    return len(tl) + len(bl)


def panel_label(fig, letter: str, x: float, y: float, *, size_delta: float = 2.0):
    """Bold panel letter in figure coordinates."""
    import matplotlib.pyplot as plt

    return fig.text(x, y, letter, fontsize=plt.rcParams["font.size"] + size_delta, fontweight="bold", va="top", ha="left")


def column_header(fig, x0: float, x1: float, y: float, text: str, *, size_delta: float = 1.0):
    """Centred header over [x0, x1] (figure coordinates) with a rule underneath."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    fig.text((x0 + x1) / 2, y, text, ha="center", va="bottom", fontsize=plt.rcParams["font.size"] + size_delta)
    fig.add_artist(Line2D([x0, x1], [y - 0.004, y - 0.004], transform=fig.transFigure, color="black", lw=0.8))


# stats panels
def match_null_box(ax, match, null, *, ylabel: str, title: str, annotation: str | None = None,
                   annotation_size: float | None = None):
    """Match (blue) vs Null (grey) boxes with an optional significance bracket (annotation text at annotation_size pt,
    default the axis-label size)."""
    match = np.asarray(match, dtype=float)
    null = np.asarray(null, dtype=float)
    match, null = match[np.isfinite(match)], null[np.isfinite(null)]
    bp = ax.boxplot([match, null], positions=[0, 1], widths=0.55, tick_labels=["Match", "Null"],
                    showfliers=False, patch_artist=True, medianprops=dict(color="black", lw=1.0),
                    whiskerprops=dict(lw=0.8), capprops=dict(lw=0.8), boxprops=dict(lw=0.6))
    for patch, c in zip(bp["boxes"], (MATCH_COLOR, NULL_COLOR)):
        patch.set_facecolor(c)
        patch.set_alpha(0.8)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlim(-0.6, 1.6)
    ax.grid(axis="y", alpha=0.25)
    if annotation and len(match) and len(null):
        def whisker_top(v):  # matplotlib's upper whisker: largest value <= Q3 + 1.5 IQR
            q1, q3 = np.percentile(v, [25, 75])
            return float(v[v <= q3 + 1.5 * (q3 - q1)].max())

        top = max(whisker_top(match), whisker_top(null),
                  match.max() if len(match) < 20 else -np.inf, null.max() if len(null) < 20 else -np.inf)
        lo = min(match.min(), null.min())
        span = max(top - lo, 1e-6)
        y = top + 0.08 * span
        ax.plot([0, 0, 1, 1], [y, y + 0.03 * span, y + 0.03 * span, y], color="black", lw=0.7)
        ax.text(0.5, y + 0.04 * span, annotation, ha="center", va="bottom",
                fontsize=annotation_size or ax.xaxis.label.get_size())
        ax.set_ylim(lo - 0.05 * span, y + 0.22 * span)
    return bp


def recall_axes(ax, *, title: str | None = None, chance_label: str = "Chance"):
    """Recall@k panel: chance diagonal, square box (diagonal at 45 degrees), default data margins."""
    ax.plot([0, 1], [0, 1], ls="--", color=CHANCE_COLOR, lw=0.9, label=chance_label)
    ax.set_xlabel(r"$k/n$ (fraction of all candidates)")
    ax.set_ylabel(r"Recall@$k$")
    ax.set_box_aspect(1)
    ax.grid(alpha=0.25)
    if title:
        ax.set_title(title)


def trapz_auc(x, y) -> float:
    return float(np.trapezoid(np.asarray(y, dtype=float), np.asarray(x, dtype=float)))


def deranged_perm(n: int, rng: np.random.Generator) -> np.ndarray:
    """Permutation with no fixed points (as in the notebooks)."""
    if n < 2:
        return np.arange(n)
    perm = rng.permutation(n)
    fixed = np.flatnonzero(perm == np.arange(n))
    if len(fixed) > 1:
        perm[fixed] = np.roll(perm[fixed], 1)
    elif len(fixed) == 1:
        i = fixed[0]
        j = (i + 1) % n
        perm[i], perm[j] = perm[j], perm[i]
    return perm


def welch_log10p(x, y) -> dict:
    """Welch t-test; the two-sided p is computed with mpmath, so log10 p can go below -300."""
    import mpmath as mp

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    x, y = x[np.isfinite(x)], y[np.isfinite(y)]
    nx, ny = len(x), len(y)
    if nx < 2 or ny < 2:
        return {"t": np.nan, "df": np.nan, "log10_p": np.nan, "n_match": nx, "n_null": ny}
    vx, vy = x.var(ddof=1), y.var(ddof=1)
    se2 = vx / nx + vy / ny
    if se2 <= 0:
        return {"t": np.nan, "df": np.nan, "log10_p": np.nan, "n_match": nx, "n_null": ny}
    t = (x.mean() - y.mean()) / np.sqrt(se2)
    df = se2 ** 2 / ((vx / nx) ** 2 / (nx - 1) + (vy / ny) ** 2 / (ny - 1))
    mp.mp.dps = 60
    df_mp, t_mp = mp.mpf(str(df)), mp.mpf(str(abs(t)))
    p = mp.betainc(df_mp / 2, mp.mpf("0.5"), 0, df_mp / (df_mp + t_mp ** 2), regularized=True)
    log10_p = float(mp.log10(p)) if p > 0 else -np.inf
    return {"t": float(t), "df": float(df), "log10_p": log10_p, "p": format_p(log10_p), "n_match": nx, "n_null": ny,
            "match_mean": float(x.mean()), "null_mean": float(y.mean()), "cohen_d": cohen_d(x, y)}


def cohen_d(x, y) -> float:
    """(mean x - mean y) / sqrt((var x + var y) / 2)."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    x, y = x[np.isfinite(x)], y[np.isfinite(y)]
    if len(x) < 2 or len(y) < 2:
        return float("nan")
    sd = np.sqrt((x.var(ddof=1) + y.var(ddof=1)) / 2.0)
    return float((x.mean() - y.mean()) / sd) if sd > 0 else float("nan")


def binom_log10_sf(k: int, n: int, p0: float) -> float:
    """log10 P(X >= k), X ~ Binomial(n, p0) (one-sided exact binomial test, mpmath)."""
    import mpmath as mp

    if k <= 0:
        return 0.0
    mp.mp.dps = 60
    p = mp.betainc(k, n - k + 1, 0, mp.mpf(str(p0)), regularized=True)  # P(X >= k) = I_p0(k, n - k + 1)
    return float(mp.log10(p)) if p > 0 else -np.inf


def format_p(log10_p: float) -> str:
    """p from its log10, e.g. -16.36 -> '4e-17', -1.2 -> '0.063'."""
    if not np.isfinite(log10_p):
        return "n/a" if np.isnan(log10_p) else "0"
    if log10_p >= -3:
        return f"{10 ** log10_p:.2g}"
    e = int(np.floor(log10_p))
    m = 10 ** (log10_p - e)
    if round(m) >= 10:
        m, e = 1.0, e + 1
    return f"{m:.0f}e{e}"


def sig_text(log10_p: float) -> str:
    """'*' for p < 1e-100 (paper convention), else 'p=<value>'."""
    if np.isnan(log10_p):
        return "n/a"
    if log10_p < -100.0:
        return "*"
    return f"p={format_p(log10_p)}"


# output
def finish(fig, stem: str, args, log, *, text_dump: dict | None = None, components: Sequence[str] = ()) -> list[Path]:
    """Save PNG + SVG and <stem>_text.json (with exclusion counts and model_sources); --paper copies the PNG into
    paper/figures/. Declared components that were never loaded (e.g. precomputed latents) are filled by a static
    lookup and marked as such."""
    import matplotlib.pyplot as plt

    missing = [c for c in components if c not in _MODEL_SOURCES]
    if missing:
        static = resolve_sources(args.model_dir)
        for comp in missing:
            _MODEL_SOURCES[comp] = static.get(comp, "unknown") + " [static lookup; not recorded at load time]"
    paths = save_fig(fig, stem, args.out)
    if args.paper:
        import shutil

        PAPER_FIG_DIR.mkdir(parents=True, exist_ok=True)
        base = next((k for k in PAPER_NAMES if stem.startswith(k)), None)
        dest = PAPER_FIG_DIR / ((PAPER_NAMES[base] + stem[len(base):]) if base else stem)
        shutil.copy2(Path(args.out) / f"{stem}.png", dest.with_suffix(".png"))
        paths.append(dest.with_suffix(".png"))
    if text_dump is not None:
        p = Path(args.out) / f"{stem}_text.json"
        if _EXCLUSION_COUNTS:
            text_dump = {**text_dump, "eval_exclusions": dict(_EXCLUSION_COUNTS)}
        md = args.model_dir
        text_dump = {**text_dump, "model_dir": rel_path(md) if md else None, "cache_tag": cache_tag(args),
                     "model_sources": dict(sorted(_MODEL_SOURCES.items())) or {"note": "no trained model is used"}}
        write_json(p, text_dump)
        paths.append(p)
    for p in paths:
        log.info("wrote %s", p)
    plt.close(fig)
    free_gpu()
    return paths
