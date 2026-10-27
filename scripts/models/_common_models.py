"""Helpers shared by the scripts/models/ stages: run dirs, inputs, batched loaders, artifact I/O, map calibration,
the brain-to-text projectors of stages 13/14, the caption script (Latin-script tokens only) and the caption titles
(stage-14 canonical names after a [NETWORK] / [REGION] steering token). Stages use
``from _common_models import *``.
"""
from __future__ import annotations

import argparse
import contextlib
import functools
import re
import shutil
import sys
import time
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "pipeline"))  # scripts/pipeline/
from common import (  # noqa: E402
    DEFAULT_OUT_ROOT, apply_pubmed_exclusions, artifacts_exist, base_parser, data_dir, fetch_aal, finalize_args, get_logger, hf_pinned,
    hf_split_pmids, read_json, seed_worker, select_device, setup_torch, sha256_file, timed, write_json,
)

RUNS_ROOT = DEFAULT_OUT_ROOT / "runs"

__all__ = [
    # from scripts/pipeline/common.py
    "artifacts_exist", "base_parser", "data_dir", "fetch_aal", "finalize_args", "get_logger", "hf_pinned", "read_json", "select_device",
    "setup_torch", "sha256_file", "timed", "write_json",
    # run dirs, loaders, artifacts
    "RUNS_ROOT", "add_model_args", "resolve_run_dir", "find_input", "run_input", "pmid_split", "install_artifacts", "seed_all",
    "lm_sdpa_backends", "gpu_can_hold",
    "load_torch", "BatchedTensorDataset", "make_batched_loader", "place_splits", "autocast_context",
    "bf16_training_forward", "fast_mlp_eval", "encode_in_batches", "coordinate_filter", "notebook_split",
    "save_module_and_state", "load_module_or_state", "RELEASED", "released_model", "released_qformer", "run_or_released", "fresh_trainer_dir",
    "summarise_trainer_run",
    "write_stage_summary",
    # map calibration
    "MAP_CALIBRATION_RULES", "MAP_CALIBRATION_DEFAULT", "MAP_CALIBRATION_FILE", "calibrate_maps", "map_calibration",
    # brain-to-text projectors
    "add_projector_args", "build_projector", "projector_config", "mlp_config_from_state", "state_projector_kind",
    "package_qformer_block", "neuro_projector_from_state", "package_projector_kind", "load_projector",
    "write_projector_package",
    # caption script
    "CAPTION_CHAR_RANGES", "caption_char_ok", "latin_script_suppress_tokens", "STEERING_TAG_RE", "strip_steering_tags",
    # title constraint (default)
    "TITLE_LEVELS", "TITLE_END_TEXTS", "TITLE_VOCABULARY_FILE", "bank_titles", "title_vocabulary", "caption_title",
    "noncanonical_titles", "title_trie", "newline_token_ids", "TitleConstraint", "TitleConstrainedGenerate",
    "install_title_constraint", "title_constraint_note", "free_titles", "constrained_titles",
]


def add_model_args(parser: argparse.ArgumentParser, *, amp_default: str = "fp32"):
    """Flags every model stage adds to common.base_parser."""
    parser.add_argument("--run-name", default=None,
                        help="Run directory under scripts/outputs/runs/ (default: timestamp; 'smoke' with --smoke).")
    parser.add_argument("--inputs-from", type=Path, default=None,
                        help="Run directory to read upstream artifacts from (default: this run, then data_dir).")
    parser.add_argument("--install", action="store_true", help="Copy the final artifacts into data_dir under the notebook names.")
    parser.add_argument("--amp", choices=["bf16", "fp32"], default=amp_default, help="Autocast dtype of the training forward on CUDA.")


def resolve_run_dir(args: argparse.Namespace) -> Path:
    """Create scripts/outputs/runs/<run_name>/ and return it."""
    args.run_name = args.run_name or ("smoke" if args.smoke else time.strftime("%Y%m%d-%H%M%S"))
    run_dir = RUNS_ROOT / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def find_input(name: str, *, run_dir: Path, args: argparse.Namespace, required: bool = True,
               data_subdirs: Sequence[str] = (), explicit: Path | None = None, stage_hint: str = "") -> Path | None:
    """First existing file of: explicit path, --inputs-from, run_dir, data_dir (and its subdirs)."""
    candidates = [Path(explicit)] if explicit is not None else []
    if args.inputs_from:
        candidates.append(Path(args.inputs_from) / name)
    dd = data_dir()
    candidates += [Path(run_dir) / name, dd / name, *(dd / sub / name for sub in data_subdirs)]
    for cand in candidates:
        if cand.exists():
            return cand
    if required:
        hint = f" ({stage_hint})" if stage_hint else ""
        raise FileNotFoundError(f"Required input {name!r}{hint} not found. Looked in:\n  " + "\n  ".join(map(str, candidates)))
    return None


def run_input(name: str, *, run_dir: Path, args: argparse.Namespace, explicit: Path | None = None) -> Path | None:
    """First existing file of: explicit path, --inputs-from, run_dir. Never data_dir (its copies may be stale)."""
    cands = ([Path(explicit)] if explicit is not None else []) + ([Path(args.inputs_from) / name] if args.inputs_from else [])
    return next((c for c in cands + [Path(run_dir) / name] if c.exists()), None)


def pmid_split(*, run_dir: Path, args: argparse.Namespace, logger, explicit: Path | None = None) -> tuple[dict, str, str]:
    """(split, source, sha256): stage 08's pmids_split.pt from the run (run_input), else the train/val/test flags of the
    pinned HF publications.parquet (the same sets as runs/final). Never data_dir."""
    path = run_input("pmids_split.pt", run_dir=run_dir, args=args, explicit=explicit)
    if path is not None:
        split = load_torch(path)
        split = {k: np.asarray(split[k]).astype(np.int64) for k in ("train", "val", "test")}
        source, digest = str(path), sha256_file(path)
    else:
        split = hf_split_pmids()
        pq = hf_pinned("neurovlm/neuro_image_papers", "publications.parquet")
        source = f"hf:neurovlm/neuro_image_papers/publications.parquet@{hf_pinned('neurovlm/neuro_image_papers')[:8]} (split flags)"
        digest = sha256_file(pq)
    logger.info("pmid split: %s (train/val/test %d/%d/%d)", source, *(len(split[k]) for k in ("train", "val", "test")))
    return split, source, digest


def install_artifacts(paths: Mapping[str, Path], logger) -> None:
    """Copy {notebook file name: run-dir path} into data_dir."""
    dd = data_dir()
    for name, src in paths.items():
        src = Path(src)
        if not src.exists():
            logger.warning("install: %s missing, skipped", src)
            continue
        shutil.copy2(src, dd / name)
        logger.info("installed %s -> %s", src.name, dd / name)


def seed_all(seed: int) -> None:
    import random
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def lm_sdpa_backends(deterministic: bool, logger) -> None:
    """Attention kernels of the frozen LM: no cuDNN attention when deterministic (its backward is not); flash and
    mem-efficient then run their deterministic backward."""
    import torch

    if deterministic:
        torch.backends.cuda.enable_cudnn_sdp(False)
    b = torch.backends.cuda
    logger.info("SDPA backends: flash %s, mem-efficient %s, cuDNN %s, math %s", b.flash_sdp_enabled(), b.mem_efficient_sdp_enabled(),
                b.cudnn_sdp_enabled(), b.math_sdp_enabled())


def gpu_can_hold(nbytes: int, device) -> bool:
    """True if nbytes fit on a CUDA device with 4 GB left for the model and optimizer."""
    import torch

    if torch.device(device).type != "cuda":
        return False
    free, _ = torch.cuda.mem_get_info(device)
    return nbytes + (4 << 30) < free


def load_torch(path: Path | str, *, weights_only: bool = False):
    """torch.load onto the CPU (some cached files were saved from CUDA); mmap when the file format allows it."""
    import torch

    try:
        return torch.load(path, map_location="cpu", weights_only=weights_only, mmap=True)
    except (RuntimeError, ValueError, TypeError):
        return torch.load(path, map_location="cpu", weights_only=weights_only)


class BatchedTensorDataset:
    """Dict of tensors indexed by a list of rows, so a batch is one gather (use with make_batched_loader)."""

    def __init__(self, **tensors):
        self.tensors = tensors

    def __len__(self):
        return len(next(iter(self.tensors.values())))

    def __getitem__(self, idx):
        import torch

        idx = torch.as_tensor(idx, dtype=torch.long, device=next(iter(self.tensors.values())).device)
        return {k: v.index_select(0, idx) for k, v in self.tensors.items()}


def make_batched_loader(dataset: BatchedTensorDataset, *, batch_size: int, shuffle: bool, seed: int = 0):
    """DataLoader yielding whole-batch dicts from a BatchedTensorDataset."""
    import torch
    from torch.utils.data import BatchSampler, DataLoader, RandomSampler, SequentialSampler

    rows = range(len(dataset))
    sampler = RandomSampler(rows, generator=torch.Generator().manual_seed(seed)) if shuffle else SequentialSampler(rows)
    return DataLoader(dataset, batch_size=None, sampler=BatchSampler(sampler, batch_size=batch_size, drop_last=False),
                      num_workers=0, pin_memory=False, generator=torch.Generator().manual_seed(seed), worker_init_fn=seed_worker)


def place_splits(splits, device, *, logger):
    """Move every split's tensors to device if they fit, else keep them on (pinned) CPU. Returns (splits, on_device)."""
    import torch

    nbytes = sum(t.numel() * t.element_size() for s in splits.values() for t in s.values())
    on_device = gpu_can_hold(nbytes, device)
    out = {}
    for name, tensors in splits.items():
        if on_device:
            out[name] = {k: v.to(device) for k, v in tensors.items()}
        else:
            out[name] = {k: (v.pin_memory() if torch.cuda.is_available() else v) for k, v in tensors.items()}
    logger.info("datasets (%.2f GB) placed on %s", nbytes / 2**30, device if on_device else "CPU")
    return out, on_device


def autocast_context(device, amp: str):
    """bf16 autocast on CUDA when amp == 'bf16', else a no-op."""
    import torch

    if amp == "bf16" and torch.device(device).type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return contextlib.nullcontext()


@contextlib.contextmanager
def bf16_training_forward(model, device, amp: str):
    """bf16 autocast of model.forward in training mode only, so the library trainer's evaluation stays fp32."""
    import torch

    if not (amp == "bf16" and torch.device(device).type == "cuda"):
        yield model
        return
    original = model.forward

    def forward(*args, **kwargs):
        if model.training:
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                return original(*args, **kwargs)
        return original(*args, **kwargs)

    model.forward = forward
    try:
        yield model
    finally:
        del model.forward  # back to the class method, so the module pickles cleanly


@contextlib.contextmanager
def fast_mlp_eval(device, enabled: bool = True):
    """Run the library evaluator's per-voxel rank AUROC (neurovlm.evaluation.mlp._safe_auroc) on the GPU.

    Same arithmetic (descending argsort, cumulative TPR/FPR, trapezoid); on the CPU it costs ~17 s per epoch.
    Checked against the original on entry, restored on exit.
    """
    import torch
    import neurovlm.evaluation.mlp as emlp

    if not enabled or torch.device(device).type != "cuda":
        yield False
        return
    original = emlp._safe_auroc
    dev = torch.device(device)

    def fast(target, probability):
        t = (target.reshape(-1) > 0.5).float().to(dev, non_blocking=True)
        p = probability.reshape(-1).float().to(dev, non_blocking=True)
        positives = int(t.sum())
        negatives = int(t.numel() - positives)
        if positives == 0 or negatives == 0:
            return float("nan")
        ranked = t[p.argsort(descending=True)]
        tpr = ranked.cumsum(0) / positives
        fpr = (1 - ranked).cumsum(0) / negatives
        zero = torch.zeros(1, device=dev)
        return float(torch.trapezoid(torch.cat((zero, tpr)), torch.cat((zero, fpr))))

    g = torch.Generator().manual_seed(0)
    tt = (torch.rand(64, 2000, generator=g) > 0.9).float()
    pp = torch.rand(64, 2000, generator=g)
    ref, got = original(tt, pp), fast(tt, pp)
    if abs(ref - got) > 1e-5:
        raise RuntimeError(f"fast_mlp_eval self-check failed: {ref} vs {got}")
    emlp._safe_auroc = fast
    try:
        yield True
    finally:
        emlp._safe_auroc = original


def encode_in_batches(fn, x, *, batch_size: int = 4096, device=None, out_device="cpu"):
    """torch.cat of fn(batch) over batches of x, under no_grad."""
    import torch

    chunks = []
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            batch = x[start:start + batch_size]
            if device is not None:
                batch = batch.to(device)
            chunks.append(fn(batch).detach().to(out_device))
    return torch.cat(chunks, dim=0)


# generated maps (text-to-brain head + decoder, or anatomical head): one global factor per model, fitted on the validation
# split, then clipped at 1 ("global"); "max" divides each map by its own maximum (the previous rule, --rescale-max)
MAP_CALIBRATION_RULES = ("global", "max", "none")
MAP_CALIBRATION_DEFAULT = "global"
MAP_CALIBRATION_FILE = "map_calibration.json"


def calibrate_maps(maps, rule: str = MAP_CALIBRATION_DEFAULT, *, factor: float | None = None, eps: float = 1e-8):
    """Calibrate generated maps (last axis = voxels); torch or numpy in, same type out.

    "global": multiply every map by the model's factor (map_calibration) and clip at 1, so a weak map stays weak.
    "max": divide each map by its own maximum (every map peaks at 1). "none": raw sigmoid maps. Percentile Dice does not
    change; encoder-based scores do.
    """
    if rule == "none":
        return maps
    if rule == "global":
        if factor is None:
            raise ValueError("map calibration 'global' needs the model's factor: map_calibration(run_dir)[kind]")
        if isinstance(maps, np.ndarray):
            return np.minimum(maps * np.float32(factor), 1).astype(maps.dtype if maps.dtype.kind == "f" else np.float32, copy=False)
        return (maps * factor).clamp(max=1.0)
    if rule != "max":
        raise ValueError(f"unknown map calibration {rule!r}; choose from {MAP_CALIBRATION_RULES}")
    if isinstance(maps, np.ndarray):
        peak = np.maximum(maps.max(axis=-1, keepdims=True), eps)
        return (maps / peak).astype(maps.dtype if maps.dtype.kind == "f" else np.float32, copy=False)
    peak = maps.amax(dim=-1, keepdim=True).clamp_min(eps)
    return maps / peak


def map_calibration(run_dir: Path | None, *, device="cpu", logger=None, specter=None) -> dict:
    """Global calibration factors of a run's generated maps: {"head": c, "adapter": c, "source": {...}}.

    For each model, c is the median over the PubMed validation split (leakage-audit items excluded) of 1 / the peak of its
    sigmoid map. Training maps all peak at 1, so c puts the median generated map at 1. Fitted once (SPECTER2 on 2,961
    texts, about 30 s) and cached as <run_dir>/map_calibration.json; refitted when a model file changes. Models missing
    from run_dir (or run_dir None) are the released ones; with only released models the cache is
    scripts/outputs/released_map_calibration.json.
    """
    import torch
    import torch.nn.functional as F
    from neurovlm.data import load_dataset
    from neurovlm.models import ProjHead
    from neurovlm.models.base import NeuroAutoEncoder

    files = {"autoencoder": ("autoencoder.pt", lambda: NeuroAutoEncoder(seed=0, out="logit")),
             "proj_head_text_mse": ("proj_head_text_mse.pt", lambda: ProjHead(768, 512, 384)), "neuro_adapter": ("adapter_anat_full.pt", None)}
    paths = {k: Path(run_dir) / f for k, (f, _) in files.items() if run_dir is not None and (Path(run_dir) / f).exists()}
    key = {k: f"{paths[k].name}:{sha256_file(paths[k])}" if k in paths else f"released:{RELEASED[k][0]}@{hf_pinned(RELEASED[k][0])[:8]}"
           for k in files}
    cache = Path(run_dir) / MAP_CALIBRATION_FILE if paths else DEFAULT_OUT_ROOT / "released_map_calibration.json"
    if cache.exists() and read_json(cache).get("source", {}).get("models") == key:
        return read_json(cache)

    t0 = time.perf_counter()
    split_path = Path(run_dir) / "pmids_split.pt" if run_dir is not None else None
    if split_path is not None and split_path.exists():
        val, split_src = np.sort(np.asarray(load_torch(split_path)["val"])), split_path.name
    else:
        val, split_src = hf_split_pmids()["val"], f"neurovlm/neuro_image_papers@{hf_pinned('neurovlm/neuro_image_papers')[:8]} val flags"
    val = val[apply_pubmed_exclusions(val, logger=logger, label="PubMed val (map calibration)")]
    df = load_dataset("pubmed_text").drop_duplicates("pmid").set_index("pmid")
    rows = df.loc[val]
    texts = (rows["name"].fillna("").astype(str) + " [SEP] " + rows["description"].fillna("").astype(str)).tolist()
    models = {k: load_module_or_state(paths[k], builder=b, device=device) if k in paths else released_model(k, device)
              for k, (_, b) in files.items()}
    specter = specter if specter is not None else released_model("specter", device)
    generators = {"head": lambda z: models["autoencoder"].decoder(models["proj_head_text_mse"](z)), "adapter": models["neuro_adapter"]}
    out = {"rule": "global: maps * factor, clipped at 1"}
    with torch.no_grad():  # as validate_models.eval_generated_maps
        text_lat = torch.cat([specter(texts[s:s + 32]).detach().float().cpu() for s in range(0, len(texts), 32)])
        for kind, fn in generators.items():
            peaks = torch.cat([torch.sigmoid(fn(F.normalize(text_lat[s:s + 512].to(device), dim=1))).float().amax(dim=1).cpu()
                               for s in range(0, len(text_lat), 512)]).numpy().astype(np.float64)
            out[kind] = float(np.median(1.0 / np.maximum(peaks, 1e-8)))
            out[f"{kind}_peak_quantiles"] = {q: float(np.quantile(peaks, float(q))) for q in ("0.01", "0.5", "0.99")}
    out["source"] = {"models": key, "split": split_src, "n_val": len(texts), "factor": "median over validation maps of 1 / map peak"}
    write_json(cache, out)
    if logger is not None:
        logger.info("map calibration %s: head %.4f, adapter %.4f (%d validation maps, %.0fs)", cache, out["head"], out["adapter"],
                    len(texts), time.perf_counter() - t0)
    return out


def coordinate_filter(pmids: np.ndarray, *, max_coords: int = 100) -> np.ndarray:
    """Mask of studies with <= max_coords peak coordinates (notebooks 08/20)."""
    import pandas as pd
    from neurovlm.data import load_dataset

    df_coords = load_dataset("pubmed_coordinates")
    counts = df_coords[df_coords["pmid"].isin(pmids)]["pmid"].value_counts()
    return pd.Series(pmids).isin(counts.index[counts <= max_coords].values).to_numpy()


def notebook_split(n: int):
    """80/10/10 train/val/test indices as notebook 08 (train_test_split, seeds 0 then 1)."""
    import torch
    from sklearn.model_selection import train_test_split

    inds = torch.arange(n)
    train_inds, test_inds = train_test_split(inds, train_size=0.8, random_state=0)
    test_inds, val_inds = train_test_split(test_inds, train_size=0.5, random_state=1)
    return train_inds, val_inds, test_inds


def save_module_and_state(module, stem: Path, *, logger) -> None:
    """Write <stem>.pt (whole module on CPU, as the notebooks) and <stem>.safetensors (state dict)."""
    import torch
    from safetensors.torch import save_file

    module = module.cpu().eval()
    pt = Path(stem).with_suffix(".pt")
    torch.save(module, pt)
    save_file({k: v.detach().contiguous() for k, v in module.state_dict().items()}, str(Path(stem).with_suffix(".safetensors")))
    logger.info("saved %s (+ .safetensors)", pt)


def load_module_or_state(path: Path, *, builder=None, device="cpu"):
    """Load a module pickle (.pt), or a state dict (.safetensors or .pt) into builder()."""
    import torch

    path = Path(path)
    if path.suffix == ".safetensors":
        from safetensors.torch import load_file

        module = builder()
        module.load_state_dict(load_file(str(path)), strict=True)
        return module.to(device).eval()
    obj = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(obj, dict) and builder is not None:
        module = builder()
        module.load_state_dict(obj, strict=True)
        return module.to(device).eval()
    return obj.to(device).eval()


# released weights by name: (HF repo, file), loaded at the revision pinned in scripts/config/hf_manifest.json
RELEASED = {
    "autoencoder": ("neurovlm/NeuroAutoEncoder", "autoencoder.safetensors"),
    "proj_head_text_infonce": ("neurovlm/ProjectionHeads", "proj_head_text_infonce.safetensors"),
    "proj_head_image_infonce": ("neurovlm/ProjectionHeads", "proj_head_image_infonce.safetensors"),
    "proj_head_text_mse": ("neurovlm/ProjectionHeads", "proj_head_text_mse.safetensors"),
    "neuro_adapter": ("neurovlm/NeuroAdapter", "model.safetensors"),
}


def _released_module(name: str):
    from safetensors.torch import load_file
    from neurovlm.models import ProjHead
    from neurovlm.models.base import NeuroAutoEncoder

    repo, fname = RELEASED[name]
    if name == "neuro_adapter":  # as neurovlm.resources.loaders._load_neuro_adapter
        from neurovlm.models.adapter import InterleavedDecoderAdapter

        cfg = read_json(hf_pinned(repo, "config.json")).get("adapter", {})
        module = InterleavedDecoderAdapter(_released_module("autoencoder"), _released_module("proj_head_text_mse"), **cfg)
    elif name == "autoencoder":
        module = NeuroAutoEncoder(seed=0, out="logit")
    else:
        module = ProjHead(384, 384, 384) if name == "proj_head_image_infonce" else ProjHead(768, 512, 384)
    module.load_state_dict(load_file(str(hf_pinned(repo, fname))), strict=True)
    return module


def released_model(name: str, device="cpu"):
    """Released model on device, in eval mode: the pinned HF safetensors for the RELEASED names, else
    neurovlm.models.load_model(name) (SPECTER2; the library takes no revision, fetch_hf.py pins its cache ref)."""
    if name in RELEASED:
        return _released_module(name).to(device).eval()
    from neurovlm.models import load_model

    model = load_model(name)
    if hasattr(model, "to"):
        model = model.to(device)
    if hasattr(model, "eval"):
        model.eval()
    elif hasattr(model, "specter"):  # Specter is a wrapper, not an nn.Module
        model.specter.eval()
    return model


def run_or_released(name: str, filename: str, *, run_dir: Path, args: argparse.Namespace, device, logger, builder=None,
                    explicit: Path | None = None):
    """(module, source): the run's ``filename`` (explicit, --inputs-from, run dir), else the pinned released ``name``.
    Never data_dir."""
    path = run_input(filename, run_dir=run_dir, args=args, explicit=explicit)
    if path is not None:
        module, source = load_module_or_state(path, builder=builder, device=device), str(path)
    else:
        repo = RELEASED[name][0]
        module, source = released_model(name, device), f"released:{repo}@{hf_pinned(repo)[:8]}"
    logger.info("%s: %s", name, source)
    return module, source


def released_qformer(variant: str = "canonical", device="cpu"):
    """Released NeuroQFormer package (canonical: neurovlm/NeuroQformer, pubmed: neurovlm/NeuroQformer-PubMed) at the
    pinned revision; same construction as neurovlm.resources.loaders._load_neuro_qformer."""
    from safetensors.torch import load_file
    from neurovlm.models.qformer import NeuroQFormer

    repo = {"canonical": "neurovlm/NeuroQformer", "pubmed": "neurovlm/NeuroQformer-PubMed"}[variant]
    cfg = read_json(hf_pinned(repo, "config.json"))
    payload = {"state_dict": load_file(str(hf_pinned(repo, "model.safetensors")), device="cpu"), "config": cfg.get("qformer", {}),
               "canonical_metadata": cfg.get("canonical_metadata", {})}
    model = NeuroQFormer.from_state_dict_payload(payload, map_location=device).eval()
    model.projection_temp, model.canonical_basis = 0.05, "all"
    return model


def fresh_trainer_dir(output_root: Path, run_id: str) -> None:
    """Delete an earlier library-trainer run dir (its manifest rejects config changes)."""
    path = Path(output_root) / run_id
    if path.exists():
        shutil.rmtree(path)
    Path(output_root).mkdir(parents=True, exist_ok=True)


def summarise_trainer_run(result, *, seconds: float) -> dict:
    """JSON summary of an MLPTrainResult, with its train/val loss curves."""
    out = {
        "run_dir": str(result.run_dir),
        "best_checkpoint": str(result.best_checkpoint),
        "epochs_completed": int(result.epochs_completed),
        "best_metric": float(result.best_metric),
        "seconds": float(seconds),
        "sec_per_epoch": float(seconds) / max(int(result.epochs_completed), 1),
        "test_metrics": {k: (float(v) if isinstance(v, (int, float, np.floating)) else v) for k, v in dict(result.test_metrics or {}).items()},
    }
    history = Path(result.run_dir) / "metrics" / "history.csv"
    if history.exists():
        import pandas as pd

        hist = pd.read_csv(history)
        for split in ("val", "train"):
            rows = hist[(hist["split"] == split) & (hist["metric"] == "loss")].dropna(subset=["epoch"]).sort_values("epoch")
            out[f"{split}_loss_curve"] = [[int(e), float(v)] for e, v in zip(rows["epoch"], rows["value"])]
    return out


def write_stage_summary(run_dir: Path, stage: str, payload: dict) -> None:
    payload = dict(payload)
    payload.setdefault("stage", stage)
    payload.setdefault("ts", time.strftime("%Y-%m-%dT%H:%M:%S"))
    write_json(Path(run_dir) / f"{stage}.summary.json", payload)


# Brain-to-text projectors (stages 13/14): (autoencoder latent [B, 384], semantic vector [B, 384]) -> 32 NeuroQwen3
# input tokens [B, 32, 1024], plus semantic_from_visual for the alignment loss. Two interchangeable modules:
#   mlp      ResidualMLPProjector below (default; mlp_m of review analysis A5.1: width 704, depth 2, 28.34 M parameters)
#   qformer  neurovlm.models.qformer.QFormer (6 layers, 28.46 M parameters): the first submission's projector, deprecated
#            (constructing one warns); kept for the A5 projector comparison and old checkpoints
# Both are wrapped in the library's NeuroQFormer, so NeuroVLM.generate_text accepts either. Packages store the kind in
# config.json["projector"]; packages without the key are Q-Formers.
PROJECTOR_KINDS = ("qformer", "mlp")
MLP_DEFAULTS = {"width": 704, "depth": 2, "expansion": 2, "dropout": 0.05, "head": "tokens", "align_hidden": 512}  # A5 mlp_m


@functools.cache
def _projector_classes() -> dict:
    """Define the MLP projector classes on first use (importing this module does not import torch)."""
    import torch
    import torch.nn.functional as F
    from torch import nn

    class ResidualBlock(nn.Module):
        """Pre-LN residual MLP block: x + Linear(Dropout(GELU(Linear(LN(x)))))."""

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
        """Drop-in for QFormer (same forward(raw, semantic) -> [B, Q, lm_dim] and semantic_from_visual), no attention.

        LN(raw) ++ LN(semantic) -> Linear(768, W) -> D pre-LN residual blocks -> LN -> Linear(W, Q*lm_dim, no bias)
        -> + learned per-query embedding -> LayerNorm(lm_dim). head="single" emits one token repeated Q times. The last
        LayerNorm's gain/bias can start at the LM embedding scale/centroid (token_scale, token_center).
        """

        def __init__(self, image_dim=384, semantic_dim=384, lm_dim=1024, num_queries=32, *, width=704, depth=2, expansion=2,
                     dropout=0.05, head="tokens", align_hidden=512, token_scale: float | None = None, token_center=None):
            super().__init__()
            if head not in ("tokens", "single"):
                raise ValueError(head)
            self.image_dim, self.semantic_dim, self.lm_dim, self.num_queries = int(image_dim), int(semantic_dim), int(lm_dim), int(num_queries)
            self.width, self.depth, self.expansion, self.head_kind = int(width), int(depth), int(expansion), head
            self.dropout, self.align_hidden = float(dropout), int(align_hidden)
            self.norm_raw, self.norm_sem = nn.LayerNorm(image_dim), nn.LayerNorm(semantic_dim)
            self.in_proj = nn.Linear(image_dim + semantic_dim, width)
            self.blocks = nn.ModuleList([ResidualBlock(width, expansion, dropout) for _ in range(depth)])
            self.out_norm = nn.LayerNorm(width)
            self.head = nn.Linear(width, num_queries * lm_dim if head == "tokens" else lm_dim, bias=False)
            self.query_pos = nn.Parameter(torch.randn(1, num_queries, lm_dim) * 0.02)
            self.token_norm = nn.LayerNorm(lm_dim)
            with torch.no_grad():
                if token_scale is not None:
                    self.token_norm.weight.fill_(float(token_scale))
                if token_center is not None:
                    self.token_norm.bias.copy_(torch.as_tensor(token_center, dtype=self.token_norm.bias.dtype))
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

        def projector_config(self) -> dict:
            """Constructor arguments for build_projector('mlp', **cfg)."""
            return {"image_dim": self.image_dim, "semantic_dim": self.semantic_dim, "lm_dim": self.lm_dim, "num_queries": self.num_queries,
                    "width": self.width, "depth": self.depth, "expansion": self.expansion, "dropout": self.dropout,
                    "head": self.head_kind, "align_hidden": self.align_hidden}

    for cls in (ResidualBlock, ResidualMLPProjector):  # picklable as _common_models.<name>
        cls.__qualname__ = cls.__name__
        cls.__module__ = __name__
    return {"ResidualBlock": ResidualBlock, "ResidualMLPProjector": ResidualMLPProjector}


def __getattr__(name):  # _common_models.ResidualMLPProjector defines the classes on first access
    if name in ("ResidualBlock", "ResidualMLPProjector"):
        return _projector_classes()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def add_projector_args(parser: argparse.ArgumentParser, *, allow_auto: bool = False, mlp_shape: bool = True):
    """--projector (and --mlp-width/--mlp-depth) for the brain-to-text stages."""
    if allow_auto:
        parser.add_argument("--projector", choices=("auto",) + PROJECTOR_KINDS, default="mlp",
                            help="Projector to fine-tune: mlp (needs a stage-13 MLP package), qformer (deprecated Q-Former), "
                                 "or auto (whatever stage 13 packaged).")
    else:
        parser.add_argument("--projector", choices=PROJECTOR_KINDS, default="mlp",
                            help="Soft-prompt projector: mlp (residual MLP) or qformer (the deprecated Q-Former of the first submission).")
    if mlp_shape:
        parser.add_argument("--mlp-width", type=int, default=MLP_DEFAULTS["width"], help="MLP trunk width (704 = mlp_m, 192 = mlp_s).")
        parser.add_argument("--mlp-depth", type=int, default=MLP_DEFAULTS["depth"], help="MLP residual blocks.")


def build_projector(kind: str, *, image_dim: int = 384, semantic_dim: int = 384, lm_dim: int = 1024, num_queries: int = 32,
                    width: int = MLP_DEFAULTS["width"], depth: int = MLP_DEFAULTS["depth"], expansion: int = MLP_DEFAULTS["expansion"],
                    dropout: float = MLP_DEFAULTS["dropout"], head: str = MLP_DEFAULTS["head"], align_hidden: int = MLP_DEFAULTS["align_hidden"],
                    token_scale: float | None = None, token_center=None, **_ignored):
    """New fp32 CPU projector: the MLP, or the deprecated QFormer(384, 384, 1024, 32 queries, 512 hidden, 8 heads, 6 layers)."""
    if kind == "qformer":
        from neurovlm.models.qformer import QFormer

        return QFormer(image_dim=image_dim, semantic_dim=semantic_dim, lm_dim=lm_dim, num_queries=num_queries,
                       hidden_dim=512, num_heads=8, num_layers=6)
    if kind == "mlp":
        return _projector_classes()["ResidualMLPProjector"](
            image_dim, semantic_dim, lm_dim, num_queries, width=width, depth=depth, expansion=expansion, dropout=dropout,
            head=head, align_hidden=align_hidden, token_scale=token_scale, token_center=token_center)
    raise ValueError(f"unknown projector {kind!r}; choose from {PROJECTOR_KINDS}")


def projector_kind(module) -> str:
    """'mlp' for a ResidualMLPProjector (bare or inside a NeuroQFormer), else 'qformer'."""
    inner = getattr(module, "qformer", module)
    return "mlp" if type(inner).__name__ == "ResidualMLPProjector" else "qformer"


def projector_config(module) -> dict:
    """Architecture of a projector (what build_projector(kind, **cfg) needs) plus its parameter count."""
    inner = getattr(module, "qformer", module)
    if projector_kind(inner) == "mlp":
        cfg = {"class": "ResidualMLPProjector", **inner.projector_config()}
    else:
        cfg = {"class": "QFormer", "image_dim": int(inner.image_dim), "semantic_dim": int(inner.semantic_dim), "lm_dim": int(inner.lm_dim),
               "num_queries": int(inner.num_queries), "hidden_dim": int(inner.hidden_dim), "num_heads": int(inner.num_heads),
               "num_layers": int(inner.num_layers), "dropout": float(inner.dropout)}
    cfg["params"] = int(sum(p.numel() for p in inner.parameters()))
    return cfg


def mlp_config_from_state(state: Mapping) -> dict:
    """ResidualMLPProjector constructor arguments read off its state dict."""
    w_in, head, pos = state["in_proj.weight"], state["head.weight"], state["query_pos"]
    depth = len({k.split(".")[1] for k in state if k.startswith("blocks.")})
    width, lm_dim, num_queries = int(w_in.shape[0]), int(pos.shape[2]), int(pos.shape[1])
    return {"image_dim": int(state["norm_raw.weight"].shape[0]), "semantic_dim": int(state["norm_sem.weight"].shape[0]), "lm_dim": lm_dim,
            "num_queries": num_queries, "width": width, "depth": depth,
            "expansion": int(state["blocks.0.fc1.weight"].shape[0]) // width if depth else MLP_DEFAULTS["expansion"],
            "head": "tokens" if int(head.shape[0]) == num_queries * lm_dim else "single", "align_hidden": int(state["align_head.1.weight"].shape[0])}


def state_projector_kind(state: Mapping) -> str:
    """Projector kind of a bare state dict: in_proj.weight -> mlp, query_tokens -> qformer."""
    if "in_proj.weight" in state:
        return "mlp"
    if "query_tokens" in state:
        return "qformer"
    raise ValueError("state dict is neither a QFormer nor a ResidualMLPProjector")


def package_qformer_block(packaged) -> dict:
    """config.json["qformer"] of a packaged NeuroQFormer.

    For an MLP it has no hidden_dim/num_heads/num_layers on purpose, so the released loader (which ignores
    config["projector"]) fails instead of building a Q-Former.
    """
    if projector_kind(packaged) == "qformer":
        return packaged.architecture_config()
    inner = packaged.qformer
    return {"architecture": "NeuroQFormer", "projector": "mlp", **inner.projector_config(), "projection_temp": packaged.projection_temp,
            "canonical_basis": packaged.canonical_basis, "use_canonical_projection": packaged.use_canonical_projection,
            "projection_type": type(packaged.proj_head_image).__name__}


def neuro_projector_from_state(state: Mapping, config: Mapping, *, device="cpu"):
    """Packaged NeuroQFormer from its safetensors state and config.json, for either projector kind (strict loading)."""
    from neurovlm.models.qformer import CanonicalProjection, NeuroQFormer

    config = config or {}
    kind = config.get("projector", "qformer")
    block = dict(config.get("qformer", {}))
    if kind == "qformer":  # the released load path
        payload = {"state_dict": dict(state), "config": block, "canonical_metadata": config.get("canonical_metadata", {})}
        return NeuroQFormer.from_state_dict_payload(payload, map_location=device).eval()
    if kind != "mlp":
        raise ValueError(f"unknown projector {kind!r} in config.json; known: {PROJECTOR_KINDS}")
    import torch

    inner_state = {k[len("qformer."):]: v for k, v in state.items() if k.startswith("qformer.")}
    pcfg = dict(config.get("projector_config") or mlp_config_from_state(inner_state))
    projector = build_projector("mlp", **{k: v for k, v in pcfg.items() if k not in ("class", "params")})
    head = None
    if "proj_head_image.aligner.0.weight" in state:
        from neurovlm.models.base import ProjHead

        first, last = state["proj_head_image.aligner.0.weight"], state["proj_head_image.aligner.2.weight"]
        head = ProjHead(int(first.shape[1]), int(first.shape[0]), int(last.shape[0]))
    sem_dim = int(pcfg.get("semantic_dim", 384))
    banks = {n: state.get(f"canonical_projection.{n}_bank", torch.empty(0, sem_dim)) for n in CanonicalProjection.bank_names}
    model = NeuroQFormer(qformer=projector, proj_head_image=head, canonical_banks=banks, semantic_dim=int(projector.semantic_dim),
                         projection_temp=block.get("projection_temp", 0.05), canonical_basis=block.get("canonical_basis", "all"),
                         use_canonical_projection=block.get("use_canonical_projection", True))
    model.load_state_dict(dict(state), strict=True)
    return model.to(device).eval()


def package_projector_kind(package_dir: Path | str) -> str:
    """config.json["projector"] of a packaged neuro_qformer_*/ directory ('qformer' when absent)."""
    cfg_path = Path(package_dir) / "config.json"
    return read_json(cfg_path).get("projector", "qformer") if cfg_path.exists() else "qformer"


def load_projector(package_dir: Path | str, *, device="cpu"):
    """Load a packaged neuro_qformer_{pubmed,canonical}/ (model.safetensors + config.json) as a NeuroQFormer."""
    from safetensors.torch import load_file

    package_dir = Path(package_dir)
    config = read_json(package_dir / "config.json")
    return neuro_projector_from_state(load_file(str(package_dir / "model.safetensors")), config, device=device)


def write_projector_package(pkg_dir: Path, packaged, config: dict) -> None:
    """Write <pkg_dir>/model.safetensors and config.json (projector kind and config first, then config)."""
    from safetensors.torch import save_file

    pkg_dir = Path(pkg_dir)
    pkg_dir.mkdir(parents=True, exist_ok=True)
    save_file({k: v.detach().contiguous().cpu() for k, v in packaged.state_dict().items()}, str(pkg_dir / "model.safetensors"))
    write_json(pkg_dir / "config.json", {"projector": projector_kind(packaged), "projector_config": projector_config(packaged), **config})


# Caption script. The frozen multilingual NeuroQwen3 drifts into CJK, Cyrillic, Thai, ... tokens at low-margin steps when it
# reads soft prefix tokens instead of text (no training caption holds such characters). Generation therefore suppresses
# every token with a character outside ASCII, Latin-1, Greek and U+2000-2BFF (punctuation, arrows, math), and every
# byte-fragment token that is not valid UTF-8 on its own; special tokens are kept. model.generate reads
# lm.generation_config.suppress_tokens, so NeuroVLM.generate_text needs no change.
CAPTION_CHAR_RANGES = ((0x0000, 0x00FF), (0x0370, 0x03FF), (0x2000, 0x2BFF))


def caption_char_ok(ch: str) -> bool:
    """True if ``ch`` lies in CAPTION_CHAR_RANGES."""
    return any(a <= ord(ch) <= b for a, b in CAPTION_CHAR_RANGES)


@functools.cache
def latin_script_suppress_tokens(tokenizer) -> tuple[int, ...]:
    """Sorted ids of every non-special token whose text has a character outside CAPTION_CHAR_RANGES or is not valid
    UTF-8 alone. Use: ``lm.generation_config.suppress_tokens = list(latin_script_suppress_tokens(tok))``."""
    from transformers.models.gpt2.tokenization_gpt2 import bytes_to_unicode

    byte_of = {c: b for b, c in bytes_to_unicode().items()}  # byte-level BPE alphabet -> byte
    special = set(tokenizer.all_special_ids) | set(tokenizer.added_tokens_decoder)
    ids = []
    for i in range(len(tokenizer)):
        tok = tokenizer.convert_ids_to_tokens(i)
        if i in special or tok is None or any(c not in byte_of for c in tok):
            continue
        try:
            text = bytes(byte_of[c] for c in tok).decode("utf-8")
        except UnicodeDecodeError:  # byte fragment of a multi-byte character
            ids.append(i)
            continue
        if not all(caption_char_ok(c) for c in text):
            ids.append(i)
    return tuple(ids)


STEERING_TAG_RE = re.compile(r"(?i)[ \t]*\[(?:network|region|function)\][ \t]*")


def strip_steering_tags(text: str) -> str:
    """Drop the [NETWORK] / [REGION] / [FUNCTION] tags the LM sometimes writes into a caption, with the spaces around
    them (a tag between two words leaves one space); lines without a tag are left as they are."""
    return "\n".join(STEERING_TAG_RE.sub(" ", line).strip() if STEERING_TAG_RE.search(line) else line
                     for line in str(text).split("\n"))


# Title constraint (default for every caption). A caption is "[LEVEL]Title\nbody". After a prompt that ends with the
# [NETWORK] or [REGION] steering token, the title can only be one of the names of that level in the run's stage-14
# canonical bank (title_vocabulary: 48 network and 83 region names in runs/final): the names' token ids form a trie and a
# logits processor keeps every beam on a trie path until the newline that ends the title. When the prompt ends with no
# steering token (basis "all"), the model writes the tag itself: the title after a [NETWORK] / [REGION] tag it has just
# written is held on the same trie until its newline. The body and [FUNCTION] titles stay free, and so does all text of a
# prompt without a tag that never writes one (stage 1). NeuroVLM.generate_text passes no logits processor and
# lm.generation_config cannot hold one, so install_title_constraint wraps ``generate`` of the LM instance; the wrapper
# reads the steering token from the end of the prompt (the inputs_embeds rows equal to its token embeddings, or the
# input_ids). install_title_constraint(lm, tok, None) or a ``free_titles(lm)`` block decode free titles (--free-titles).
TITLE_LEVELS = ("network", "region")  # constrained levels; [FUNCTION] titles stay free
STEERING_LEVELS = ("network", "region", "function")  # steering tags; a written [FUNCTION] tag ends a constrained title
TITLE_END_TEXTS = ("\n", "\n\n")
TITLE_VOCABULARY_FILE = "title_vocabulary.json"  # next to canonical_semantic_banks.pt in the run dir
TITLE_MATCH_MIN_COS = 0.999


def _stage14_texts():
    """(levels, titles, SPECTER inputs) of the curated stage-14 texts before its self-consistency filter
    (14_qformer_anat.build_text_table on the pinned text_synth_less.parquet)."""
    import importlib.util

    spec = importlib.util.spec_from_file_location("_stage14_texts", Path(__file__).resolve().parent / "14_qformer_anat.py")
    s14 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(s14)
    df, texts = s14.build_text_table(hf_pinned("neurovlm/synthetic_pairs", "text_synth_less.parquet"))
    levels = df["text"].str.extract(r"^\[([A-Z]+)\]", expand=False).str.lower().to_numpy()
    return levels, df["title"].astype(str).to_numpy(), list(texts)


def bank_titles(banks: Mapping, text_head, specter, *, device="cpu", return_rows: bool = False):
    """{level: title of every row of banks[level]}. A stage-14 bank row is the projected SPECTER2 embedding of one curated
    text, normalize(text_head(normalize(specter("Title [SEP] body")))), so each row is matched to its text; a best match
    below TITLE_MATCH_MIN_COS raises ValueError (the bank was not built from these texts with this head).
    ``return_rows``: also {level: index of each row's text in _stage14_texts()}."""
    import torch
    import torch.nn.functional as F

    levels, titles, texts = _stage14_texts()
    with torch.no_grad():
        txt = torch.cat([F.normalize(specter(texts[s:s + 256]), dim=1).float() for s in range(0, len(texts), 256)])
        sem = F.normalize(text_head.to(device).eval()(txt.to(device)).float(), dim=1).cpu()
    out, rows = {}, {}
    for lv in ("network", "region", "function"):
        idx = np.flatnonzero(levels == lv)
        best = (F.normalize(torch.as_tensor(banks[lv]).float().cpu(), dim=1) @ sem[idx].T).max(1)
        if float(best.values.min()) < TITLE_MATCH_MIN_COS:
            raise ValueError(f"{lv} bank: a row matches no curated stage-14 text (best cosine {float(best.values.min()):.4f})")
        rows[lv] = idx[best.indices.numpy()]
        out[lv] = titles[rows[lv]].tolist()
    return (out, rows) if return_rows else out


def title_vocabulary(run_dir: Path | None, *, device="cpu", logger=None, specter=None, text_head=None) -> dict:
    """Canonical title vocabulary of a run's stage-14 banks: {"titles": {level: sorted names}, "bank_rows": {level: name
    of each bank row}, "source": {...}}. Built once (SPECTER2 + the run's proj_head_text_infonce on the 1,338 curated
    texts, ~1 min) and cached as <run_dir>/title_vocabulary.json; rebuilt when canonical_semantic_banks.pt changes.
    ``run_dir`` None or without banks: the released NeuroQformer's banks and text head, cached as
    scripts/outputs/released_title_vocabulary_<rev>.json."""
    import torch

    banks_path = Path(run_dir) / "canonical_semantic_banks.pt" if run_dir is not None else None
    if banks_path is not None and banks_path.exists():
        key = {"banks": banks_path.name, "banks_sha256": sha256_file(banks_path)}
        cache = Path(run_dir) / TITLE_VOCABULARY_FILE
    else:
        rev = hf_pinned("neurovlm/NeuroQformer")
        key = {"banks": f"released:neurovlm/NeuroQformer@{rev[:8]}", "banks_sha256": None}
        cache, run_dir = DEFAULT_OUT_ROOT / f"released_title_vocabulary_{rev[:8]}.json", None
    if cache.exists():
        vocab = read_json(cache)
        if {k: vocab.get("source", {}).get(k) for k in key} == key:
            return vocab
    if run_dir is not None:
        banks = torch.load(banks_path, map_location="cpu", weights_only=False)
        head_path = Path(run_dir) / "proj_head_text_infonce.pt"
        head_source = head_path.name if head_path.exists() else "released:proj_head_text_infonce"
    else:
        buffers = dict(released_qformer("canonical").named_buffers())
        banks = {lv: next(v for k, v in buffers.items() if k.endswith(f"{lv}_bank")) for lv in ("network", "region", "function")}
        head_path, head_source = None, "released:proj_head_text_infonce"
    if text_head is None:
        if head_path is not None and head_path.exists():
            from neurovlm.models import ProjHead

            text_head = load_module_or_state(head_path, builder=lambda: ProjHead(768, 512, 384), device=device)
        else:
            text_head = released_model("proj_head_text_infonce", device)
    t0 = time.perf_counter()
    rows = bank_titles(banks, text_head, specter if specter is not None else released_model("specter", device), device=device)
    vocab = {"source": {**key, "text_head": head_source, "texts": "14_qformer_anat.build_text_table(neurovlm/synthetic_pairs "
                        f"text_synth_less.parquet@{hf_pinned('neurovlm/synthetic_pairs')[:8]})", "min_cos": TITLE_MATCH_MIN_COS},
             "titles": {lv: sorted(set(r)) for lv, r in rows.items()}, "bank_rows": rows}
    write_json(cache, vocab)
    if logger is not None:
        logger.info("title vocabulary %s: %s titles (%.0fs)", cache, {lv: len(v) for lv, v in vocab["titles"].items()},
                    time.perf_counter() - t0)
    return vocab


def caption_title(text: str) -> str:
    """Title of a "[LEVEL]Title\\nbody" caption: its first line without the steering tag."""
    return strip_steering_tags(str(text).split("\n", 1)[0]).strip()


def noncanonical_titles(texts: Sequence[str], names: Sequence[str]) -> list[bool]:
    """Per caption: True when its title is none of ``names`` (case and spacing ignored)."""
    def key(t):
        return " ".join(str(t).casefold().split())

    allowed = {key(n) for n in names}
    return [key(caption_title(t)) not in allowed for t in texts]


def title_trie(tokenizer, names) -> dict:
    """Nested {token id: subtrie} of ``names``, tokenised alone (as after the [LEVEL] tag); key -1 ends a name."""
    root: dict = {}
    for name in names:
        node = root
        for t in tokenizer(str(name), add_special_tokens=False)["input_ids"]:
            node = node.setdefault(int(t), {})
        node[-1] = {}
    return root


@functools.cache
def newline_token_ids(tokenizer) -> frozenset:
    """Ids of every token whose text holds a newline (any of them ends a free title)."""
    return frozenset(i for i in range(len(tokenizer)) if "\n" in tokenizer.decode([i]))


@functools.cache
def open_bracket_token_ids(tokenizer) -> frozenset:
    """Ids of every token whose text ends with '[' (the first token of a steering tag the model writes itself)."""
    return frozenset(i for i in range(len(tokenizer)) if tokenizer.decode([i]).endswith("["))


class TitleConstraint:
    """Logits processor for ``model.generate``: row r (beam-major, as HF orders beams) keeps its title on
    ``tries[r // num_beams]`` until a newline ends it; ``None`` leaves that map's title free. With ``written`` ({level:
    (tag token ids after its '[', trie or None)}), a row whose prompt ends with no steering tag (``tagged[r // num_beams]``
    False, e.g. basis "all") also keeps the title after the last tag it wrote itself on that level's trie, until a
    newline ends it (None: [FUNCTION], free), and closes such a tag with its own ']' token (not a merged ']]', '][' or
    ']\n'). ``offset``: prompt tokens at the start of input_ids (0 with inputs_embeds). Text after a title is never
    changed."""

    def __init__(self, tokenizer, tries, num_beams: int = 1, *, written=None, tagged=None, offset: int = 0):
        self.tries, self.num_beams = list(tries), int(num_beams)
        self.end_ids = [tokenizer(t, add_special_tokens=False)["input_ids"][0] for t in TITLE_END_TEXTS]
        self.newline = newline_token_ids(tokenizer)
        self.written, self.offset = written or {}, int(offset)
        self.tagged = [t is not None for t in self.tries] if tagged is None else [bool(t) for t in tagged]
        if self.written:
            self.open, self._ids = open_bracket_token_ids(tokenizer), {}

    def _ids_on(self, name, ids, device):
        import torch

        if name not in self._ids or self._ids[name].device != device:
            self._ids[name] = torch.as_tensor(sorted(ids), dtype=torch.long, device=device)
        return self._ids[name]

    def _written_titles(self, input_ids, rows) -> dict:
        """{row: trie node} for the rows (of ``rows``) inside a title after a tag they wrote (the last tag of the
        generated tokens is a [NETWORK] / [REGION] tag and no newline follows it) or inside such a tag (the tokens end
        with '[' and the level's name: the node allows only the tag's closing token)."""
        import torch

        x = input_ids[rows][:, self.offset:]
        n, L = x.shape
        pos = torch.arange(L, device=x.device).expand(n, L)
        nl = torch.where(torch.isin(x, self._ids_on("newline", self.newline, x.device)), pos, -1).amax(1)
        best, level = torch.full((n,), -1, device=x.device), [None] * n
        for lv, (tag, _) in self.written.items():
            k = len(tag)
            if L <= k:
                continue
            hit = torch.isin(x[:, :L - k], self._ids_on("open", self.open, x.device))  # '[' at s, tag ids at s+1..s+k
            for j, t in enumerate(tag):
                hit &= x[:, 1 + j:L - k + 1 + j] == t
            end = torch.where(hit, pos[:, :L - k] + k, -1).amax(1)
            for i in torch.nonzero(end > best).flatten().tolist():
                level[i] = lv
            best = torch.maximum(best, end)
        out = {}
        for i in torch.nonzero(best > nl).flatten().tolist():
            node = self.written[level[i]][1]
            for t in x[i, int(best[i]) + 1:].tolist():
                if node is None:
                    break
                node = node.get(t)
            if node is not None:
                out[rows[i]] = node
        for lv, (tag, trie) in self.written.items():
            k = len(tag)
            if trie is None or L < k:
                continue
            open_tag = torch.isin(x[:, L - k], self._ids_on("open", self.open, x.device))
            for j, t in enumerate(tag[:-1]):
                open_tag &= x[:, L - k + 1 + j] == t
            for i in torch.nonzero(open_tag).flatten().tolist():
                out[rows[i]] = {tag[-1]: {}}
        return out

    def __call__(self, input_ids, scores):
        import torch

        nodes = {}
        for r, seq in enumerate(input_ids[:, self.offset:].tolist()):  # generated tokens only (prompt ids skipped)
            if not self.tagged[r // self.num_beams]:
                continue
            node = self.tries[r // self.num_beams]
            for t in seq:
                if node is None or t in self.newline:
                    node = None
                    break
                node = node.get(t)
            if node is not None:
                nodes[r] = node
        if self.written:
            free = [r for r in range(input_ids.shape[0]) if not self.tagged[r // self.num_beams]]
            if free and input_ids.shape[1] > self.offset:
                nodes.update(self._written_titles(input_ids, free))
        for r, node in nodes.items():
            allowed = [t for t in node if t >= 0] + (self.end_ids if -1 in node else [])
            keep = scores[r, allowed]
            scores[r] = float("-inf")
            # a name token already banned by another processor (repetition) stays possible, at a large cost
            scores[r, allowed] = torch.where(torch.isinf(keep), torch.full_like(keep, -1e4), keep)
        return scores


def _prompt_levels(lm, tags: Mapping[str, Sequence[int]], args, kwargs) -> list:
    """Per prompt row: the level whose steering tag ends the prompt (its token embeddings as the last inputs_embeds rows,
    or its token ids as the last input_ids), else None."""
    import torch

    emb = kwargs.get("inputs_embeds")
    ids = kwargs.get("input_ids", args[0] if args else None)
    x = emb if emb is not None else ids
    if x is None:
        return []
    levels = [None] * x.shape[0]
    for lv, tag in tags.items():
        k = len(tag)
        if x.shape[1] < k:
            continue
        if emb is not None:
            table = lm.get_input_embeddings().weight
            ref = table[torch.as_tensor(tag, device=table.device)].to(device=emb.device, dtype=emb.dtype)
            hit = (emb[:, -k:, :] == ref).flatten(1).all(1)
        else:
            hit = (ids[:, -k:] == torch.as_tensor(tag, device=ids.device)).all(1)
        for r in torch.nonzero(hit).flatten().tolist():
            levels[r] = lv
    return levels


class TitleConstrainedGenerate:
    """``lm.generate`` plus a TitleConstraint (set by install_title_constraint) for the rows whose prompt ends with a
    constrained steering tag and for the tags written after a prompt without one. A caller's own TitleConstraint
    (constrained_titles) replaces the default."""

    def __init__(self, lm, generate):
        self.lm, self.generate = lm, generate

    def __call__(self, *args, **kwargs):
        spec = self.lm.__dict__.get("_title_spec")
        procs = list(kwargs.get("logits_processor") or [])
        if spec is not None and not any(type(p).__name__ == "TitleConstraint" for p in procs):
            levels = _prompt_levels(self.lm, spec["tags"], args, kwargs)  # [FUNCTION] prompts: tagged, free title
            tries = [spec["tries"].get(lv) for lv in levels]
            if any(t is not None for t in tries) or not all(lv is not None for lv in levels):
                from transformers import LogitsProcessorList

                ids = kwargs.get("input_ids", args[0] if args else None)
                beams = int(kwargs.get("num_beams") or getattr(self.lm.generation_config, "num_beams", 1) or 1)
                kwargs["logits_processor"] = LogitsProcessorList([*procs, TitleConstraint(
                    spec["tokenizer"], tries, beams, written=spec["written"], tagged=[lv is not None for lv in levels],
                    offset=0 if ids is None else ids.shape[1])])
        return self.generate(*args, **kwargs)


def install_title_constraint(lm, tokenizer, titles: Mapping[str, Sequence[str]] | None) -> None:
    """Default caption decoding: the title after a [NETWORK] / [REGION] steering token, in the prompt or written by the
    model, is one of ``titles[level]`` (title_vocabulary(...)["titles"]); None = free titles (--free-titles). Wraps
    ``generate`` of this LM instance once; a second call only changes the names."""
    spec = None
    if titles is not None:
        names = {lv: sorted(set(map(str, titles[lv]))) for lv in TITLE_LEVELS}
        spec = {"tokenizer": tokenizer, "n": {lv: len(v) for lv, v in names.items()},
                "tags": {lv: list(tokenizer(f"[{lv.upper()}]", add_special_tokens=False)["input_ids"]) for lv in STEERING_LEVELS},
                "tries": {lv: title_trie(tokenizer, v) for lv, v in names.items()}}
        # a tag the model writes: any token ending in '[' and then the tag's other tokens ("[", "NETWORK", "]")
        assert all(tokenizer.decode(tag[:1]) == "[" for tag in spec["tags"].values()), spec["tags"]
        spec["written"] = {lv: (tag[1:], spec["tries"].get(lv)) for lv, tag in spec["tags"].items()}
    lm.__dict__["_title_spec"] = spec
    if not isinstance(lm.__dict__.get("generate"), TitleConstrainedGenerate):
        lm.__dict__["generate"] = TitleConstrainedGenerate(lm, lm.generate)


def title_constraint_note(lm) -> str:
    """'titles: canonical vocabulary (48 network / 83 region names)' or 'titles: free'."""
    spec = lm.__dict__.get("_title_spec")
    return "titles: free" if spec is None else "titles: canonical vocabulary (" + " / ".join(f"{n} {lv}" for lv, n in spec["n"].items()) + " names)"


@contextlib.contextmanager
def free_titles(lm):
    """Inside the block the LM decodes free titles (as --free-titles)."""
    spec = lm.__dict__.get("_title_spec")
    lm.__dict__["_title_spec"] = None
    try:
        yield
    finally:
        lm.__dict__["_title_spec"] = spec


@contextlib.contextmanager
def constrained_titles(lm, processor):
    """Inside the block ``lm.generate`` also applies ``processor`` (e.g. a TitleConstraint with other names per map,
    which then replaces the default constraint)."""
    from transformers import LogitsProcessorList

    own = lm.__dict__.get("generate")
    lm.generate = functools.partial(lm.generate, logits_processor=LogitsProcessorList([processor]))
    try:
        yield
    finally:
        if own is None:
            del lm.generate
        else:
            lm.generate = own
