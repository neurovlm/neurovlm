"""Helpers for the loss ablation: PubMed data, eval exclusions, the 09 projection-head recipes and their metrics.

Each a5_* directory has its own copy; only the installed ``neurovlm`` package and its cached data are used.
"""
from __future__ import annotations

import contextlib
import json
import os
import random
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


def set_offline_env() -> None:
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("MPLBACKEND", "Agg")


def seed_all(seed: int) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def select_device(name: str = "auto"):
    import torch

    if name != "auto":
        return torch.device(name)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def data_dir() -> Path:
    from neurovlm.data import data_dir as _dd

    return Path(_dd)


def load_torch(path: Path | str, *, mmap: bool = True):
    import torch

    try:
        return torch.load(Path(path), map_location="cpu", weights_only=False, mmap=mmap)
    except (RuntimeError, ValueError, TypeError):
        return torch.load(Path(path), map_location="cpu", weights_only=False)


def write_json(path: Path | str, obj: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_safe(obj), indent=2, sort_keys=True))


def read_json(path: Path | str):
    return json.loads(Path(path).read_text())


def json_safe(value: Any) -> Any:
    import torch

    if isinstance(value, Mapping):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        v = float(value)
        return v if np.isfinite(v) else None
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if torch.is_tensor(value):
        return json_safe(value.detach().cpu().tolist())
    return value


class Timings:
    """Named wall-clock sections saved to ``timings.json`` (merged with earlier runs)."""

    def __init__(self, path: Path | str):
        self.path = Path(path)
        self.rows: dict[str, dict] = read_json(self.path) if self.path.exists() else {}

    @contextlib.contextmanager
    def time(self, name: str, **meta):
        t0 = time.perf_counter()
        row = dict(meta)
        try:
            yield row
        finally:
            row["seconds"] = time.perf_counter() - t0
            row["ts"] = time.strftime("%Y-%m-%dT%H:%M:%S")
            self.rows[name] = row
            write_json(self.path, self.rows)


def get_logger(name: str):
    import logging

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s", datefmt="%H:%M:%S")
    return logging.getLogger(name)


# PubMed data
def load_pubmed_latents():
    """Autoencoder latents [N, 384] + pmids of the selected model (_final_model: the run's latent_neuro.pt, else released)."""
    from _final_model import pubmed_latents

    return pubmed_latents()


def load_specter_latents(pmids: np.ndarray):
    """SPECTER2 (adhoc_query) embeddings [N, 768] aligned to ``pmids``."""
    import torch

    local = data_dir() / "latent_specter2_adhoc.pt"
    if local.exists():
        payload = load_torch(local)
        lat, pm = payload["latent"].float(), np.asarray(payload["pmid"])
    else:
        from _hf_pins import load_latent

        lat, pm = load_latent("pubmed_text")
        lat, pm = torch.as_tensor(lat).float().cpu(), np.asarray(pm)
    pos = {p: i for i, p in enumerate(pm)}
    idx = torch.as_tensor([pos[p] for p in pmids], dtype=torch.long)
    return lat[idx].clone()


def split_masks(pmids: np.ndarray) -> dict[str, np.ndarray]:
    """Train/val/test masks from the selected model's ``pmids_split.pt``."""
    import pandas as pd
    from _final_model import split_pmids

    split = split_pmids()
    pm = pd.Series(pmids)
    return {k: pm.isin(np.asarray(split[k])).to_numpy() for k in ("train", "val", "test")}


def subsample_masks(masks: Mapping[str, np.ndarray], max_samples: int | None,
                    fractions: Mapping[str, float] = {"train": 0.75, "val": 0.125, "test": 0.125}) -> dict[str, np.ndarray]:
    """Keep a split-proportional prefix of each split (smoke runs)."""
    if not max_samples:
        return dict(masks)
    out = {}
    for name, m in masks.items():
        k = max(2, int(round(max_samples * fractions.get(name, 1 / len(masks)))))
        keep = np.zeros_like(m)
        keep[np.flatnonzero(m)[:k]] = True
        out[name] = keep
    return out


def load_pubmed_maps(pmids: np.ndarray):
    """Raw flat maps [N, 28542] (fp32, CPU) for ``pmids``."""
    import torch

    nv = data_dir() / "neuro_vectors.pt"
    if nv.exists():
        payload = load_torch(nv, mmap=True)
        vectors, pm = payload["neuro_vectors"], np.asarray(payload["pmid"])
    else:
        from _hf_pins import load_dataset

        vectors, pm = load_dataset("pubmed_images")
        vectors, pm = torch.as_tensor(vectors), np.asarray(pm)
    pos = {p: i for i, p in enumerate(pm)}
    idx = torch.as_tensor([pos[p] for p in pmids], dtype=torch.long)
    return vectors[idx].float().clone()


# evaluation exclusions: held-out PubMed papers with a near-duplicate in the training split (16 test + 26 val PMIDs),
# written by a4_leakage_audit/run_analysis.py --write-exclusions
EVAL_EXCLUSIONS_DIR = Path(__file__).resolve().parents[5] / "scripts" / "eval_exclusions"


def pubmed_keep_mask(pmids, *, enabled: bool = True, label: str = "PubMed", log=print) -> np.ndarray:
    """False for the excluded PMIDs; all True when ``enabled`` is False."""
    import pandas as pd

    pmids = np.asarray(pmids)
    if not enabled or len(pmids) == 0:
        return np.ones(len(pmids), dtype=bool)
    excl = {int(x) for x in pd.read_csv(EVAL_EXCLUSIONS_DIR / "pubmed_exclusions.csv")["pmid"]}
    keep = ~pd.to_numeric(pd.Series(pmids.ravel()), errors="coerce").astype("Int64").isin(excl).fillna(False).to_numpy(dtype=bool)
    if log is not None:
        log(f"eval exclusions, {label}: dropped {int((~keep).sum())} of {len(keep)} PMIDs")
    return keep


def exclude_heldout_rows(masks: Mapping[str, np.ndarray], pmids, *, enabled: bool = True, splits: Sequence[str] = ("val", "test"),
                         log=print) -> tuple[dict[str, np.ndarray], dict]:
    """Drop the excluded PMIDs from the val/test masks (train untouched). Returns (masks, the "exclusions" record)."""
    pmids = np.asarray(pmids)
    out = {k: np.asarray(v, dtype=bool).copy() for k, v in masks.items()}
    block: dict[str, Any] = {"enabled": bool(enabled), "source": "scripts/eval_exclusions/pubmed_exclusions.csv"}
    if not enabled and log is not None:
        log("eval exclusions disabled (--no-exclusions)")
    for s in splits:
        rows = np.flatnonzero(out[s])
        keep = pubmed_keep_mask(pmids[rows], enabled=enabled, label=f"PubMed {s}", log=log)
        out[s][rows[~keep]] = False
        block[f"pubmed_{s}_dropped"] = int((~keep).sum())
        block[f"pubmed_{s}_kept"] = int(keep.sum())
    return out, block


def same_exclusions(prev: Mapping | None, enabled: bool) -> bool:
    """True when earlier outputs used the same exclusions setting (no record = disabled)."""
    return bool(((prev or {}).get("exclusions") or {}).get("enabled", False)) == bool(enabled)


# metrics
def deranged_indices(n: int, rng: np.random.Generator) -> np.ndarray:
    """Permutation without fixed points (null pairing)."""
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


def retrieval_block(z_text, z_img) -> dict:
    """Bidirectional recall-curve AUC / recall@k for L2-normalised embeddings."""
    from neurovlm.metrics.retrieval import bidirectional_retrieval_metrics, recall_curve

    m = bidirectional_retrieval_metrics(z_text, z_img)
    out = {k: float(v) for k, v in m.items() if not k.startswith("mean_random")}
    t2i, i2t = recall_curve(z_text, z_img, step=1)
    n = len(t2i)
    idx = np.unique(np.linspace(0, n - 1, min(n, 101)).astype(int))
    out["curve"] = {"normalized_k": ((idx + 1) / n).tolist(), "t2i": t2i[idx].tolist(), "i2t": i2t[idx].tolist()}
    out["n"] = int(n)
    return out


def shared_text(text_head, emb, device, batch_size: int = 1024):
    """Text embeddings -> shared InfoNCE space (input L2-normalised first)."""
    import torch
    import torch.nn.functional as F

    out = []
    with torch.no_grad():
        for s in range(0, len(emb), batch_size):
            x = F.normalize(emb[s:s + batch_size].float().to(device), dim=1)
            out.append(F.normalize(text_head(x).float(), dim=1).cpu())
    return torch.cat(out, dim=0)


def shared_brain(image_head, latents, device, batch_size: int = 1024):
    import torch
    import torch.nn.functional as F

    out = []
    with torch.no_grad():
        for s in range(0, len(latents), batch_size):
            out.append(F.normalize(image_head(latents[s:s + batch_size].float().to(device)).float(), dim=1).cpu())
    return torch.cat(out, dim=0)


def encode_latents(encoder, X, device, batch_size: int = 512):
    import torch

    out = []
    with torch.no_grad():
        for s in range(0, len(X), batch_size):
            out.append(encoder(X[s:s + batch_size].float().to(device)).detach().float().cpu())
    return torch.cat(out, dim=0)


def text_to_brain_metrics(pred_maps, true_maps, raw_maps=None, *, encoder=None, image_head=None, device=None, pct: int = 90,
                          null_seed: int = 13) -> dict:
    """Dice (percentile-thresholded) + voxel Pearson, matched vs deranged null; optional exact map retrieval AUC."""
    import torch
    from neurovlm.metrics.semantic import exact_pmid_retrieval_metrics
    from neurovlm.metrics.text_to_brain import dice_percentile

    pred_np, true_np = np.asarray(pred_maps, dtype=np.float32), np.asarray(true_maps, dtype=np.float32)
    n = len(pred_np)
    rng = np.random.default_rng(null_seed)
    null = deranged_indices(n, rng)
    out: dict[str, Any] = {"n": n, "dice_pct": pct}

    def zrows(x):
        x = x - x.mean(axis=1, keepdims=True)
        return x / np.clip(x.std(axis=1, keepdims=True), 1e-6, None)

    pz, tz = zrows(pred_np), zrows(true_np)
    d = float(pz.shape[1])
    out["pearson_matched"] = float(np.mean(np.einsum("ij,ij->i", pz, tz) / d))
    out["pearson_null"] = float(np.mean(np.einsum("ij,ij->i", pz, tz[null]) / d))
    out["dice_matched"] = float(np.mean([dice_percentile(pred_np[i], true_np[i], pct=pct) for i in range(n)]))
    out["dice_null"] = float(np.mean([dice_percentile(pred_np[i], true_np[null[i]], pct=pct) for i in range(n)]))
    if raw_maps is not None:
        raw_np = np.asarray(raw_maps, dtype=np.float32)
        rz = zrows(raw_np)
        out["pearson_vs_raw_matched"] = float(np.mean(np.einsum("ij,ij->i", pz, rz) / d))
        out["pearson_vs_raw_null"] = float(np.mean(np.einsum("ij,ij->i", pz, rz[null]) / d))
        out["dice_vs_raw_matched"] = float(np.mean([dice_percentile(pred_np[i], raw_np[i], pct=pct) for i in range(n)]))
        out["dice_vs_raw_null"] = float(np.mean([dice_percentile(pred_np[i], raw_np[null[i]], pct=pct) for i in range(n)]))
    if encoder is not None and image_head is not None and n >= 2:
        z_pred = shared_brain(image_head, encode_latents(encoder, torch.as_tensor(pred_np), device), device)
        z_true = shared_brain(image_head, encode_latents(encoder, torch.as_tensor(true_np), device), device)
        ret = exact_pmid_retrieval_metrics(z_pred @ z_true.T)
        out["map_recall_auc"] = float(ret["normalized_k_recall_curve_auc"])
        out["map_recall@10"] = float(ret.get("recall@10", float("nan")))
    return out


# 09 projection-head recipes with pluggable losses
def train_text_to_brain_head(text_tr, maps_tr, text_va, maps_va, *, decoder, in_dim: int, device, epochs: int = 201, batch_size: int = 1024,
                             lr: float = 5e-5, seed: int = 123, loss: str = "decoder_bce", latent_tr=None, latent_va=None,
                             val_interval: int = 20, max_batches: int | None = None, log=None) -> tuple[Any, dict]:
    """Stage A of 09: ProjHead(in_dim, 512, 384) trained through the frozen decoder; keeps the best validation state.

    ``loss``: decoder_bce (paper, BCE on the decoded map), latent_mse or latent_cosine (on the 384-d latent).
    """
    import copy
    import torch
    import torch.nn.functional as F
    from torch import nn
    from neurovlm.models import ProjHead

    head = ProjHead(seed=seed, latent_in_dim=in_dim, hidden_dim=512, latent_out_dim=384).to(device)
    decoder = copy.deepcopy(decoder).to(device).eval()
    for p in decoder.parameters():
        p.requires_grad_(False)
    bce = nn.BCEWithLogitsLoss()
    X_tr, X_va = text_tr.float().to(device), text_va.float().to(device)
    Y_tr, Y_va = maps_tr.float(), maps_va.float()
    L_tr = latent_tr.float().to(device) if latent_tr is not None else None
    L_va = latent_va.float().to(device) if latent_va is not None else None
    on_device = device.type == "cuda" and (Y_tr.numel() + Y_va.numel()) * 4 < 0.5 * torch.cuda.mem_get_info(device)[0]
    if on_device:
        Y_tr, Y_va = Y_tr.to(device), Y_va.to(device)

    def step_loss(z, idx, Y, L):
        if loss == "decoder_bce":
            y = Y[idx] if on_device else Y[idx.cpu()].to(device)
            return bce(decoder(z).float(), y)
        if loss == "latent_mse":
            return F.mse_loss(z, L[idx])
        if loss == "latent_cosine":
            return (1.0 - F.cosine_similarity(z, L[idx], dim=1)).mean()
        raise ValueError(loss)

    def val_losses():
        head.eval()
        tot = {"objective": 0.0, "decoder_bce": 0.0}
        with torch.no_grad():
            for s in range(0, len(X_va), 1024):
                idx = torch.arange(s, min(s + 1024, len(X_va)), device=device)
                z = head(X_va[idx])
                y = Y_va[idx] if on_device else Y_va[idx.cpu()].to(device)
                tot["decoder_bce"] += float(bce(decoder(z).float(), y)) * len(idx)
                tot["objective"] += float(step_loss(z, idx, Y_va, L_va)) * len(idx)
        return {k: v / len(X_va) for k, v in tot.items()}

    optimizer = torch.optim.AdamW(head.parameters(), lr=lr)
    gen = torch.Generator(device="cpu")
    best, best_state, curve, n = float("inf"), None, [], len(X_tr)
    t0 = time.perf_counter()
    steps = 0
    for epoch in range(epochs):
        head.train()
        gen.manual_seed(epoch)
        perm = torch.randperm(n, generator=gen).to(device)
        for bi, start in enumerate(range(0, n, batch_size)):
            if max_batches is not None and bi >= max_batches:
                break
            idx = perm[start:start + batch_size]
            l = step_loss(head(X_tr[idx]), idx, Y_tr, L_tr)
            optimizer.zero_grad(set_to_none=True)
            l.backward()
            optimizer.step()
            steps += 1
        if epoch % val_interval == 0 or epoch == epochs - 1:
            vl = val_losses()
            curve.append({"epoch": epoch, **vl})
            if log:
                log.info("t2b[%s] epoch %d val objective %.5g decoder_bce %.5g", loss, epoch, vl["objective"], vl["decoder_bce"])
            if vl["objective"] < best:
                best = vl["objective"]
                best_state = {k: v.detach().cpu().clone() for k, v in head.state_dict().items()}
    seconds = time.perf_counter() - t0
    if best_state is not None:
        head.load_state_dict(best_state)
    return head.cpu().eval(), {"loss": loss, "best_val_objective": best, "val_curve": curve, "epochs": epochs, "steps": steps,
                               "seconds": seconds, "sec_per_epoch": seconds / max(epochs, 1), "sec_per_step": seconds / max(steps, 1)}


def train_infonce_heads(text_tr, img_tr, text_va, img_va, *, text_head_init, in_dim: int, device, loss_fn, epochs: int = 301,
                        batch_size: int = 2048, lr: float = 1e-5, seed: int = 123, val_interval: int = 20, max_batches: int | None = None,
                        log=None) -> tuple[Any, Any, dict]:
    """Stage B of 09: text head (from stage A, or new when ``text_head_init`` is None) + image head; returns the final weights.

    ``loss_fn(text_proj, image_proj)``; inputs are L2-normalised text embeddings and raw 384-d brain latents.
    """
    import copy
    import torch
    import torch.nn.functional as F
    from neurovlm.models import ProjHead

    loss_fn = loss_fn.to(device)
    text_head = (copy.deepcopy(text_head_init) if text_head_init is not None
                 else ProjHead(seed=seed, latent_in_dim=in_dim, hidden_dim=512, latent_out_dim=384)).to(device)
    image_head = ProjHead(seed=seed, latent_in_dim=384, hidden_dim=384, latent_out_dim=384).to(device)
    X_tr, X_va = F.normalize(text_tr.float().to(device), dim=1, eps=1e-8), F.normalize(text_va.float().to(device), dim=1, eps=1e-8)
    I_tr, I_va = img_tr.float().to(device), img_va.float().to(device)
    params = [*text_head.parameters(), *image_head.parameters(), *(p for p in loss_fn.parameters() if p.requires_grad)]
    optimizer = torch.optim.AdamW(params, lr=lr)
    curve, n, steps = [], len(X_tr), 0
    t0 = time.perf_counter()
    for epoch in range(epochs):
        text_head.train()
        image_head.train()
        torch.manual_seed(epoch)
        perm = torch.randperm(n, device=device)
        for bi, start in enumerate(range(0, n, batch_size)):
            if max_batches is not None and bi >= max_batches:
                break
            idx = perm[start:start + batch_size]
            if len(idx) < 2:
                continue
            l = loss_fn(text_head(X_tr[idx]), image_head(I_tr[idx]))
            optimizer.zero_grad(set_to_none=True)
            l.backward()
            optimizer.step()
            steps += 1
        if epoch % val_interval == 0 or epoch == epochs - 1:
            text_head.eval()
            image_head.eval()
            with torch.no_grad():
                vl = float(loss_fn(text_head(X_va), image_head(I_va)))
            curve.append({"epoch": epoch, "val_loss": vl})
            if log:
                log.info("infonce epoch %d val loss %.5g", epoch, vl)
    seconds = time.perf_counter() - t0
    return text_head.cpu().eval(), image_head.cpu().eval(), {"val_curve": curve, "epochs": epochs, "steps": steps, "seconds": seconds,
                                                             "sec_per_epoch": seconds / max(epochs, 1), "sec_per_step": seconds / max(steps, 1),
                                                             "final_temperature": float(loss_fn.temperature_value())}


def eval_heads_retrieval(text_head, image_head, text_te, img_te, device) -> dict:
    """Paper heads metric: bidirectional recall-curve AUC on the test split."""
    z_text = shared_text(text_head.to(device), text_te, device)
    z_img = shared_brain(image_head.to(device), img_te, device)
    return retrieval_block(z_text, z_img)


def eval_text_to_brain_head(head, autoencoder, text_te, X_te, device, *, text_va=None, image_head=None, pct: int = 90,
                            batch_size: int = 512) -> dict:
    """Dice/Pearson of the calibrated map sigmoid(decoder(head(text))) vs sigmoid(decoder(encoder(X))) and vs raw X. The global
    calibration factor is this head's own: median over its validation maps (``text_va``) of 1 / map peak."""
    import torch
    from _final_model import MAP_CALIBRATION, calibrate_maps

    head = head.to(device).eval()
    ae = autoencoder.to(device).eval()

    def generate(t):
        return torch.sigmoid(ae.decoder(head(t.float().to(device)))).float()

    factor = None
    with torch.no_grad():
        if MAP_CALIBRATION == "global":
            peaks = torch.cat([generate(text_va[s:s + batch_size]).amax(dim=1).cpu() for s in range(0, len(text_va), batch_size)])
            factor = float(np.median(1.0 / np.maximum(peaks.double().numpy(), 1e-8)))
    pred, true = [], []
    with torch.no_grad():
        for s in range(0, len(text_te), batch_size):
            pred.append(calibrate_maps(generate(text_te[s:s + batch_size]), factor=factor).cpu())
            true.append(torch.sigmoid(ae.decoder(ae.encoder(X_te[s:s + batch_size].float().to(device)))).float().cpu())
    pred, true = torch.cat(pred), torch.cat(true)
    return {"map_calibration_factor": factor,
            **text_to_brain_metrics(pred.numpy(), true.numpy(), X_te.numpy(), encoder=ae.encoder, image_head=image_head, device=device, pct=pct)}


def train_text_to_brain_v2(text_tr, latent_tr, text_va, latent_va, *, decoder, in_dim: int, device, epochs: int = 80, batch_size: int = 512,
                           lr: float = 1e-3, weight_decay: float = 0.01, noise: float = 0.5, w_logit: float = 1.0, w_prob: float = 0.3,
                           dice_pct: int = 70, seed: int = 0, head_seed: int = 123, max_batches: int | None = None, log=None) -> tuple[Any, dict]:
    """Text-to-brain head v2 (scripts/models/09_projection_head.py, train_text_to_brain): ProjHead(in_dim, 512, 384) started at the
    mean training latent, latent MSE + Pearson terms on decoder logits and sigmoid maps, input noise, kept at its best validation
    Dice (``dice_pct`` percentile masks vs the decoded true latent)."""
    import copy
    import torch
    import torch.nn.functional as F
    from neurovlm.models import ProjHead

    X_tr, Y_tr = text_tr.float().to(device), latent_tr.float().to(device)
    X_va, Y_va = text_va.float().to(device), latent_va.float().to(device)
    decoder = copy.deepcopy(decoder).to(device).eval()
    decoder.requires_grad_(False)

    def zrows(x):
        x = x - x.mean(1, keepdim=True)
        return x / x.std(1, keepdim=True, unbiased=False).clamp_min(1e-6)

    def pct_mask(x):  # x > row-wise np.percentile(x, dice_pct) (linear interpolation)
        v, _ = x.sort(dim=1)
        pos = dice_pct / 100.0 * (x.shape[1] - 1)
        lo, hi = int(np.floor(pos)), int(np.ceil(pos))
        return x > (v[:, lo] + (v[:, hi] - v[:, lo]) * (pos - lo))[:, None]

    def corr_loss(a, b):
        return (1 - (zrows(a) * zrows(b)).mean(1)).mean()

    head = ProjHead(seed=head_seed, latent_in_dim=in_dim, hidden_dim=512, latent_out_dim=384).to(device)
    with torch.no_grad():  # latents are far from 0 (mean norm ~35)
        head.aligner[-1].bias += Y_tr.mean(0)

    def evaluate():
        head.eval()
        with torch.no_grad():
            pred = torch.cat([head(X_va[s:s + 2048]) for s in range(0, len(X_va), 2048)])
            dice, r = [], []
            for s in range(0, len(pred), 512):
                p, t = torch.sigmoid(decoder(pred[s:s + 512])), torch.sigmoid(decoder(Y_va[s:s + 512]))
                pb, tb = pct_mask(p), pct_mask(t)
                den = (pb.sum(1) + tb.sum(1)).double()
                dice.append(torch.where(den > 0, 2 * (pb & tb).sum(1).double() / den.clamp_min(1), torch.ones_like(den)))
                r.append((zrows(p) * zrows(t)).mean(1))
        head.train()
        return {"dice": float(torch.cat(dice).mean()), "pearson": float(torch.cat(r).mean())}

    def snapshot():
        return {k: v.detach().cpu().clone() for k, v in head.state_dict().items()}

    optimizer = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=weight_decay)
    gen = torch.Generator(device="cpu").manual_seed(seed)
    noise_gen = torch.Generator(device=device).manual_seed(seed)
    best = {"epoch": 0, "val": evaluate(), "state": snapshot()}
    curve, n, steps = [{"epoch": 0, **best["val"]}], len(X_tr), 0
    t0 = time.perf_counter()
    head.train()
    for epoch in range(1, epochs + 1):
        perm = torch.randperm(n, generator=gen).to(device)
        for bi, start in enumerate(range(0, n, batch_size)):
            if max_batches is not None and bi >= max_batches:
                break
            idx = perm[start:start + batch_size]
            x, y = X_tr[idx], Y_tr[idx]
            x = F.normalize(x + noise * torch.randn(x.shape, device=device, generator=noise_gen) / x.shape[1] ** 0.5, dim=1)
            pred = head(x)
            logits = decoder(pred)
            with torch.no_grad():
                target_logits = decoder(y)
            loss = F.mse_loss(pred, y) + w_logit * corr_loss(logits, target_logits) + w_prob * corr_loss(torch.sigmoid(logits), torch.sigmoid(target_logits))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            steps += 1
        res = evaluate()
        curve.append({"epoch": epoch, **res})
        if res["dice"] > best["val"]["dice"]:
            best = {"epoch": epoch, "val": res, "state": snapshot()}
        if log:
            log.info("t2b[v2] epoch %d val dice %.4f pearson %.4f (best %.4f @ %d)", epoch, res["dice"], res["pearson"], best["val"]["dice"], best["epoch"])
    seconds = time.perf_counter() - t0
    head.load_state_dict(best["state"])
    return head.cpu().eval(), {"loss": "head_v2", "best_epoch": best["epoch"], "val_best": best["val"], "val_curve": curve, "epochs": epochs,
                               "steps": steps, "seconds": seconds, "sec_per_epoch": seconds / max(epochs, 1), "sec_per_step": seconds / max(steps, 1)}
