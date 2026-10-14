#!/usr/bin/env python
"""Residual-MLP projectors vs the released PubMed Q-Former for brain-to-text (reviewer 2.1, plan item O7).

Training = scripts/models/13 with the projector swapped; evaluation = the qformer_pubmed component of scripts/pipeline/validate_models.py.
Usage: run_analysis.py --variants mlp_m --device cuda   (GPU queue: run_variants.sh; smoke: --smoke --device cpu)
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import re
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from _a5_common import (  # noqa: E402
    Timings, cached_bertscore, count_params, deranged_indices, exclude_heldout_rows, get_logger, load_lm, load_minilm,
    load_pubmed_latents, load_pubmed_maps, load_specter_latents, log_phase, match_lm_token_norm, pubmed_keep_mask, read_json,
    release_cuda_cache, released, same_exclusions, seed_all, select_device, set_offline_env, setup_torch, sha_key, split_masks,
    validity_checks, write_json,
)

set_offline_env()
log = get_logger("a5_mlp_projector")
REPO = HERE.parents[4]
VALIDATOR_REF = REPO / "scripts" / "outputs" / "validation" / "released_excl"  # read-only cross-check of the reference

# scripts/models/13 recipe
SEED = 17
CANONICAL_TOP_K_TEXT, CANONICAL_TOP_K_IMAGE = 32, 16
MIN_LM_WEIGHT, MIN_ALIGN_WEIGHT = 0.12, 0.20
CLINICAL_LM_MULT, CLINICAL_ALIGN_MULT, ARTIFACT_LM_MULT, ARTIFACT_ALIGN_MULT = 0.35, 0.60, 0.25, 0.50
CLINICAL_PATTERN = re.compile(
    r"\b(patient|patients|clinical|disease|disorder|syndrome|symptom|diagnosis|treatment|therapy|medication|drug|"
    r"schizophrenia|hallucination|psychosis|autism|adhd|depression|depressive|bipolar|anxiety|ptsd|"
    r"alzheimer|dementia|parkinson|stroke|tumou?r|lesion|addiction|substance|smoking|mdd|asd|ocd|tbi)\b", flags=re.IGNORECASE)
MAX_TOKENS, TRAIN_BATCH, EVAL_BATCH = 192, 32, 32
EPOCHS, PEAK_LR, MIN_LR, WARMUP_FRAC = 30, 3e-4, 3e-5, 0.05
ALIGN_WEIGHT, TOKEN_SPACE_WEIGHT, WEIGHT_DECAY = 0.35, 0.03, 0.03
NUM_QUERIES, QF_HIDDEN, DROPOUT = 32, 512, 0.05
# scripts/pipeline/validate_models.py qformer_pubmed decoding
GEN_KW = dict(max_new_tokens=128, num_beams=1, do_sample=False, repetition_penalty=1.25, no_repeat_ngram_size=4)
VALIDATOR_GEN_BATCH, VALIDATOR_ENCODE_BATCH, VALIDATOR_TEXT_BATCH = 512, 512, 32
GPU_REF = {"steps_per_epoch": 704, "val_batches": 89}  # full 13 run, for the runtime extrapolation
# O7 adoption tolerances (validate_models.py: AUC 0.01, cosine-type 0.02; validity stricter than its 0.05)
TOL = {"recall_auc": 0.01, "bertscore_f1_matched": 0.02, "semantic_sim_matched": 0.02, "contrastive_sim_matched": 0.02,
       "validity_frac_degenerate": 0.01, "validity_frac_duplicated_outputs": 0.02, "validity_fabrication_any": 0.02}
HIGHER = ("recall_auc", "bertscore_f1_matched", "semantic_sim_matched", "contrastive_sim_matched")
LOWER = ("validity_frac_degenerate", "validity_frac_duplicated_outputs", "validity_fabrication_any")

# width = trunk width, depth = residual blocks; xattn_lite (one cross-attention block) is optional and not in the default queue
VARIANTS = {
    "mlp_s": {"arch": "mlp", "width": 192, "depth": 2, "expansion": 2, "head": "tokens", "label": "MLP-S"},
    "mlp_m": {"arch": "mlp", "width": 704, "depth": 2, "expansion": 2, "head": "tokens", "label": "MLP-M (param-matched)"},
    "mlp_l": {"arch": "mlp", "width": 1536, "depth": 3, "expansion": 2, "head": "tokens", "label": "MLP-L"},
    "mlp_xl": {"arch": "mlp", "width": 2560, "depth": 4, "expansion": 2, "head": "tokens", "label": "MLP-XL"},
    "mlp_m_notoken": {"arch": "mlp", "width": 704, "depth": 2, "expansion": 2, "head": "single", "label": "MLP-M, 1 token repeated"},
    "mlp_m_lowlr": {"arch": "mlp", "width": 704, "depth": 2, "expansion": 2, "head": "tokens", "peak_lr": 1e-4, "min_lr": 1e-5,
                    "label": "MLP-M, lr 1e-4"},
    "xattn_lite": {"arch": "xattn_lite", "label": "Cross-attention lite"},
}
DEFAULT_VARIANTS = ("mlp_m", "mlp_l", "mlp_xl", "mlp_s", "mlp_m_lowlr", "mlp_m_notoken")
SMOKE = {"max_samples": 64, "epochs": 1, "batch_size": 8, "max_train_batches": 2, "max_eval_batches": 2, "max_tokens": 48, "gen_limit": 8,
         "max_new_tokens": 16, "gen_batch": 8}


def clean_space(x) -> str:
    return re.sub(r"\s+", " ", str(x).replace("\n", " ").strip())


class ResidualBlock(nn.Module):
    """Pre-LN residual MLP block: x + Linear(Dropout(GELU(Linear(LN(x)))))."""

    def __init__(self, width: int, expansion: int = 2, dropout: float = DROPOUT):
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.fc1 = nn.Linear(width, width * expansion)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)
        self.fc2 = nn.Linear(width * expansion, width)

    def forward(self, x):
        return x + self.fc2(self.drop(self.act(self.fc1(self.norm(x)))))


class ResidualMLPProjector(nn.Module):
    """Drop-in for ``QFormer`` (same ``forward(raw, semantic) -> [B, Q, lm_dim]`` and ``semantic_from_visual``), no attention.

    LN(raw) ++ LN(semantic) -> Linear -> residual blocks -> LN -> Linear to Q tokens (``head="tokens"``) or to one token
    repeated Q times (``"single"``) -> + learned per-query embedding -> LayerNorm, whose gain/bias start at the LM embedding
    scale/centroid (``token_scale``, ``token_center``)."""

    def __init__(self, image_dim=384, semantic_dim=384, lm_dim=1024, num_queries=NUM_QUERIES, *, width=704, depth=2, expansion=2,
                 dropout=DROPOUT, head="tokens", align_hidden=QF_HIDDEN, token_scale: float | None = None, token_center=None):
        super().__init__()
        self.lm_dim, self.num_queries, self.head_kind = int(lm_dim), int(num_queries), head
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


class CrossAttentionLite(nn.Module):
    """One cross-attention block (32 learned queries over the 2 projected input tokens) + FFN, then the Q-Former's ``to_lm``."""

    def __init__(self, image_dim=384, semantic_dim=384, lm_dim=1024, num_queries=NUM_QUERIES, hidden_dim=QF_HIDDEN, num_heads=8,
                 dropout=DROPOUT):
        super().__init__()
        self.raw_proj = nn.Sequential(nn.LayerNorm(image_dim), nn.Linear(image_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, hidden_dim))
        self.semantic_proj = nn.Sequential(nn.LayerNorm(semantic_dim), nn.Linear(semantic_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, hidden_dim))
        self.query_tokens = nn.Parameter(torch.randn(1, num_queries, hidden_dim) * 0.02)
        self.norm_q, self.norm_m = nn.LayerNorm(hidden_dim), nn.LayerNorm(hidden_dim)
        self.attn = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm_f = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(nn.Linear(hidden_dim, hidden_dim * 4), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_dim * 4, hidden_dim))
        self.to_lm = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, hidden_dim * 2), nn.GELU(), nn.Linear(hidden_dim * 2, lm_dim))
        self.align_head = nn.Sequential(nn.LayerNorm(lm_dim), nn.Linear(lm_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, semantic_dim))

    def forward(self, raw_images, semantic_images):
        dtype = self.query_tokens.dtype
        mem = torch.stack([self.raw_proj(raw_images.to(dtype)), self.semantic_proj(semantic_images.to(dtype))], dim=1)
        q = self.query_tokens.expand(raw_images.size(0), -1, -1)
        a, _ = self.attn(self.norm_q(q), self.norm_m(mem), self.norm_m(mem), need_weights=False)
        q = q + a
        q = q + self.ffn(self.norm_f(q))
        return self.to_lm(q)

    def semantic_from_visual(self, visual_tokens):
        x = visual_tokens.to(dtype=self.query_tokens.dtype).mean(dim=1)
        return F.normalize(self.align_head(x).float(), dim=1)


def build_projector(name: str, *, image_dim=384, semantic_dim=384, lm_dim=1024, token_scale=None, token_center=None) -> nn.Module:
    spec = VARIANTS[name]
    if spec["arch"] == "xattn_lite":
        return CrossAttentionLite(image_dim, semantic_dim, lm_dim, NUM_QUERIES)
    return ResidualMLPProjector(image_dim, semantic_dim, lm_dim, NUM_QUERIES, width=spec["width"], depth=spec["depth"],
                                expansion=spec["expansion"], head=spec["head"], token_scale=token_scale, token_center=token_center)


class SemanticPrompt(nn.Module):
    """``NeuroQFormer.forward`` with the canonical projection off: projector(raw, normalize(image_head(raw)))."""

    def __init__(self, projector: nn.Module, image_head: nn.Module):
        super().__init__()
        self.projector, self.image_head = projector, image_head

    def forward(self, raw):
        sem = F.normalize(self.image_head(raw.float()), dim=1)
        return self.projector(raw, F.normalize(sem.float(), dim=1))


def variant_table(lm_dim: int = 1024) -> list[dict]:
    """Parameter counts of the Q-Former and every variant (built on the meta device)."""
    rows = []
    from neurovlm.models.qformer import QFormer

    with torch.device("meta"):
        qf = QFormer(384, 384, lm_dim, num_queries=NUM_QUERIES, hidden_dim=QF_HIDDEN, num_heads=8, num_layers=6)
    rows.append({"variant": "qformer", "label": "Q-Former (released)", "params": count_params(qf), "width": QF_HIDDEN, "depth": 6,
                 "head": "6-layer cross-attention decoder", "peak_lr": PEAK_LR})
    for name, spec in VARIANTS.items():
        with torch.device("meta"):
            m = build_projector(name, lm_dim=lm_dim)
        rows.append({"variant": name, "label": spec["label"], "params": count_params(m), "width": spec.get("width", QF_HIDDEN),
                     "depth": spec.get("depth", 1), "head": spec.get("head", "to_lm"), "peak_lr": spec.get("peak_lr", PEAK_LR)})
    return rows


# data (13 cells 2-5, 8)
def load_rows(max_samples: int | None, *, exclusions: bool = True):
    """``pubmed_summary_qa`` rows aligned to the HF latents with the paper's split flags.

    Excluded PMIDs only lose their val/test flag (they stay in the quality-weight neighbour pool), before the prefix."""
    import pandas as pd
    from _hf_pins import hf_file

    # the parquet behind the HF dataset's train split, at its pinned revision (as scripts/models/13_qformer_pubmed.py)
    summary = pd.read_parquet(hf_file("neurovlm/pubmed_summary_qa", "pubmed_summary_qa.parquet"))
    summary = summary.rename(columns={"title": "name", "summary": "description"}).sort_values("pmid").reset_index(drop=True)
    latents_all, pmids_all = load_pubmed_latents()
    shared = np.intersect1d(summary["pmid"].to_numpy(), pmids_all)
    summary = summary[summary["pmid"].isin(shared)].sort_values("pmid").reset_index(drop=True)
    pos = {p: i for i, p in enumerate(pmids_all)}
    idx = np.array([pos[p] for p in summary["pmid"].to_numpy()])
    latents, pmids = latents_all[torch.as_tensor(idx)], pmids_all[idx]
    masks, excl = exclude_heldout_rows(split_masks(pmids), pmids, enabled=exclusions, log=log.info)
    for k in masks:
        summary[k] = masks[k]
    if max_samples and max_samples < len(summary):
        keep = np.concatenate([np.flatnonzero(masks[s])[:k] for s, k in (("train", max(4, int(max_samples * 0.75))),
                                                                       ("val", max(2, max_samples // 8)), ("test", max(2, max_samples // 8)))])
        keep = np.sort(keep)
        summary = summary.iloc[keep].reset_index(drop=True)
        latents, pmids = latents[torch.as_tensor(keep)], pmids[keep]
    log.info("training rows (HF pubmed_summary_qa x HF pubmed_images latents): %d | train %d | val %d | test %d", len(summary),
             summary["train"].sum(), summary["val"].sum(), summary["test"].sum())
    return summary, latents, pmids, excl


def quality_weights(df, latents, pmids, train_mask, text_head, image_head, device):
    """13 cell 4: permissive quality scoring -> lm/align weights per row (nothing dropped)."""
    latent_text = load_specter_latents(pmids)

    @torch.no_grad()
    def project(head, x, bs=1024):
        return F.normalize(torch.cat([head(x[s:s + bs].to(device)).detach().float().cpu() for s in range(0, len(x), bs)]), dim=1)

    @torch.no_grad()
    def topk_cross(query, key, k, bs=64):
        key_d = key.to(device)
        vals, idx = [], []
        for s in range(0, len(query), bs):
            top = (query[s:s + bs].to(device) @ key_d.T).topk(k, dim=1)
            vals.append(top.values.cpu())
            idx.append(top.indices.cpu())
        return torch.cat(vals), torch.cat(idx)

    @torch.no_grad()
    def topk_self(x, k, bs=64):
        x_d = x.to(device)
        out = []
        for s in range(0, len(x), bs):
            e = min(s + bs, len(x))
            sim = x[s:e].to(device) @ x_d.T
            sim[torch.arange(e - s, device=device), torch.arange(s, e, device=device)] = -float("inf")
            out.append(sim.topk(min(k, sim.shape[1] - 1), dim=1).indices.cpu())
        return torch.cat(out)

    text_emb, image_emb = project(text_head, latent_text), project(image_head, latents)
    retrieved_sim, retrieved_idx = topk_cross(image_emb, text_emb, min(CANONICAL_TOP_K_TEXT, len(text_emb)))
    neighbor_idx = topk_self(image_emb, CANONICAL_TOP_K_IMAGE)
    true_text = text_emb.unsqueeze(1)
    retrieved_consistency = (true_text * text_emb[retrieved_idx]).sum(-1).mean(1)
    neighbor_consistency = (true_text * text_emb[neighbor_idx]).sum(-1).mean(1)
    paired = (image_emb * text_emb).sum(1)
    score = 0.35 * retrieved_consistency + 0.35 * neighbor_consistency + 0.20 * paired + 0.10 * retrieved_sim[:, 0]
    text = df["name"].fillna("") + " " + df["description"].fillna("")
    clinical = text.str.contains(CLINICAL_PATTERN, regex=True, na=False).to_numpy()
    artifact = text.str.contains(r"\[[0-9JjIi]+\]|\bPMID\b|\bdoi\b", regex=True, na=False).to_numpy()
    tr = score[torch.as_tensor(train_mask)]
    lo, hi = torch.quantile(tr, 0.05), torch.quantile(tr, 0.95)
    quality = ((score - lo) / (hi - lo).clamp_min(1e-6)).clamp(0, 1).numpy()
    lm_w = MIN_LM_WEIGHT + (1 - MIN_LM_WEIGHT) * quality
    al_w = MIN_ALIGN_WEIGHT + (1 - MIN_ALIGN_WEIGHT) * quality
    lm_w[clinical] *= CLINICAL_LM_MULT
    al_w[clinical] *= CLINICAL_ALIGN_MULT
    lm_w[artifact] *= ARTIFACT_LM_MULT
    al_w[artifact] *= ARTIFACT_ALIGN_MULT
    df["lm_weight"] = np.clip(lm_w, MIN_LM_WEIGHT, 1.0).astype("float32")
    df["align_weight"] = np.clip(al_w, MIN_ALIGN_WEIGHT, 1.0).astype("float32")
    log.info("quality weights: clinical-like %d | artifact-like %d | lm mean %.3f | align mean %.3f", clinical.sum(), artifact.sum(),
             df["lm_weight"].mean(), df["align_weight"].mean())
    return image_emb, text_emb


def tokenize_cached(texts, tok, *, max_tokens: int, cache_dir: Path):
    key = sha_key("\n\x00".join(texts), max_tokens, tok.eos_token_id, tok.name_or_path)
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"tokens-{key}.pt"
    if path.exists():
        payload = torch.load(path, weights_only=True)
        return payload["ids"], payload["lengths"]
    enc = tok(list(texts), truncation=True, max_length=max_tokens - 1, add_special_tokens=False)
    ids = []
    for row in enc["input_ids"]:
        row = list(row)
        if not row or row[-1] != tok.eos_token_id:
            row.append(tok.eos_token_id)
        ids.append(torch.tensor(row[:max_tokens], dtype=torch.long))
    lengths = torch.tensor([len(r) for r in ids], dtype=torch.long)
    torch.save({"ids": ids, "lengths": lengths}, path)
    return ids, lengths


class Rows:
    """Row store + ``collate`` for a list of indices (13 cell 8)."""

    def __init__(self, raw, sem_img, sem_txt, lm_w, align_w, ids, lengths, pad_id):
        self.raw, self.sem_img, self.sem_txt = raw.float(), sem_img.float(), sem_txt.float()
        self.lm_w, self.align_w = torch.as_tensor(lm_w, dtype=torch.float32), torch.as_tensor(align_w, dtype=torch.float32)
        self.ids, self.lengths, self.pad_id = ids, lengths, pad_id

    def __len__(self):
        return len(self.ids)

    def collate(self, idx):
        from torch.nn.utils.rnn import pad_sequence

        idx_t = torch.as_tensor(idx, dtype=torch.long)
        lengths = self.lengths[idx_t]
        input_ids = pad_sequence([self.ids[i] for i in idx], batch_first=True, padding_value=self.pad_id)
        attn = (torch.arange(input_ids.size(1))[None, :] < lengths[:, None]).long()
        return (self.raw[idx_t], self.sem_img[idx_t], self.sem_txt[idx_t], self.lm_w[idx_t], self.align_w[idx_t], input_ids, attn)


def epoch_batches(rows: Rows, *, batch_size: int, shuffle: bool, seed: int, drop_last: bool, bucket: bool):
    n = len(rows)
    if not shuffle:
        order = torch.arange(n)
        return [order[s:s + batch_size].tolist() for s in range(0, n, batch_size)]
    g = torch.Generator().manual_seed(seed)
    order = torch.randperm(n, generator=g)
    if bucket:
        mega, chunks = batch_size * 50, []
        for s in range(0, n, mega):
            part = order[s:s + mega]
            chunks.append(part[torch.argsort(rows.lengths[part], descending=True, stable=True)])
        order = torch.cat(chunks)
    batches = [order[s:s + batch_size].tolist() for s in range(0, n, batch_size)]
    if drop_last and len(batches) > 1 and len(batches[-1]) < batch_size:
        batches = batches[:-1]
    if bucket:
        batches = [batches[i] for i in torch.randperm(len(batches), generator=g).tolist()]
    return batches


# objective (13 cell 12)
def make_objective(lm, lm_stats, device, amp: bool):
    def token_space_reg(vis):
        vf = vis.float()
        reg_norm = ((vf.norm(dim=-1).mean() - lm_stats["norm"]) / max(lm_stats["norm"], 1e-6)).pow(2)
        reg_std = ((vf.std() - lm_stats["std"]) / max(lm_stats["std"], 1e-6)).pow(2)
        return reg_norm + 0.25 * reg_std

    def wmean(v, w):
        return (v * w).sum() / w.sum().clamp_min(1e-6)

    def objective(module, batch, image_mode: str = "correct"):
        raw, sem_img, sem_txt, lm_w, al_w, input_ids, attn = (t.to(device, non_blocking=True) for t in batch)
        sem_txt = F.normalize(sem_txt.float(), dim=1)
        if image_mode == "shuffled":
            perm = torch.randperm(raw.size(0), device=device)
            raw, sem_img = raw[perm], sem_img[perm]
        elif image_mode == "zero":
            raw, sem_img = torch.zeros_like(raw), torch.zeros_like(sem_img)
        B = raw.size(0)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=(device.type == "cuda" and amp)):
            vis_raw = module(raw, sem_img)
            vis = match_lm_token_norm(vis_raw, lm_stats["norm"])
            Q = vis.size(1)
            with torch.no_grad():
                txt = lm.get_input_embeddings()(input_ids)
            embeds = torch.cat([vis.to(lm.dtype), txt.to(lm.dtype)], dim=1)
            mask = torch.cat([torch.ones(B, Q, dtype=torch.long, device=device), attn], dim=1)
            labels = torch.cat([torch.full((B, Q), -100, dtype=torch.long, device=device), input_ids.masked_fill(attn == 0, -100)], dim=1)
            out = lm(inputs_embeds=embeds, attention_mask=mask, use_cache=False)
            logits = out.logits[:, :-1].float().contiguous()
            shift = labels[:, 1:].contiguous()
            valid = shift.ne(-100)
            safe = shift.masked_fill(~valid, 0)
            tok_loss = F.cross_entropy(logits.view(-1, logits.size(-1)), safe.view(-1), reduction="none").view_as(safe)
            per_lm = (tok_loss * valid.float()).sum(1) / valid.float().sum(1).clamp_min(1.0)
            lm_term = wmean(per_lm, lm_w.float())
            pred_sem = module.semantic_from_visual(vis_raw)
            align = wmean(1.0 - (pred_sem * sem_txt).sum(1), al_w.float())
            reg = token_space_reg(vis_raw)
            loss = lm_term + ALIGN_WEIGHT * align + TOKEN_SPACE_WEIGHT * reg
            n_tok = valid.float().sum()
            unweighted_lm = (tok_loss * valid.float()).sum() / n_tok.clamp_min(1.0)
        return loss, {"lm": lm_term.detach(), "align": align.detach(), "reg": reg.detach(), "lm_unweighted": unweighted_lm.detach(), "tokens": n_tok.detach()}

    return objective


@torch.no_grad()
def eval_rows(module, objective, rows: Rows, *, batch_size: int, image_mode: str = "correct", max_batches: int | None = None) -> dict:
    module.eval()
    totals = {k: 0.0 for k in ("loss", "lm", "align", "reg")}
    tok_sum, tok_n, n = 0.0, 0.0, 0
    for idx in epoch_batches(rows, batch_size=batch_size, shuffle=False, seed=0, drop_last=False, bucket=False):
        loss, parts = objective(module, rows.collate(idx), image_mode=image_mode)
        totals["loss"] += float(loss)
        for k in ("lm", "align", "reg"):
            totals[k] += float(parts[k])
        tok_sum += float(parts["lm_unweighted"]) * float(parts["tokens"])
        tok_n += float(parts["tokens"])
        n += 1
        if max_batches is not None and n >= max_batches:
            break
    out = {k: v / max(n, 1) for k, v in totals.items()}
    out["token_ce"] = tok_sum / max(tok_n, 1.0)
    out["perplexity"] = math.exp(min(out["token_ce"], 50.0))
    out["batches"] = n
    return out


# training (13 cells 10, 14) with per-epoch resume
def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def train_variant(name: str, module: nn.Module, objective, train_rows: Rows, val_rows: Rows, *, lm, device, epochs: int, batch_size: int,
                  eval_batch: int, max_train_batches, max_eval_batches, bucket: bool, ckpt_dir: Path, timings: Timings, seed: int,
                  ablation_every: int, grad_ckpt: bool, fingerprint: dict, resume: bool, log_every: int = 100) -> dict:
    spec = VARIANTS[name]
    peak_lr, min_lr = spec.get("peak_lr", PEAK_LR), spec.get("min_lr", MIN_LR)
    n_batches = len(epoch_batches(train_rows, batch_size=batch_size, shuffle=True, seed=0, drop_last=True, bucket=bucket))
    if max_train_batches:
        n_batches = min(n_batches, max_train_batches)
    total_steps = max(1, epochs * n_batches)
    warmup = max(1, int(WARMUP_FRAC * total_steps))
    optimizer = torch.optim.AdamW(module.parameters(), lr=peak_lr, weight_decay=WEIGHT_DECAY, fused=(device.type == "cuda"))

    def lr_lambda(step):
        if step < warmup:
            return (step + 1) / warmup
        progress = (step - warmup) / max(1, total_steps - warmup)
        return min_lr / peak_lr + (1 - min_lr / peak_lr) * 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    history = {k: [] for k in ("train", "train_lm", "train_align", "val", "val_lm", "val_align", "shuffle", "zero", "lr", "epoch_seconds",
                               "train_seconds", "val_seconds", "eval_batches", "sec_per_step")}
    best_val, best_epoch, start_epoch, first_steps = float("inf"), -1, 0, []
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_path, final_path, resume_path = ckpt_dir / "best.pt", ckpt_dir / "final.pt", ckpt_dir / "resume.pt"
    fp_key = sha_key(json.dumps(fingerprint, sort_keys=True, default=str))
    if resume and resume_path.exists():
        st = torch.load(resume_path, map_location=device, weights_only=False)
        if st.get("fingerprint") == fp_key:
            module.load_state_dict(st["module"])
            optimizer.load_state_dict(st["optimizer"])
            scheduler.load_state_dict(st["scheduler"])
            history, best_val, best_epoch, first_steps = st["history"], st["best_val"], st["best_epoch"], st.get("first_steps", [])
            start_epoch = st["epoch"] + 1
            log.info("%s: resuming after epoch %d (best val %.4f @ %d)", name, start_epoch, best_val, best_epoch + 1)
        else:
            log.warning("%s: resume.pt has a different fingerprint; training from scratch", name)
    log.info("training %s: %d epochs x %d steps (total %d, warm-up %d) | lr %.0e -> %.0e | params %s | grad-ckpt %s", name, epochs, n_batches,
             total_steps, warmup, peak_lr, min_lr, f"{count_params(module):,}", grad_ckpt)
    t_all = time.perf_counter()
    for epoch in range(start_epoch, epochs):
        t_epoch = time.perf_counter()
        module.train()
        lm.train(grad_ckpt)  # gradient checkpointing needs train mode; Qwen3 has no dropout
        totals = {k: torch.zeros((), device=device) for k in ("loss", "lm", "align")}
        batches = epoch_batches(train_rows, batch_size=batch_size, shuffle=True, seed=seed + epoch, drop_last=True, bucket=bucket)
        if max_train_batches:
            batches = batches[:max_train_batches]
        sync(device)
        t_train = time.perf_counter()
        for bi, idx in enumerate(batches):
            t0 = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            loss, parts = objective(module, train_rows.collate(idx))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(module.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            totals["loss"] += loss.detach()
            totals["lm"] += parts["lm"]
            totals["align"] += parts["align"]
            if bi < 3 or (bi + 1) % log_every == 0:
                sync(device)
                dt = time.perf_counter() - t0
                if epoch == start_epoch and bi < 3:
                    first_steps.append(dt)
                log.info("%s epoch %d step %d/%d loss %.4f lr %.2e (%.3fs/step)", name, epoch + 1, bi + 1, len(batches), float(loss.detach()),
                         scheduler.get_last_lr()[0], dt)
        sync(device)
        train_sec = time.perf_counter() - t_train
        lm.eval()
        nb = max(len(batches), 1)
        t_val = time.perf_counter()
        val = eval_rows(module, objective, val_rows, batch_size=eval_batch, max_batches=max_eval_batches)
        shuf = zero = None
        n_eval = val["batches"]
        if ablation_every > 0 and (epoch % ablation_every == 0 or epoch == epochs - 1):
            r_s = eval_rows(module, objective, val_rows, batch_size=eval_batch, image_mode="shuffled", max_batches=max_eval_batches or 100)
            r_z = eval_rows(module, objective, val_rows, batch_size=eval_batch, image_mode="zero", max_batches=max_eval_batches or 100)
            shuf, zero, n_eval = r_s["loss"], r_z["loss"], n_eval + r_s["batches"] + r_z["batches"]
        val_sec = time.perf_counter() - t_val
        for k, v in (("train", float(totals["loss"]) / nb), ("train_lm", float(totals["lm"]) / nb), ("train_align", float(totals["align"]) / nb),
                     ("val", val["loss"]), ("val_lm", val["lm"]), ("val_align", val["align"]), ("shuffle", shuf), ("zero", zero),
                     ("lr", scheduler.get_last_lr()[0]), ("epoch_seconds", time.perf_counter() - t_epoch), ("train_seconds", train_sec),
                     ("val_seconds", val_sec), ("eval_batches", n_eval), ("sec_per_step", train_sec / nb)):
            history[k].append(v)
        state = {k: v.detach().cpu() for k, v in module.state_dict().items()}
        torch.save(state, final_path)
        if val["loss"] < best_val:
            best_val, best_epoch = val["loss"], epoch
            torch.save(state, best_path)
        if resume and epoch < epochs - 1:
            torch.save({"module": module.state_dict(), "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(), "epoch": epoch,
                        "history": history, "best_val": best_val, "best_epoch": best_epoch, "first_steps": first_steps, "fingerprint": fp_key},
                       resume_path)
        gap = f"shuffle gap {shuf - val['loss']:+.4f} zero gap {zero - val['loss']:+.4f}" if shuf is not None else "ablations skipped"
        log.info("%s epoch %3d/%d | train %.4f | val %.4f lm %.4f align %.4f | %s | %.3f s/step, val %.0fs, epoch %.0fs | ETA %.1f min", name,
                 epoch + 1, epochs, history["train"][-1], val["loss"], val["lm"], val["align"], gap, train_sec / nb, val_sec,
                 history["epoch_seconds"][-1], (epochs - epoch - 1) * float(np.mean(history["epoch_seconds"])) / 60)
        write_json(ckpt_dir / "history.json", history)
    resume_path.unlink(missing_ok=True)
    sps = float(np.median(history["sec_per_step"])) if history["sec_per_step"] else None
    info = {"variant": name, "complete": True, "fingerprint": fingerprint, "fingerprint_key": fp_key, "best_epoch": best_epoch,
            "best_val_loss": best_val, "final_val_loss": history["val"][-1] if history["val"] else None, "epochs": epochs,
            "steps_per_epoch": n_batches, "batch_size": batch_size, "peak_lr": peak_lr, "min_lr": min_lr, "device": str(device),
            "sec_per_step_median": sps, "first_steps_sec": first_steps, "train_seconds_total": float(np.sum(history["train_seconds"])),
            "val_seconds_mean": float(np.mean(history["val_seconds"])) if history["val_seconds"] else None,
            "wall_seconds_this_call": time.perf_counter() - t_all, "grad_ckpt": grad_ckpt, "ablation_every": ablation_every,
            "best_path": str(best_path), "final_path": str(final_path)}
    if sps:
        scale = TRAIN_BATCH / batch_size  # linear-in-batch approximation (only matters for smoke runs)
        full_train_min = sps * scale * GPU_REF["steps_per_epoch"] * EPOCHS / 60
        evb = float(np.sum(history["val_seconds"])) / max(1, int(np.sum(history["eval_batches"]))) * (EVAL_BATCH / eval_batch)
        n_eval_batches = EPOCHS * GPU_REF["val_batches"] * (1 + 2 / max(ablation_every, 1)) if ablation_every > 0 else EPOCHS * GPU_REF["val_batches"]
        info["extrapolated_full_run"] = {"device": str(device), "train_minutes": full_train_min, "val_minutes": evb * n_eval_batches / 60,
                                         "note": "measured sec/step on this device scaled linearly to batch 32 x 704 steps x 30 epochs"}
        log.info("%s: %.3f s/step at batch %d on %s -> full 30-epoch run on this device ~ %.0f min train + %.0f min validation", name, sps,
                 batch_size, device, full_train_min, info["extrapolated_full_run"]["val_minutes"])
    timings.rows[f"train_{name}"] = {"seconds": info["train_seconds_total"], "sec_per_step_median": sps, "epochs": epochs,
                                     "steps_per_epoch": n_batches, "batch_size": batch_size, "device": str(device),
                                     "ts": time.strftime("%Y-%m-%dT%H:%M:%S")}
    timings.save()
    return info


# brain-to-text evaluation (validate_models.py::eval_qformer)
def b2t_test_pmids(*, exclusions: bool, limit: int | None, log_fn=None) -> tuple[np.ndarray, dict]:
    """Validator test set: sorted test-split PMIDs minus the excluded ones, then the ``limit`` prefix."""
    from _hf_pins import hf_split_pmids

    split = hf_split_pmids()  # split flags of the pinned HF publications.parquet, never the data-dir copy
    test = np.sort(np.asarray(split["test"]))
    keep = pubmed_keep_mask(test, enabled=exclusions, label="PubMed test (brain-to-text set)", log=log_fn)
    block = {"enabled": bool(exclusions), "source": "scripts/eval_exclusions/pubmed_exclusions.csv", "b2t_test_before": int(len(test)),
             "b2t_test_dropped": int((~keep).sum()), "b2t_test_kept": int(keep.sum())}
    test = test[keep]
    return (test[:limit] if limit else test), block


def load_b2t_material(*, exclusions: bool, limit: int | None) -> dict:
    """Maps + validator truth texts (``name [SEP] summary``) of the brain-to-text test set."""
    from _hf_pins import load_dataset

    pmids, block = b2t_test_pmids(exclusions=exclusions, limit=limit, log_fn=log.info)
    X = load_pubmed_maps(pmids)
    df = load_dataset("pubmed_text").drop_duplicates("pmid").set_index("pmid")
    rows = df.loc[pmids]
    names = rows["name"].fillna("").astype(str).tolist()
    summ = load_dataset("pubmed_summaries")
    summ["pmid"] = summ["pmid"].astype(df.index.dtype)
    summ = summ.drop_duplicates("pmid").set_index("pmid")
    summaries = summ.reindex(pmids)["summary"].fillna("").astype(str).str.strip("\n").str.strip(" \n").tolist()
    truth = [f"{n} [SEP] {s}" for n, s in zip(names, summaries)]
    log.info("brain-to-text test set: %d maps (validator recipe; %d leakage PMIDs dropped)", len(pmids), block["b2t_test_dropped"])
    return {"pmids": pmids, "X": X, "truth": truth, "exclusions": block, "pmid_key": sha_key(",".join(map(str, pmids)))}


@torch.no_grad()
def encode_raw(ae, X, device, bs=VALIDATOR_ENCODE_BATCH):
    return torch.cat([ae.encoder(X[s:s + bs].to(device)).detach().float().cpu() for s in range(0, len(X), bs)])


@torch.no_grad()
def soft_prompt_stats(prompt_fn, raw, lm_stats, device) -> dict:
    """Shape and scale of the soft prompt before the LM-norm rescaling, and its norm after it."""
    with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
        vis = prompt_fn(raw.to(device)).float()
    norms = vis.norm(dim=-1)
    vn = F.normalize(vis, dim=-1)
    q = vis.size(1)
    within = ((vn @ vn.transpose(1, 2)).sum((1, 2)) - q) / (q * (q - 1))
    matched = match_lm_token_norm(vis, lm_stats["norm"])
    return {"shape": list(vis.shape), "token_norm_mean": float(norms.mean()), "token_norm_std": float(norms.std()),
            "token_norm_min": float(norms.min()), "token_norm_max": float(norms.max()), "elem_mean": float(vis.mean()),
            "elem_std": float(vis.std()), "within_prompt_token_cosine": float(within.mean()),
            "across_sample_std": float(vis.std(dim=0).mean()) if len(vis) > 1 else None,
            "lm_seen_token_norm_mean": float(matched.norm(dim=-1).mean()), "lm_embedding_norm_mean": lm_stats["norm"],
            "lm_embedding_elem_std": lm_stats["std"]}


@torch.no_grad()
def generate_validator_style(prompt_fn, X, *, ae, lm, tok, lm_stats, device, gen_batch: int, max_new_tokens: int, name: str) -> list[str]:
    """``NeuroVLM._generate_brain_text`` as called by the validator for the PubMed Q-Former (no prefix, greedy)."""
    model_dtype = next(lm.parameters()).dtype
    texts, t0 = [], time.perf_counter()
    kw = dict(GEN_KW, max_new_tokens=max_new_tokens)
    for start in range(0, len(X), gen_batch):
        raw = ae.encoder(X[start:start + gen_batch].to(device))
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            vis = match_lm_token_norm(prompt_fn(raw), lm_stats["norm"]).to(model_dtype)
            attn = torch.ones(vis.shape[:2], dtype=torch.long, device=device)
            cuda_devices = [device.index or torch.cuda.current_device()] if device.type == "cuda" else []
            with torch.random.fork_rng(devices=cuda_devices, enabled=True):
                torch.manual_seed(123 + start)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(123 + start)
                out = lm.generate(inputs_embeds=vis, attention_mask=attn, eos_token_id=tok.eos_token_id,
                                  pad_token_id=tok.pad_token_id or tok.eos_token_id, temperature=1.0, top_p=1.0, top_k=50, **kw)
        texts.extend(tok.decode(row, skip_special_tokens=True).strip() for row in out)
        log.info("%s generated %d/%d (%.0fs)", name, len(texts), len(X), time.perf_counter() - t0)
    return [t.replace("\n\n", "\n") for t in texts]


def b2t_metrics(generated, X, truth, *, ae, specter, text_head, image_head, device, bert_model: str, st_model, tag: str = "") -> dict:
    """``validate_models.py::eval_qformer`` metrics: validity, shared-space retrieval, BERTScore (deberta, else roberta-large)
    and MiniLM cosine, each with a deranged null."""
    from neurovlm.metrics.semantic import exact_pmid_retrieval_metrics
    from tqdm.auto import tqdm

    generated = [str(g) for g in generated]
    n = len(generated)
    out = {"n": n}
    out.update(validity_checks(generated))
    null = deranged_indices(n, np.random.default_rng(0))
    with log_phase(log, f"{tag} SPECTER2 + shared-space retrieval", n=n), torch.no_grad():
        text_lat = torch.cat([torch.as_tensor(specter(generated[s:s + VALIDATOR_TEXT_BATCH])).detach().float().cpu()
                              for s in tqdm(range(0, n, VALIDATOR_TEXT_BATCH), desc=f"{tag} SPECTER2", leave=False, disable=n <= 256)])
        brain_lat = encode_raw(ae, X, device)
        z_text = torch.cat([F.normalize(text_head(F.normalize(text_lat[s:s + 512].to(device), dim=1, eps=1e-8)).float(), dim=1, eps=1e-8).cpu()
                            for s in range(0, n, 512)])
        z_brain = torch.cat([F.normalize(image_head(brain_lat[s:s + 512].to(device)).float(), dim=1, eps=1e-8).cpu() for s in range(0, n, 512)])
    scores = z_text @ z_brain.T
    diag = scores.diag()
    out["contrastive_sim_matched"] = float(diag.mean())
    out["contrastive_sim_null"] = float(scores[torch.arange(n), torch.as_tensor(null)].mean())
    out["contrastive_null_test_frac_p_lt_0.05"] = float(((scores > diag[:, None]).float().mean(dim=1) < 0.05).float().mean())
    ret = exact_pmid_retrieval_metrics(scores)
    out["recall_auc"] = float(ret["normalized_k_recall_curve_auc"])
    out.update({f"retrieval_{k}": float(v) for k, v in ret.items() if not k.startswith("random")})
    if bert_model != "none":
        with log_phase(log, f"{tag} BERTScore matched + null", n=n):
            pairs = {"matched": list(truth), "null": [truth[i] for i in null]}
            scorer = cached_bertscore(bert_model, device, log)
            f1 = scorer.f1_many(generated, pairs, desc=f"{tag} BERTScore")
            if f1 is None and bert_model == "deberta":
                log.warning("deberta-xlarge-mnli unavailable (%s); falling back to roberta-large (as the validator)", scorer.note)
                scorer = cached_bertscore("roberta", device, log)
                f1 = scorer.f1_many(generated, pairs, desc=f"{tag} BERTScore")
            if f1 is not None:
                out["bertscore_model"] = scorer.model_type
                out["bertscore_f1_matched"], out["bertscore_f1_null"] = float(f1["matched"].mean()), float(f1["null"].mean())
            else:
                out["bertscore_note"] = scorer.note
                log.warning("BERTScore skipped: %s", scorer.note)
    try:
        with log_phase(log, f"{tag} MiniLM (generated + reference, one call)", texts=2 * n):
            e = st_model.encode(generated + [str(t) for t in truth], batch_size=32, convert_to_numpy=True, normalize_embeddings=True,
                                show_progress_bar=2 * n > 2000)
            g, t = e[:n], e[n:]
            out["semantic_sim_matched"] = float(np.mean(np.einsum("ij,ij->i", g, t)))
            out["semantic_sim_null"] = float(np.mean(np.einsum("ij,ij->i", g, t[null])))
            pair = g @ g.T
            out["diversity_mean_pairwise_cosine"] = float((pair.sum() - np.trace(pair)) / max(n * (n - 1), 1))
    except Exception as exc:
        out["semantic_note"] = f"skipped: {type(exc).__name__}: {exc}"
        log.warning("MiniLM metrics skipped: %s", exc)
    return out


def reference_crosscheck(ref: dict, gen_df) -> dict | None:
    """Compare our released-Q-Former reference with the validator's released_excl outputs."""
    mpath, gpath = VALIDATOR_REF / "metrics.json", VALIDATOR_REF / "df_gen_pubmed.parquet"
    if not mpath.exists():
        return None
    try:
        import pandas as pd

        v = read_json(mpath)["components"]["qformer_pubmed"]
        keys = HIGHER + LOWER + ("bertscore_f1_null", "semantic_sim_null", "contrastive_sim_null")
        out = {"validator_metrics": str(mpath.relative_to(REPO)), "validator_n": v.get("n"), "ours_n": ref.get("n"),
               "comparable": ref.get("n") == v.get("n"), "validator": {k: v.get(k) for k in keys},
               "delta_ours_minus_validator": {k: (ref[k] - v[k]) for k in keys if k in ref and k in v}}
        if gpath.exists():
            d = pd.read_parquet(gpath)
            m = dict(zip(d["pmid"].astype(str), d["text"].astype(str)))
            pairs = [(g, m[p]) for p, g in zip(gen_df["pmid"].astype(str), gen_df["generated"].astype(str)) if p in m]
            out["n_texts_compared"] = len(pairs)
            out["identical_text_frac"] = float(np.mean([a == b for a, b in pairs])) if pairs else None
            # greedy: a shorter generation must be a prefix of the validator's 128-token text
            out["prefix_match_frac"] = float(np.mean([b.startswith(a) for a, b in pairs])) if pairs else None
        return out
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}


# O7 adoption
def adoption_verdict(ref: dict, cand: dict) -> dict:
    checks, margins = {}, []
    for k in HIGHER + LOWER:
        if ref.get(k) is None or cand.get(k) is None:
            checks[k] = {"status": "n/a"}
            continue
        r, c, tol = float(ref[k]), float(cand[k]), TOL[k]
        ok = (c >= r - tol) if k in HIGHER else (c <= r + tol)
        margin = ((c - (r - tol)) if k in HIGHER else ((r + tol) - c)) / tol
        margins.append(margin)
        checks[k] = {"reference": r, "candidate": c, "delta": c - r, "tolerance": tol, "direction": "higher" if k in HIGHER else "lower",
                     "pass": bool(ok), "margin_in_tolerances": margin}
    decided = [v for v in checks.values() if "pass" in v]
    failed = [k for k, v in checks.items() if v.get("pass") is False]
    return {"adopt": bool(decided) and len(decided) == len(checks) and not failed, "failed": failed, "n_failed": len(failed),
            "n_undecided": len(checks) - len(decided), "mean_margin": float(np.mean(margins)) if margins else None, "criteria": checks}


def adoption_summary(ref: dict, results: dict, *, smoke: bool) -> dict:
    per = {v: {**adoption_verdict(ref, r), "params": r.get("params")} for v, r in results.items()}
    passing = [v for v in per if per[v]["adopt"]]
    rec = min(passing, key=lambda v: (per[v]["params"] or 0, -(per[v]["mean_margin"] or 0))) if passing else None
    closest = min(per, key=lambda v: (per[v]["n_failed"] + per[v]["n_undecided"], -(per[v]["mean_margin"] or -1e9))) if per else None
    verdict = f"ADOPT {rec}" if rec else "KEEP Q-FORMER"
    return {"verdict": verdict + (" (SMOKE RUN: untrained projectors, 8 items; not a decision)" if smoke else ""), "recommended": rec,
            "closest_if_none_pass": None if rec else closest, "passing": passing, "variants": per, "tolerances": TOL,
            "reference": {"variant": "qformer", "source": "released:neurovlm/NeuroQformer-PubMed", "params": ref.get("params"),
                          **{k: ref.get(k) for k in HIGHER + LOWER}},
            "rule": ("O7: a projector replaces the Q-Former only if, on the full PubMed test split (leakage PMIDs excluded), retrieval AUC >= "
                     "ref - 0.01, BERTScore F1 / MiniLM cosine / shared-space cosine >= ref - 0.02, and degenerate <= ref + 0.01, duplicated "
                     "<= ref + 0.02, fabricated-specifics <= ref + 0.02. Recommended = smallest passing variant (ties: larger mean margin)."),
            "smoke": smoke}


def tag_for(name: str, exclusions: bool) -> str:
    return name if exclusions else f"{name}_noexcl"


def eval_fingerprint(*, material_key: str, n: int, max_new_tokens: int, bert_model: str, checkpoint: str, train_key: str | None) -> dict:
    return {"pmid_key": material_key, "n": n, "max_new_tokens": max_new_tokens, "bert_model": bert_model, "checkpoint": checkpoint,
            "train_fingerprint_key": train_key, "decoding": {k: v for k, v in GEN_KW.items() if k != "max_new_tokens"}}


def eval_cached(path: Path, fingerprint: dict, exclusions: bool) -> bool:
    if not path.exists():
        return False
    prev = read_json(path)
    return prev.get("eval_fingerprint") == json.loads(json.dumps(fingerprint)) and same_exclusions(prev, exclusions)


def aggregate(out: Path, *, exclusions: bool, smoke: bool, meta: dict) -> dict:
    """metrics.json, table.csv and adoption.json from every cached eval/*.json."""
    import pandas as pd

    eval_dir = out / "eval"
    results = {}
    for n in ["qformer", *VARIANTS]:
        path = eval_dir / f"{tag_for(n, exclusions)}.json"
        if path.exists():
            results[n] = read_json(path)
    rows = []
    for k, r in results.items():
        tr, arch, sp = r.get("training") or {}, r.get("arch", {}), r.get("soft_prompt") or {}
        rows.append({"variant": k, "label": "Q-Former (released)" if k == "qformer" else VARIANTS[k]["label"], "params_M": r["params"] / 1e6,
                     **{c: arch.get(c) for c in ("width", "depth", "head", "peak_lr")},
                     "val_loss": r["val"]["loss"], "test_loss": r["test"]["loss"], "test_token_ce": r["test"]["token_ce"],
                     "shuffled_gap": r.get("test_shuffled_gap"),
                     **{c: r.get(c) for c in ("recall_auc", "contrastive_sim_matched", "contrastive_sim_null", "bertscore_f1_matched",
                                               "bertscore_f1_null", "semantic_sim_matched", "semantic_sim_null")},
                     "frac_degenerate": r.get("validity_frac_degenerate"), "frac_duplicated": r.get("validity_frac_duplicated_outputs"),
                     "fabrication_any": r.get("validity_fabrication_any"), "mean_words": r.get("validity_mean_words"),
                     "token_norm_mean": sp.get("token_norm_mean"), "within_prompt_token_cos": sp.get("within_prompt_token_cosine"),
                     "best_epoch": tr.get("best_epoch"), "sec_per_step": tr.get("sec_per_step_median"),
                     "train_minutes": (tr.get("train_seconds_total") or 0) / 60 if tr else None, "n": r.get("n")})
    table = pd.DataFrame(rows)
    table.to_csv(out / "table.csv", index=False)
    summary = {"results": results, "smoke": smoke, "exclusions_enabled": exclusions, **meta}
    cands = {k: v for k, v in results.items() if k != "qformer"}
    if "qformer" in results and cands:
        summary["adoption"] = adoption_summary(results["qformer"], cands, smoke=smoke)
        write_json(out / "adoption.json", summary["adoption"])
        log.info("O7 verdict: %s", summary["adoption"]["verdict"])
        for v, a in summary["adoption"]["variants"].items():
            log.info("  %-14s %s | failed: %s", v, "PASS" if a["adopt"] else "fail", ", ".join(a["failed"]) or "-")
    elif "qformer" not in results:
        log.info("no reference evaluation yet; adoption.json not written")
    write_json(out / "metrics.json", summary)
    if len(table):
        cols = ["variant", "params_M", "test_loss", "recall_auc", "bertscore_f1_matched", "semantic_sim_matched", "contrastive_sim_matched",
                "frac_degenerate", "frac_duplicated", "fabrication_any", "sec_per_step"]
        print(table[cols].to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    return summary


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--variants", default=",".join(DEFAULT_VARIANTS), help=f"Comma list from {', '.join(VARIANTS)} (or 'all').")
    p.add_argument("--out", type=Path, default=None, help="Output dir (default outputs/, outputs/smoke/ with --smoke).")
    p.add_argument("--device", default="auto")
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--epochs", type=int, default=None, help="Training epochs (default 30; 1 in --smoke).")
    p.add_argument("--batch-size", type=int, default=None, help="Train batch (default 32; 8 in --smoke).")
    p.add_argument("--eval-batch-size", type=int, default=None, help="Validation batch (default 32; train batch in --smoke).")
    p.add_argument("--max-samples", dest="max_samples", type=int, default=None, help="Cap rows (split-proportional prefix).")
    p.add_argument("--max-train-batches", type=int, default=None)
    p.add_argument("--max-eval-batches", type=int, default=None)
    p.add_argument("--max-tokens", type=int, default=None, help="Target truncation (default 192; 48 in --smoke).")
    p.add_argument("--gen-limit", type=int, default=None, help="Generate for the first N test maps (default all; 8 in --smoke).")
    p.add_argument("--gen-batch", type=int, default=None, help=f"Generation batch (default {VALIDATOR_GEN_BATCH}; 8 in --smoke).")
    p.add_argument("--max-new-tokens", type=int, default=None, help="Default 128; 16 in --smoke.")
    p.add_argument("--bert-model", choices=["deberta", "roberta", "none"], default="deberta")
    p.add_argument("--checkpoint", choices=["best", "final"], default="best", help="Evaluate the best-validation or final-epoch weights.")
    p.add_argument("--ablation-every", type=int, default=5, help="Shuffled/zero-image validation every N epochs (0 = never).")
    p.add_argument("--grad-ckpt", choices=["auto", "on", "off"], default="auto", help="LM gradient checkpointing (auto: on below 24 GB free).")
    p.add_argument("--no-bucket", action="store_true", help="No length bucketing.")
    p.add_argument("--amp", choices=["bf16", "fp32"], default="bf16", help="Autocast on CUDA only.")
    p.add_argument("--skip-reference", action="store_true", help="Do not evaluate the released Q-Former.")
    p.add_argument("--eval-only", action="store_true", help="Never train; evaluate variants with complete checkpoints.")
    p.add_argument("--aggregate-only", action="store_true", help="Only rebuild metrics.json/table.csv/adoption.json from eval/*.json.")
    p.add_argument("--smoke", action="store_true", help="64 rows, 1 epoch x 2 batches of 8, 8 short generations -> outputs/smoke/.")
    p.add_argument("--force", action="store_true", help="Retrain and re-evaluate.")
    p.add_argument("--force-eval", action="store_true", help="Re-evaluate (and regenerate) without retraining.")
    p.add_argument("--no-resume", action="store_true", help="Do not write/read per-epoch resume.pt.")
    p.add_argument("--no-exclusions", action="store_true", help="Keep the leakage-audit papers in val/test.")
    args = p.parse_args(argv)
    variants = list(DEFAULT_VARIANTS) if args.variants.strip() == "all" else [v.strip() for v in args.variants.split(",") if v.strip()]
    unknown = [v for v in variants if v not in VARIANTS]
    if unknown:
        p.error(f"unknown variants {unknown}; choose from {list(VARIANTS)}")
    if args.smoke:
        for k, v in SMOKE.items():
            setattr(args, k, getattr(args, k) or v)
    out = args.out or (HERE / "outputs" / ("smoke" if args.smoke else ""))
    out.mkdir(parents=True, exist_ok=True)
    exclusions = not args.no_exclusions
    epochs = args.epochs or EPOCHS
    batch_size = args.batch_size or TRAIN_BATCH
    eval_batch = args.eval_batch_size or (batch_size if args.smoke else EVAL_BATCH)
    max_tokens = args.max_tokens or MAX_TOKENS
    max_new_tokens = args.max_new_tokens or GEN_KW["max_new_tokens"]
    gen_batch = args.gen_batch or VALIDATOR_GEN_BATCH
    meta = {"epochs": epochs, "batch_size": batch_size, "max_tokens": max_tokens, "max_new_tokens": max_new_tokens, "gen_limit": args.gen_limit,
            "checkpoint": args.checkpoint, "inputs": "HF pubmed_images latents + released InfoNCE heads (train); released autoencoder on "
            "neuro_vectors test maps (eval), as the released NeuroQformer-PubMed"}

    vt = variant_table()
    log.info("variants (trainable parameters incl. the 0.72 M alignment head):")
    for r in vt:
        log.info("  %-14s %8.2f M | W %5s D %s | head %-32s | lr %.0e", r["variant"], r["params"] / 1e6, r["width"], r["depth"], r["head"], r["peak_lr"])
    write_json(out / "variants.json", vt)
    if args.aggregate_only:
        aggregate(out, exclusions=exclusions, smoke=args.smoke, meta=meta)
        return 0
    setup_torch(args.seed)
    device = select_device(args.device)

    timings = Timings(out / "timings.json")
    log.info("device %s | variants %s | epochs %d | batch %d | smoke %s | exclusions %s | out %s", device, variants, epochs, batch_size, args.smoke,
             exclusions, out)
    eval_dir, ckpt_root = out / "eval", out / "checkpoints"
    eval_dir.mkdir(parents=True, exist_ok=True)

    # what is left to do (cheap checks before loading anything heavy)
    test_pmids, _ = b2t_test_pmids(exclusions=exclusions, limit=args.gen_limit)
    material_key = sha_key(",".join(map(str, test_pmids)))
    train_fps = {v: {"variant": v, "spec": {k: x for k, x in VARIANTS[v].items() if k != "label"}, "epochs": epochs, "batch_size": batch_size,
                     "max_tokens": max_tokens, "max_samples": args.max_samples, "max_train_batches": args.max_train_batches,
                     "seed": args.seed, "exclusions": exclusions, "bucket": not args.no_bucket, "amp": args.amp, "inputs": "hf+released"}
                 for v in variants}

    def train_key(v):
        tj = ckpt_root / tag_for(v, exclusions) / "train.json"
        if not tj.exists() or not (tj.parent / f"{args.checkpoint}.pt").exists():
            return None
        t = read_json(tj)
        return t.get("fingerprint_key") if t.get("complete") and t.get("fingerprint") == json.loads(json.dumps(train_fps[v], default=str)) else None

    fingerprints = {v: eval_fingerprint(material_key=material_key, n=len(test_pmids), max_new_tokens=max_new_tokens, bert_model=args.bert_model,
                                        checkpoint="released" if v == "qformer" else args.checkpoint, train_key=None if v == "qformer" else train_key(v))
                    for v in ["qformer", *variants]}
    need_ref = not args.skip_reference and (args.force_eval or not eval_cached(eval_dir / f"{tag_for('qformer', exclusions)}.json", fingerprints["qformer"], exclusions))
    pending = [v for v in variants if args.force or args.force_eval or fingerprints[v]["train_fingerprint_key"] is None
               or not eval_cached(eval_dir / f"{tag_for(v, exclusions)}.json", fingerprints[v], exclusions)]
    missing = []
    if args.eval_only:
        missing = [v for v in pending if train_key(v) is None]
        for v in missing:
            log.warning("%s: no complete checkpoint for these settings; skipped (--eval-only)", v)
        pending = [v for v in pending if v not in missing]
    log.info("to do: reference %s | variants %s (cached: %s%s)", need_ref, pending or "-",
             [v for v in variants if v not in pending and v not in missing] or "-", f"; skipped: {missing}" if missing else "")
    if not need_ref and not pending:
        aggregate(out, exclusions=exclusions, smoke=args.smoke, meta=meta)
        return 0

    # shared material: training rows, frozen LM, eval models, brain-to-text test set
    with timings.time("load_data"), log_phase(log, "load training rows + quality weights"):
        df, latents, pmids, excl = load_rows(args.max_samples, exclusions=exclusions)
        text_head, image_head = released("proj_head_text_infonce", device), released("proj_head_image_infonce", device)
        image_sem, text_sem = quality_weights(df, latents, pmids, df["train"].to_numpy(dtype=bool), text_head, image_head, device)
        targets = [f"{clean_space(n)}\n{clean_space(d)}" for n, d in zip(df["name"], df["description"])]
    with timings.time("load_lm"), log_phase(log, "load frozen LM"):
        lm, tok, lm_stats = load_lm(device)
        free_gb = torch.cuda.mem_get_info(device)[0] / 2**30 if device.type == "cuda" else 0.0
        grad_ckpt = {"on": True, "off": False, "auto": device.type == "cuda" and free_gb < 24}[args.grad_ckpt]
        if grad_ckpt:
            lm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        with torch.no_grad():
            emb = lm.get_input_embeddings().weight.detach().float()
            token_center = emb.mean(0).cpu()
            del emb
        token_scale = lm_stats["norm"] / math.sqrt(lm.config.hidden_size)
        log.info("LM embeddings: mean token norm %.4f, element std %.4f -> token LayerNorm gain %.4f | grad-ckpt %s (free %.1f GB)",
                 lm_stats["norm"], lm_stats["std"], token_scale, grad_ckpt, free_gb)
    with timings.time("tokenize"):
        ids, lengths = tokenize_cached(targets, tok, max_tokens=max_tokens, cache_dir=out / "cache")

    def subset(mask):
        idx = np.flatnonzero(mask)
        return Rows(latents[idx], image_sem[idx], text_sem[idx], df["lm_weight"].to_numpy()[idx], df["align_weight"].to_numpy()[idx],
                    [ids[i] for i in idx], lengths[idx], tok.pad_token_id)

    train_rows, val_rows, test_rows = (subset(df[k].to_numpy(dtype=bool)) for k in ("train", "val", "test"))
    objective = make_objective(lm, lm_stats, device, amp=(args.amp == "bf16"))
    with timings.time("load_eval_models"), log_phase(log, "load eval models + brain-to-text test set"):
        ae = released("autoencoder", device)
        specter = released("specter", device)
        st_model = load_minilm(device)
        mat = load_b2t_material(exclusions=exclusions, limit=args.gen_limit)
    assert mat["pmid_key"] == material_key
    stats_raw = encode_raw(ae, mat["X"][: min(16, len(mat["X"]))], device)

    def evaluate(name: str, module: nn.Module, prompt_fn, *, extra: dict) -> dict:
        tag = tag_for(name, exclusions)
        res = {"variant": name, "params": count_params(module), "exclusions": {**excl, **mat["exclusions"]}, "eval_fingerprint": fingerprints[name],
               "smoke": args.smoke, **extra}
        module.eval()
        log.info("%s: released CUDA cache before evaluation (%s)", name, release_cuda_cache())
        with timings.time(f"loss_eval_{tag}"), log_phase(log, f"{name} val/test loss evaluation"):
            res["val"] = eval_rows(module, objective, val_rows, batch_size=eval_batch, max_batches=args.max_eval_batches)
            res["test"] = eval_rows(module, objective, test_rows, batch_size=eval_batch, max_batches=args.max_eval_batches)
            res["test_shuffled_gap"] = eval_rows(module, objective, test_rows, batch_size=eval_batch, image_mode="shuffled",
                                                 max_batches=args.max_eval_batches or 50)["loss"] - res["test"]["loss"]
        res["soft_prompt"] = soft_prompt_stats(prompt_fn, stats_raw, lm_stats, device)
        sp = res["soft_prompt"]
        log.info("%s soft prompt: token norm %.3f +- %.3f, within-prompt token cos %.3f", name, sp["token_norm_mean"], sp["token_norm_std"],
                 sp["within_prompt_token_cosine"])
        import pandas as pd

        gen_path = eval_dir / f"{tag}_generated.csv"
        df_gen = None
        if gen_path.exists() and not (args.force or args.force_eval):
            prev = pd.read_csv(gen_path, dtype={"pmid": str}).fillna("")
            if list(prev["pmid"]) == [str(x) for x in mat["pmids"]] and (eval_dir / f"{tag}.gen_fp.json").exists() and \
                    read_json(eval_dir / f"{tag}.gen_fp.json") == json.loads(json.dumps(fingerprints[name])):
                df_gen = prev
                log.info("%s: reusing generations %s", name, gen_path)
        if df_gen is None:
            with timings.time(f"generate_{tag}", n=len(mat["pmids"]), max_new_tokens=max_new_tokens, gen_batch=gen_batch) as grow, \
                    log_phase(log, f"{name} generation", n=len(mat["pmids"]), max_new_tokens=max_new_tokens, gen_batch=gen_batch):
                texts = generate_validator_style(prompt_fn, mat["X"], ae=ae, lm=lm, tok=tok, lm_stats=lm_stats, device=device, gen_batch=gen_batch,
                                                 max_new_tokens=max_new_tokens, name=name)
            res["generation_seconds"] = grow["seconds"]
            df_gen = pd.DataFrame({"pmid": [str(x) for x in mat["pmids"]], "generated": texts, "reference": mat["truth"]})
            df_gen.to_csv(gen_path, index=False)
            write_json(eval_dir / f"{tag}.gen_fp.json", fingerprints[name])
        # generation leaves the CUDA cache full of KV caches; free it before the metric encoders
        log.info("%s: released CUDA cache before metrics (%s)", name, release_cuda_cache())
        with timings.time(f"metrics_{tag}", n=len(df_gen)), log_phase(log, f"{name} metric block", n=len(df_gen)):
            res.update(b2t_metrics(df_gen["generated"].tolist(), mat["X"], mat["truth"], ae=ae, specter=specter, text_head=text_head,
                                   image_head=image_head, device=device, bert_model=args.bert_model, st_model=st_model, tag=name))
        log.info("%s: AUC %.4f | BERTScore %s | MiniLM %.4f | shared cos %.4f | degenerate %.3f duplicated %.3f fabricated %.3f | n %d", name,
                 res["recall_auc"], f"{res['bertscore_f1_matched']:.4f}" if "bertscore_f1_matched" in res else "n/a", res.get("semantic_sim_matched", float("nan")),
                 res["contrastive_sim_matched"], res["validity_frac_degenerate"], res["validity_frac_duplicated_outputs"],
                 res["validity_fabrication_any"], res["n"])
        if name == "qformer":
            res["validator_crosscheck"] = reference_crosscheck(res, df_gen)
            cc = res["validator_crosscheck"]
            if cc and "delta_ours_minus_validator" in cc:
                log.info("reference vs validator: identical texts %s, prefix-consistent %s of %s | deltas %s%s",
                         cc.get("identical_text_frac"), cc.get("prefix_match_frac"), cc.get("n_texts_compared"),
                         {k: round(v, 4) for k, v in cc["delta_ours_minus_validator"].items()},
                         "" if cc["comparable"] else f" (not comparable: n {cc['ours_n']} vs {cc['validator_n']})")
        write_json(eval_dir / f"{tag}.json", res)
        return res

    failures = {}
    # reference: the released PubMed Q-Former through the same path
    if need_ref:
        from _hf_pins import released_qformer

        with timings.time("load_qformer_reference"):
            nq = released_qformer("pubmed", device)
            nq.projection_temp, nq.canonical_basis, nq.use_canonical_projection = 0.035, "all", False  # validator generate_text kwargs
            qf = nq.qformer.to(device).float().eval()
        evaluate("qformer", qf, nq, extra={"source": "released:neurovlm/NeuroQformer-PubMed", "checkpoint": "released (final epoch)",
                                            "arch": {"width": QF_HIDDEN, "depth": 6, "head": "to_lm", "peak_lr": PEAK_LR}})
        del nq, qf
        log.info("reference done: released CUDA cache (%s)", release_cuda_cache())

    for v in pending:
        tag = tag_for(v, exclusions)
        ckpt_dir = ckpt_root / tag
        module = None
        try:
            seed_all(args.seed)
            module = build_projector(v, image_dim=latents.shape[1], semantic_dim=image_sem.shape[1], lm_dim=lm.config.hidden_size,
                                     token_scale=token_scale, token_center=token_center).to(device)
            tkey = train_key(v)
            if tkey is not None and not args.force:
                info = read_json(ckpt_dir / "train.json")
                log.info("%s: complete checkpoint found (%s); skipping training", v, ckpt_dir)
            else:
                if args.force:
                    (ckpt_dir / "resume.pt").unlink(missing_ok=True)  # --force = train from scratch
                with log_phase(log, f"{v} training", epochs=epochs):
                    info = train_variant(v, module, objective, train_rows, val_rows, lm=lm, device=device, epochs=epochs, batch_size=batch_size,
                                         eval_batch=eval_batch, max_train_batches=args.max_train_batches, max_eval_batches=args.max_eval_batches,
                                         bucket=not args.no_bucket, ckpt_dir=ckpt_dir, timings=timings, seed=args.seed,
                                         ablation_every=args.ablation_every, grad_ckpt=grad_ckpt, fingerprint=train_fps[v], resume=not args.no_resume)
                info["params"] = count_params(module)
                info["exclusions"] = excl
                write_json(ckpt_dir / "train.json", info)
                write_json(ckpt_dir / "config.json", {"variant": v, **VARIANTS[v], "image_dim": latents.shape[1],
                                                      "semantic_dim": image_sem.shape[1], "lm_dim": lm.config.hidden_size, "num_queries": NUM_QUERIES})
                fingerprints[v]["train_fingerprint_key"] = info["fingerprint_key"]
            module.load_state_dict(torch.load(ckpt_dir / f"{args.checkpoint}.pt", map_location=device, weights_only=True))
            module.eval()
            spec = VARIANTS[v]
            evaluate(v, module, SemanticPrompt(module, image_head).eval(),
                     extra={"label": spec["label"], "checkpoint": args.checkpoint, "training": {k: x for k, x in info.items() if k != "fingerprint"},
                            "arch": {"width": spec.get("width"), "depth": spec.get("depth"), "head": spec.get("head", "to_lm"),
                                     "expansion": spec.get("expansion"), "peak_lr": spec.get("peak_lr", PEAK_LR), "min_lr": spec.get("min_lr", MIN_LR)}})
        except Exception as exc:  # keep the queue going
            failures[v] = f"{type(exc).__name__}: {exc}"
            log.error("%s FAILED: %s\n%s", v, exc, traceback.format_exc())
        finally:
            # free the projector and the CUDA cache before the next variant; the frozen LM stays loaded
            module = None
            lm.eval()
            gc.collect()
            log.info("%s: projector dropped, released CUDA cache (%s)", v, release_cuda_cache())
    if failures:
        write_json(out / "failures.json", failures)
    aggregate(out, exclusions=exclusions, smoke=args.smoke, meta={**meta, "failures": failures})
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
