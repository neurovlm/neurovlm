#!/usr/bin/env python
"""Stage 13: stage 1 projector (grounded, PubMed), brain latent + semantic -> NeuroQwen3 soft tokens.

Port of docs/03_models/13_qformer_pubmed.ipynb. Default projector is the residual MLP (best-validation epoch packaged,
early stopping); --projector qformer is the deprecated Q-Former of the first submission. File names keep the qformer_ prefix either way.

    .venv/bin/python scripts/models/13_qformer_pubmed.py --run-name <run>
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import sys
import time
from pathlib import Path

import numpy as np

# expandable segments: peak ~23 GB reserved instead of ~31 GB on the 32 GB card
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common_models import *  # noqa: E402,F403

STAGE = "13_qformer_pubmed"
log = get_logger(STAGE)

SEED = 17  # default --seed (the paper run)
# cell 4
CANONICAL_TOP_K_TEXT, CANONICAL_TOP_K_IMAGE, CANONICAL_KEEP_FRAC_FOR_REPORTING = 32, 16, 0.35
CANONICAL_PROJ_BATCH, CANONICAL_SIM_BATCH = 1024, 64
MIN_LM_WEIGHT, MIN_ALIGN_WEIGHT = 0.12, 0.20
CLINICAL_LM_MULT, CLINICAL_ALIGN_MULT, ARTIFACT_LM_MULT, ARTIFACT_ALIGN_MULT = 0.35, 0.60, 0.25, 0.50
CLINICAL_PATTERN = re.compile(
    r"\b("
    r"patient|patients|clinical|disease|disorder|syndrome|symptom|diagnosis|treatment|therapy|medication|drug|"
    r"schizophrenia|hallucination|psychosis|autism|adhd|depression|depressive|bipolar|anxiety|ptsd|"
    r"alzheimer|dementia|parkinson|stroke|tumou?r|lesion|addiction|substance|smoking|"
    r"mdd|asd|ocd|tbi"
    r")\b",
    flags=re.IGNORECASE,
)
# cells 8, 10
MAX_TOKENS, TRAIN_BATCH, EVAL_BATCH = 192, 32, 32
EPOCHS, PEAK_LR, MIN_LR, WARMUP_FRAC = 30, 3e-4, 3e-5, 0.05
ALIGN_WEIGHT, TOKEN_SPACE_WEIGHT, WEIGHT_DECAY = 0.35, 0.03, 0.03
PROBE_INDICES = (69, 23, 127, 107, 137, 21, 94, 17)
CLINICAL_BAD_WORDS = [
    "schizophrenia", "hallucination", "hallucinations", "Alzheimer", "Parkinson", "depression", "bipolar",
    "autism", "ADHD", "patients", "patient", "disorder", "disease", "syndrome", "clinical", "treatment",
]


def clean_space(x):
    return re.sub(r"\s+", " ", str(x).replace("\n", " ").strip())


def load_rows(args, latent_path: Path | None = None):
    """Cell 2: summaries aligned by pmid to the brain latents (run latent_neuro.pt, or HF pubmed_images) + split flags."""
    import torch
    import pandas as pd
    from neurovlm.data import load_dataset, load_latent

    pubs = load_dataset("pubmed_text").sort_values(by="pmid").reset_index(drop=True)
    # the pinned parquet (datasets.load_dataset would resolve the repo's main branch)
    summary = pd.read_parquet(hf_pinned("neurovlm/pubmed_summary_qa", "pubmed_summary_qa.parquet"))
    summary = summary.rename(columns={"title": "name", "summary": "description"}).sort_values(by="pmid").reset_index(drop=True)

    if latent_path is not None:
        payload = load_torch(latent_path)
        latents, pmids = payload["latent"], payload["pmid"]
    else:
        latents, pmids = load_latent("pubmed_images")
    latents = torch.as_tensor(latents, dtype=torch.float32).cpu()
    pmids = np.asarray(pmids)
    shared = np.intersect1d(summary["pmid"].to_numpy(), pmids)
    df_pubs = summary[summary["pmid"].isin(shared)].sort_values("pmid").reset_index(drop=True)
    lookup = {pmid: i for i, pmid in enumerate(pmids)}
    image_idx = np.array([lookup[pmid] for pmid in df_pubs["pmid"].to_numpy()])
    latent_images = latents[torch.as_tensor(image_idx, dtype=torch.long)]
    pmids_images = pmids[image_idx]
    split_meta = pubs[pubs["pmid"].isin(df_pubs["pmid"])].sort_values("pmid").reset_index(drop=True)
    assert (df_pubs["pmid"].to_numpy() == split_meta["pmid"].to_numpy()).all()
    df_pubs[["train", "test", "val"]] = split_meta[["train", "test", "val"]]

    if args.max_samples and args.max_samples < len(df_pubs):
        n = args.max_samples
        keep = np.concatenate([np.flatnonzero(df_pubs[s].to_numpy(dtype=bool))[:k]
                               for s, k in (("train", max(4, int(n * 0.75))), ("val", max(2, n // 8)), ("test", max(2, n // 8)))])
        keep = np.sort(keep)
        df_pubs = df_pubs.iloc[keep].reset_index(drop=True)
        latent_images = latent_images[torch.as_tensor(keep)]
        pmids_images = pmids_images[keep]
    train_mask = df_pubs["train"].to_numpy(dtype=bool)
    val_mask = df_pubs["val"].to_numpy(dtype=bool)
    test_mask = df_pubs["test"].to_numpy(dtype=bool)
    log.info("aligned dataset: %d rows | train %d | val %d | test %d", len(df_pubs), train_mask.sum(), val_mask.sum(), test_mask.sum())
    return df_pubs, latent_images, pmids_images, train_mask, val_mask, test_mask


def load_heads(args, run_dir, device):
    """InfoNCE heads from the run (09), else the released ones."""
    from neurovlm.models import ProjHead

    text_path = image_path = None
    if args.heads != "released":
        # --inputs-from or the run dir; data_dir copies are never used
        text_path = run_input("proj_head_text_infonce.pt", run_dir=run_dir, args=args)
        image_path = run_input("proj_head_image_infonce.pt", run_dir=run_dir, args=args)
        if args.heads == "run" and (text_path is None or image_path is None):
            raise FileNotFoundError("--heads run requires proj_head_{text,image}_infonce.pt in the run directory")
    if text_path is not None and image_path is not None:
        text_head = load_module_or_state(text_path, builder=lambda: ProjHead(768, 512, 384), device=device)
        image_head = load_module_or_state(image_path, builder=lambda: ProjHead(384, 384, 384), device=device)
        source = {"text": str(text_path), "image": str(image_path)}
    else:
        text_head = released_model("proj_head_text_infonce", device)
        image_head = released_model("proj_head_image_infonce", device)
        rel = f"released:neurovlm/ProjectionHeads@{hf_pinned('neurovlm/ProjectionHeads')[:8]}"
        source = {"text": rel, "image": rel}
    log.info("projection heads: %s", source)
    return text_head.eval(), image_head.eval(), source


def resolve_latents(args, head_source):
    """(latent_neuro.pt, autoencoder.pt) next to run heads, or (None, None) for the HF latents of the released autoencoder.

    The latents must come from the autoencoder the image head was trained on.
    """
    image_head = head_source["image"]
    run_heads = not image_head.startswith("released:")
    mode = args.latents if args.latents != "auto" else ("run" if run_heads else "hf")
    if mode == "hf":
        if run_heads:
            log.warning("--latents hf with run heads (%s): the HF latents come from the released autoencoder", image_head)
        log.info("brain latents: HF pubmed_images (released autoencoder)")
        return None, None
    if not run_heads:
        raise ValueError("--latents run needs run-directory heads (--heads run/auto with 09 outputs)")
    head_dir = Path(image_head).parent
    latent_path, ae_path = head_dir / "latent_neuro.pt", head_dir / "autoencoder.pt"
    if not latent_path.exists():
        raise FileNotFoundError(f"latent_neuro.pt (08) must sit next to the run heads {image_head}; rerun 08 or pass --heads released")
    ae_path = ae_path if ae_path.exists() else None
    log.info("brain latents: %s (autoencoder %s)", latent_path, ae_path or "missing")
    return latent_path, ae_path


def quality_weights(df_pubs, latent_images, pmids_images, train_mask, proj_head_text, proj_head_image, device):
    """Cell 4: permissive canonical-quality scores -> lm/align weights (no rows dropped); returns the image/text semantics."""
    import torch
    import torch.nn.functional as F
    from neurovlm.data import load_latent

    latent_text_raw, pmids_text_raw = load_latent("pubmed_text")
    latent_text_raw = torch.as_tensor(latent_text_raw, dtype=torch.float32).cpu()
    pmids_text_raw = np.asarray(pmids_text_raw)
    assert np.all(np.sort(pmids_text_raw) == pmids_text_raw)
    text_lookup = {pmid: i for i, pmid in enumerate(pmids_text_raw)}
    missing = [pmid for pmid in pmids_images if pmid not in text_lookup]
    assert not missing, f"Missing {len(missing)} PMIDs from pubmed_text latents; first few: {missing[:5]}"
    text_idx = np.array([text_lookup[pmid] for pmid in pmids_images])
    latent_text = latent_text_raw[torch.as_tensor(text_idx, dtype=torch.long)]

    @torch.no_grad()
    def project_batched(head, x, batch_size=CANONICAL_PROJ_BATCH):
        outs = [head(x[s:s + batch_size].to(device)).detach().float().cpu() for s in range(0, len(x), batch_size)]
        return F.normalize(torch.cat(outs, dim=0), dim=1)

    @torch.no_grad()
    def topk_cross(query, key, k, batch_size=CANONICAL_SIM_BATCH):
        key_gpu = key.to(device)
        vals_all, idx_all = [], []
        for start in range(0, len(query), batch_size):
            sim = query[start:start + batch_size].to(device) @ key_gpu.T
            vals, idx = sim.topk(k, dim=1)
            vals_all.append(vals.cpu())
            idx_all.append(idx.cpu())
        return torch.cat(vals_all, dim=0), torch.cat(idx_all, dim=0)

    @torch.no_grad()
    def topk_self_without_self(x, k, batch_size=CANONICAL_SIM_BATCH):
        x_gpu = x.to(device)
        idx_all = []
        for start in range(0, len(x), batch_size):
            end = min(start + batch_size, len(x))
            sim = x[start:end].to(device) @ x_gpu.T
            sim[torch.arange(end - start, device=device), torch.arange(start, end, device=device)] = -float("inf")
            idx_all.append(sim.topk(min(k, sim.shape[1] - 1), dim=1).indices.cpu())
        return torch.cat(idx_all, dim=0)

    text_emb_all = project_batched(proj_head_text, latent_text)
    image_emb_all = project_batched(proj_head_image, latent_images)
    retrieved_text_sim, retrieved_text_idx = topk_cross(image_emb_all, text_emb_all, min(CANONICAL_TOP_K_TEXT, len(text_emb_all)))
    image_neighbor_idx = topk_self_without_self(image_emb_all, CANONICAL_TOP_K_IMAGE)
    true_text = text_emb_all.unsqueeze(1)
    retrieved_text_consistency = (true_text * text_emb_all[retrieved_text_idx]).sum(dim=-1).mean(dim=1)
    image_neighbor_text_consistency = (true_text * text_emb_all[image_neighbor_idx]).sum(dim=-1).mean(dim=1)
    paired_image_text_score = (image_emb_all * text_emb_all).sum(dim=1)
    top_retrieved_score = retrieved_text_sim[:, 0]
    canonical_score = (0.35 * retrieved_text_consistency + 0.35 * image_neighbor_text_consistency
                       + 0.20 * paired_image_text_score + 0.10 * top_retrieved_score)
    filter_text = df_pubs["name"].fillna("") + " " + df_pubs["description"].fillna("")
    clinical_mask = filter_text.str.contains(CLINICAL_PATTERN, regex=True, na=False).to_numpy()
    artifact_mask = filter_text.str.contains(r"\[[0-9JjIi]+\]|\bPMID\b|\bdoi\b", regex=True, na=False).to_numpy()
    train_scores = canonical_score[torch.as_tensor(train_mask)]
    score_lo, score_hi = torch.quantile(train_scores, 0.05), torch.quantile(train_scores, 0.95)
    quality = ((canonical_score - score_lo) / (score_hi - score_lo).clamp_min(1e-6)).clamp(0, 1).numpy()
    report_threshold = torch.quantile(train_scores, 1.0 - CANONICAL_KEEP_FRAC_FOR_REPORTING)
    lm_weights = MIN_LM_WEIGHT + (1.0 - MIN_LM_WEIGHT) * quality
    align_weights = MIN_ALIGN_WEIGHT + (1.0 - MIN_ALIGN_WEIGHT) * quality
    lm_weights[clinical_mask] *= CLINICAL_LM_MULT
    align_weights[clinical_mask] *= CLINICAL_ALIGN_MULT
    lm_weights[artifact_mask] *= ARTIFACT_LM_MULT
    align_weights[artifact_mask] *= ARTIFACT_ALIGN_MULT
    lm_weights = np.clip(lm_weights, MIN_LM_WEIGHT, 1.0).astype("float32")
    align_weights = np.clip(align_weights, MIN_ALIGN_WEIGHT, 1.0).astype("float32")
    df_pubs["canonical_score"] = canonical_score.numpy()
    df_pubs["canonical_quality"] = quality
    df_pubs["strict_like"] = canonical_score.numpy() >= float(report_threshold)
    df_pubs["clinical_like"] = clinical_mask
    df_pubs["artifact_like"] = artifact_mask
    df_pubs["lm_weight"] = lm_weights
    df_pubs["align_weight"] = align_weights
    log.info("permissive scoring: kept all %d rows | clinical-like %d | artifact-like %d | lm weight mean %.3f | align weight mean %.3f",
             len(df_pubs), clinical_mask.sum(), artifact_mask.sum(), lm_weights.mean(), align_weights.mean())
    return image_emb_all, text_emb_all


def tokenize_cached(texts, tokenizer, *, max_tokens, eos_id, cache_dir: Path):
    """Cell 8 ``set_texts``, run once and cached on disk."""
    import torch

    key = hashlib.sha256(("\n\x00".join(texts) + f"|{max_tokens}|{eos_id}|{tokenizer.name_or_path}").encode()).hexdigest()[:16]
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"tokens-{key}.pt"
    if path.exists():
        payload = torch.load(path, weights_only=True)
        log.info("tokens from cache %s", path)
        return payload["ids"], payload["lengths"]
    enc = tokenizer(list(texts), truncation=True, max_length=max_tokens - 1, add_special_tokens=False)
    ids = []
    for row in enc["input_ids"]:
        row = list(row)
        if len(row) == 0 or row[-1] != eos_id:
            row.append(eos_id)
        ids.append(torch.tensor(row[:max_tokens], dtype=torch.long))
    lengths = torch.tensor([len(r) for r in ids], dtype=torch.long)
    torch.save({"ids": ids, "lengths": lengths}, path)
    log.info("tokenised %d rows -> %s", len(ids), path)
    return ids, lengths


class GroundedRows:
    """Rows of cell 8: raw latents, semantic image/text vectors, weights, token ids."""

    def __init__(self, raw, sem_img, sem_txt, lm_w, align_w, ids, lengths, pad_id):
        import torch

        self.raw, self.sem_img, self.sem_txt = raw.float(), sem_img.float(), sem_txt.float()
        self.lm_w, self.align_w = torch.as_tensor(lm_w, dtype=torch.float32), torch.as_tensor(align_w, dtype=torch.float32)
        self.ids, self.lengths, self.pad_id = ids, lengths, pad_id

    def __len__(self):
        return len(self.ids)

    def collate(self, idx):
        """Cell 8 ``collate_fn`` for a list of row indices."""
        import torch
        from torch.nn.utils.rnn import pad_sequence

        idx_t = torch.as_tensor(idx, dtype=torch.long)
        lengths = self.lengths[idx_t]
        input_ids = pad_sequence([self.ids[i] for i in idx], batch_first=True, padding_value=self.pad_id)
        attn_mask = (torch.arange(input_ids.size(1))[None, :] < lengths[:, None]).long()
        return (self.raw[idx_t], self.sem_img[idx_t], self.sem_txt[idx_t], self.lm_w[idx_t], self.align_w[idx_t], input_ids, attn_mask)


def epoch_batches(rows: GroundedRows, *, batch_size, shuffle, seed, drop_last, bucket):
    """Batch index lists. bucket: shuffle, sort by length within chunks of 50 batches, then shuffle the batch order."""
    import torch

    n = len(rows)
    if not shuffle:
        order = torch.arange(n)
        return [order[s:s + batch_size].tolist() for s in range(0, n, batch_size)]
    g = torch.Generator().manual_seed(seed)
    order = torch.randperm(n, generator=g)
    if bucket:
        chunks = [order[s:s + batch_size * 50] for s in range(0, n, batch_size * 50)]
        order = torch.cat([c[torch.argsort(rows.lengths[c], descending=True, stable=True)] for c in chunks])
    batches = [order[s:s + batch_size].tolist() for s in range(0, n, batch_size)]
    if drop_last and len(batches) > 1 and len(batches[-1]) < batch_size:
        batches = batches[:-1]
    if bucket:
        perm = torch.randperm(len(batches), generator=g).tolist()
        batches = [batches[i] for i in perm]
    return batches


def load_probe_networks(autoencoder, device, limit=None):
    """Cell 7: latents of the atlas networks used as caption probes."""
    import nibabel as nib
    import torch
    from nilearn.image import resample_to_img
    from neurovlm.data import load_dataset, load_masker

    masker = load_masker()
    networks = load_dataset("networks")
    network_imgs = [(k, a, nib.Nifti1Image(networks[k][a]["array"], affine=networks[k][a]["affine"]))
                    for k in networks for a in networks[k]]
    network_imgs = [x for x in network_imgs if x[0] not in ("UKBICA", "HCPICA")]
    network_imgs = [x for i, x in enumerate(network_imgs) if i in PROBE_INDICES]
    if limit:
        network_imgs = network_imgs[:limit]
    latents, labels = [], []
    autoencoder = autoencoder.to(device).eval()
    for k, a, img in network_imgs:
        img_re = resample_to_img(img, masker.mask_img, interpolation="nearest", force_resample=False, copy_header=True)
        v = torch.from_numpy(masker.transform(img_re)).float().to(device)
        with torch.no_grad():
            latents.append(autoencoder.encoder(v).detach().float().cpu().reshape(-1))
        labels.append(f"{k}_{a}")
    log.info("probe networks: %s", labels)
    return torch.stack(latents), labels


def main(argv=None):
    p = base_parser("Stage 13: train the stage 1 brain-to-text projector (grounded, PubMed).", stage=STAGE)
    add_model_args(p, amp_default="bf16")
    p.add_argument("--heads", choices=["auto", "run", "released"], default="auto", help="InfoNCE heads for the semantic inputs/weights.")
    p.add_argument("--latents", choices=["auto", "run", "hf"], default="auto",
                   help="Brain latents: auto = the run's latent_neuro.pt with run heads, else HF pubmed_images; hf = always HF.")
    p.add_argument("--max-tokens", type=int, default=MAX_TOKENS)
    p.add_argument("--eval-batch-size", type=int, default=EVAL_BATCH, help="Validation batch size (notebook: 32).")
    p.add_argument("--grad-ckpt", choices=["auto", "on", "off"], default="auto", help="LM gradient checkpointing (auto: on below 24 GB free).")
    p.add_argument("--no-bucket", action="store_true", help="Disable length bucketing of training batches.")
    p.add_argument("--probe-every", type=int, default=5, help="Atlas-probe captions every N epochs (0 = off).")
    p.add_argument("--no-save-every-epoch", dest="save_every_epoch", action="store_false", help="Only keep best/final checkpoints.")
    p.add_argument("--log-every", type=int, default=100, help="Training steps between progress log lines.")
    add_projector_args(p)
    p.add_argument("--package-checkpoint", choices=["auto", "final", "best"], default="auto",
                   help="Checkpoint packaged into neuro_qformer_pubmed/: auto = final for qformer, best for mlp.")
    p.add_argument("--early-stop-patience", type=int, default=None,
                   help="Stop after N epochs without val improvement (default: 3 for mlp, 0 = off for qformer).")
    p.add_argument("--min-epochs", type=int, default=5, help="No early stop before this many epochs.")
    p.set_defaults(seed=SEED)  # init, dropout, batch order
    args = finalize_args(p.parse_args(argv))
    run_dir = resolve_run_dir(args)
    setup_torch(args.seed, args.deterministic)
    import torch
    import torch.nn.functional as F
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from neurovlm.models import ProjHead
    from neurovlm.models.base import NeuroAutoEncoder
    from neurovlm.models.qformer import NeuroQFormer, QFormer
    from neurovlm.resources.loaders import NEURO_QWEN_REPO_ID

    device = select_device(args.device)
    if device.type != "cuda":
        log.warning("no CUDA: running the LM in fp32 on %s (the notebook recipe is bf16 on CUDA)", device)
    epochs = args.epochs or EPOCHS
    train_batch = args.batch_size or TRAIN_BATCH
    bucket = not args.no_bucket
    meta = dict(stage=STAGE, smoke=args.smoke, run=args.run_name)
    best_path = run_dir / "qformer_grounded_permissive_neuro_best.pt"
    final_path = run_dir / f"qformer_grounded_permissive_neuro_epoch-{epochs - 1}.pt"
    pkg_dir = run_dir / "neuro_qformer_pubmed"
    package_choice = args.package_checkpoint if args.package_checkpoint != "auto" else ("final" if args.projector == "qformer" else "best")
    patience = args.early_stop_patience if args.early_stop_patience is not None else (3 if args.projector == "mlp" else 0)
    summary_path = run_dir / f"{STAGE}.summary.json"
    # after an early stop the last checkpoint is an earlier epoch
    done_final = Path(read_json(summary_path).get("final_checkpoint", final_path)) if summary_path.exists() else final_path
    if artifacts_exist(best_path, done_final, pkg_dir / "model.safetensors", force=args.force):
        existing = package_projector_kind(pkg_dir)
        if existing != args.projector:
            raise RuntimeError(f"{pkg_dir} holds a {existing!r} projector but --projector {args.projector} was requested; "
                               "pass --force to retrain in place or use another --run-name")
        log.info("artifacts exist in %s; skipping (use --force to retrain)", run_dir)
        return 0

    # cells 2-5
    with timed("load_rows", log, out=args.out, **meta):
        proj_head_text, proj_head_image, head_source = load_heads(args, run_dir, device)
        latent_path, latent_ae_path = resolve_latents(args, head_source)
        df_pubs, latent_images, pmids_images, train_mask, val_mask, test_mask = load_rows(args, latent_path)
    with timed("quality_weights", log, out=args.out, **meta):
        image_semantic, text_semantic = quality_weights(df_pubs, latent_images, pmids_images, train_mask, proj_head_text, proj_head_image, device)
        text_full = [f"{clean_space(n)}\n{clean_space(d)}" for n, d in zip(df_pubs["name"], df_pubs["description"])]
        assert not any("[SEP]" in x for x in text_full)

    # cell 6: frozen LM
    with timed("load_lm", log, out=args.out, **meta):
        lm_dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
        lm_rev = hf_pinned(NEURO_QWEN_REPO_ID)
        tokenizer = AutoTokenizer.from_pretrained(NEURO_QWEN_REPO_ID, revision=lm_rev, use_fast=True)
        model = AutoModelForCausalLM.from_pretrained(NEURO_QWEN_REPO_ID, revision=lm_rev, torch_dtype=lm_dtype,
                                                     attn_implementation="sdpa").to(device)
        lm_dim = model.config.hidden_size
        PAD_ID = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
        EOS_ID = tokenizer.eos_token_id
        for prm in model.parameters():
            prm.requires_grad = False
        lm_sdpa_backends(args.deterministic, log)
        free_gb = torch.cuda.mem_get_info(device)[0] / 2**30 if device.type == "cuda" else 0.0
        grad_ckpt = {"on": True, "off": False, "auto": free_gb < 24}[args.grad_ckpt]
        if grad_ckpt:
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        log.info("LM %s hidden=%d dtype=%s grad_ckpt=%s (free %.1f GB)", NEURO_QWEN_REPO_ID, lm_dim, lm_dtype, grad_ckpt, free_gb)
        with torch.no_grad():
            emb = model.get_input_embeddings().weight.detach().float()
            LM_EMB_STD = emb.std().item()
            LM_EMB_NORM = emb.norm(dim=1).mean().item()
            LM_EMB_CENTER = emb.mean(0).cpu()  # MLP output LayerNorm bias init
            del emb

    # cell 7: probe networks, encoded by the autoencoder of the training latents
    probe_latents, probe_labels = None, []
    if args.probe_every > 0:
        with timed("probe_setup", log, out=args.out, **meta):
            autoencoder = (load_module_or_state(latent_ae_path, builder=lambda: NeuroAutoEncoder(seed=0, out="logit"), device=device)
                           if latent_ae_path is not None else released_model("autoencoder", device))
            probe_latents, probe_labels = load_probe_networks(autoencoder, device, limit=1 if args.smoke else None)
            del autoencoder

    # cell 8: tokens + rows
    with timed("tokenize", log, out=args.out, **meta):
        ids, lengths = tokenize_cached(text_full, tokenizer, max_tokens=args.max_tokens, eos_id=EOS_ID, cache_dir=run_dir / "cache")
        lm_w, align_w = df_pubs["lm_weight"].to_numpy(dtype="float32"), df_pubs["align_weight"].to_numpy(dtype="float32")

        def subset(mask):
            idx = np.flatnonzero(mask)
            return GroundedRows(latent_images[idx], image_semantic[idx], text_semantic[idx], lm_w[idx], align_w[idx],
                                [ids[i] for i in idx], lengths[idx], PAD_ID)

        train_rows, val_rows, test_rows = subset(train_mask), subset(val_mask), subset(test_mask)
        n_train_batches = len(epoch_batches(train_rows, batch_size=train_batch, shuffle=True, seed=0, drop_last=True, bucket=bucket))
        if args.smoke:
            n_train_batches = min(n_train_batches, args.caps.max_batches)
        log.info("train %d | val %d | test %d | batches/epoch %d | max tokens %d | bucket %s",
                 len(train_rows), len(val_rows), len(test_rows), n_train_batches, args.max_tokens, bucket)

    # cells 9-10: projector + optimiser (``qformer`` is the projector of either kind)
    if args.projector == "qformer":
        qformer = QFormer(image_dim=latent_images.shape[1], semantic_dim=image_semantic.shape[1], lm_dim=lm_dim,
                          num_queries=32, hidden_dim=512, num_heads=8, num_layers=6).to(device).to(lm_dtype)
    else:  # A5.1 residual MLP: seeded init, LM-scale output LayerNorm, fp32 weights under the bf16 autocast
        seed_all(args.seed)
        qformer = build_projector("mlp", image_dim=latent_images.shape[1], semantic_dim=image_semantic.shape[1], lm_dim=lm_dim, num_queries=32,
                                  width=args.mlp_width, depth=args.mlp_depth, token_scale=LM_EMB_NORM / math.sqrt(lm_dim),
                                  token_center=LM_EMB_CENTER).to(device)
    projector_cfg = projector_config(qformer)
    arch_cfg = {k: v for k, v in projector_cfg.items() if k not in ("class", "params")}
    log.info("projector=%s params: %s | %s | package checkpoint: %s", args.projector, f"{projector_cfg['params']:,}", arch_cfg, package_choice)
    total_steps = epochs * n_train_batches
    warmup_steps = max(1, int(WARMUP_FRAC * total_steps))
    optimizer = torch.optim.AdamW(qformer.parameters(), lr=PEAK_LR, weight_decay=WEIGHT_DECAY, fused=(device.type == "cuda"))

    def lr_lambda(step):
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        min_scale = MIN_LR / PEAK_LR
        return min_scale + (1.0 - min_scale) * cosine

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    # early stopping keeps the schedule of the full --epochs run
    log.info("%s projector | epochs=%d total_steps=%d warmup=%d lr=%g->%g | early stop: %s",
             args.projector, epochs, total_steps, warmup_steps, PEAK_LR, MIN_LR,
             f"patience {patience}, min epochs {args.min_epochs}" if patience > 0 else "off")

    # cell 12: losses
    def match_lm_token_norm(vis):
        target_norm = torch.tensor(LM_EMB_NORM, device=vis.device, dtype=vis.dtype)
        vis_norm = vis.float().norm(dim=-1, keepdim=True).clamp_min(1e-6).to(vis.dtype)
        return vis * (target_norm / vis_norm)

    def token_space_reg(vis):
        vf = vis.float()
        reg_norm = ((vf.norm(dim=-1).mean() - LM_EMB_NORM) / max(LM_EMB_NORM, 1e-6)).pow(2)
        reg_std = ((vf.std() - LM_EMB_STD) / max(LM_EMB_STD, 1e-6)).pow(2)
        return reg_norm + 0.25 * reg_std

    def weighted_mean(values, weights):
        return (values * weights).sum() / weights.sum().clamp_min(1e-6)

    def grounded_loss(batch, image_mode="correct"):
        raw, sem_img, sem_txt, lm_w_b, align_w_b, input_ids, attn_mask = (t.to(device, non_blocking=True) for t in batch)
        sem_txt = F.normalize(sem_txt.float(), dim=1)
        if image_mode == "shuffled":
            perm = torch.randperm(raw.size(0), device=device)
            raw, sem_img = raw[perm], sem_img[perm]
        elif image_mode == "zero":
            raw, sem_img = torch.zeros_like(raw), torch.zeros_like(sem_img)
        batch_size = raw.size(0)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=(device.type == "cuda" and args.amp == "bf16")):
            vis_raw = qformer(raw.to(lm_dtype), sem_img.to(lm_dtype))
            vis = match_lm_token_norm(vis_raw)
            num_q = vis.size(1)
            with torch.no_grad():
                txt = model.get_input_embeddings()(input_ids)
            embeds = torch.cat([vis.to(model.dtype), txt.to(model.dtype)], dim=1)
            full_mask = torch.cat([torch.ones(batch_size, num_q, device=device, dtype=torch.long), attn_mask], dim=1)
            labels = torch.cat([torch.full((batch_size, num_q), -100, dtype=torch.long, device=device),
                                input_ids.masked_fill(attn_mask == 0, -100)], dim=1)
            out = model(inputs_embeds=embeds, attention_mask=full_mask, use_cache=False)
            shift_logits = out.logits[:, :-1].float().contiguous()
            shift_labels = labels[:, 1:].contiguous()
            valid = shift_labels.ne(-100)
            safe_labels = shift_labels.masked_fill(~valid, 0)
            token_loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), safe_labels.view(-1), reduction="none").view_as(safe_labels)
            per_sample_lm = (token_loss * valid.float()).sum(dim=1) / valid.float().sum(dim=1).clamp_min(1.0)
            lm = weighted_mean(per_sample_lm, lm_w_b.float())
            pred_sem = qformer.semantic_from_visual(vis_raw)
            per_sample_align = 1.0 - (pred_sem * sem_txt).sum(dim=1)
            align = weighted_mean(per_sample_align, align_w_b.float())
            reg = token_space_reg(vis_raw)
            loss = lm + ALIGN_WEIGHT * align + TOKEN_SPACE_WEIGHT * reg
        return loss, {"lm": lm.detach(), "align": align.detach(), "reg": reg.detach()}

    @torch.no_grad()
    def eval_rows(rows, image_mode="correct", max_batches=None):
        qformer.eval()
        model.eval()
        total = {k: torch.zeros((), device=device) for k in ("loss", "lm", "align", "reg")}
        n = 0
        for idx in epoch_batches(rows, batch_size=args.eval_batch_size, shuffle=False, seed=0, drop_last=False, bucket=False):
            loss, parts = grounded_loss(rows.collate(idx), image_mode=image_mode)
            total["loss"] += loss.detach()
            for k in parts:
                total[k] += parts[k]
            n += 1
            if max_batches is not None and n >= max_batches:
                break
        return {k: float(v) / max(n, 1) for k, v in total.items()}

    # cell 13: generation probe
    bad_words_ids = []
    for term in CLINICAL_BAD_WORDS:
        for t in (term, " " + term):
            term_ids = tokenizer(t, add_special_tokens=False).input_ids
            if term_ids:
                bad_words_ids.append(term_ids)

    @torch.no_grad()
    def generate_caption(raw_img, sem_img, max_new_tokens):
        qformer.eval()
        model.eval()
        raw_img = raw_img.reshape(1, -1).to(device=device, dtype=lm_dtype)
        sem_img = sem_img.reshape(1, -1).to(device=device, dtype=lm_dtype)
        vis = match_lm_token_norm(qformer(raw_img, sem_img)).to(model.dtype)
        attn = torch.ones(vis.shape[:2], dtype=torch.long, device=device)
        out_ids = model.generate(inputs_embeds=vis, attention_mask=attn, max_new_tokens=max_new_tokens, num_beams=3,
                                 do_sample=False, repetition_penalty=1.18, no_repeat_ngram_size=4, eos_token_id=tokenizer.eos_token_id,
                                 pad_token_id=tokenizer.eos_token_id, bad_words_ids=bad_words_ids)
        return tokenizer.decode(out_ids[0], skip_special_tokens=True).strip()

    probe_log = (run_dir / "probes.jsonl").open("a")

    def atlas_probe(epoch, max_new_tokens):
        if probe_latents is None:
            return
        with torch.no_grad():
            atlas_sem = F.normalize(proj_head_image(probe_latents.to(device)).float(), dim=1)
        for i, lab in enumerate(probe_labels):
            pred = generate_caption(probe_latents[i], atlas_sem[i], max_new_tokens)
            probe_log.write(json.dumps({"epoch": epoch, "network": lab, "caption": pred}) + "\n")
        probe_log.flush()

    # cell 14: training loop
    history = {"train": [], "train_lm": [], "train_align": [], "val": [], "val_lm": [], "val_align": [], "shuffle": [], "zero": [], "lr": [], "epoch_seconds": []}
    best_val, best_epoch, last_epoch, stopped_epoch = float("inf"), -1, -1, None
    smoke_eval = args.caps.max_batches if args.smoke else None
    val = eval_rows(val_rows, "correct", max_batches=smoke_eval or 50)
    shuf = eval_rows(val_rows, "shuffled", max_batches=smoke_eval or 50)
    zero = eval_rows(val_rows, "zero", max_batches=smoke_eval or 50)
    log.info("initial | val %.4f lm %.4f align %.4f | shuffled gap %+.4f | zero gap %+.4f", val["loss"], val["lm"], val["align"], shuf["loss"] - val["loss"], zero["loss"] - val["loss"])
    atlas_probe(-1, 16 if args.smoke else 96)

    step_times = []
    with timed("train", log, out=args.out, epochs=epochs, batch_size=train_batch, grad_ckpt=grad_ckpt, bucket=bucket, amp=args.amp, **meta) as t_train:
        for epoch in range(epochs):
            t_epoch = time.perf_counter()
            qformer.train()
            model.train()
            totals = {k: torch.zeros((), device=device) for k in ("loss", "lm", "align", "reg")}
            n_train = 0
            batches = epoch_batches(train_rows, batch_size=train_batch, shuffle=True, seed=args.seed + epoch, drop_last=True, bucket=bucket)
            if args.smoke:
                batches = batches[: args.caps.max_batches]
            for bi, idx in enumerate(batches):
                t0 = time.perf_counter()
                optimizer.zero_grad(set_to_none=True)
                loss, parts = grounded_loss(train_rows.collate(idx), image_mode="correct")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(qformer.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                totals["loss"] += loss.detach()
                for k in parts:
                    totals[k] += parts[k]
                n_train += 1
                if bi < 3 or (bi + 1) % args.log_every == 0:
                    if device.type == "cuda":
                        torch.cuda.synchronize()
                    step_times.append(time.perf_counter() - t0)
                    log.info("epoch %d step %d/%d loss %.4f lr %.2e (%.3fs/step)", epoch + 1, bi + 1, len(batches), float(loss), scheduler.get_last_lr()[0], step_times[-1])
            train_loss = {k: float(v) / max(n_train, 1) for k, v in totals.items()}
            val = eval_rows(val_rows, "correct", max_batches=smoke_eval)
            shuf = eval_rows(val_rows, "shuffled", max_batches=smoke_eval or 100)
            zero = eval_rows(val_rows, "zero", max_batches=smoke_eval or 100)
            lr = scheduler.get_last_lr()[0]
            for k, v in (("train", train_loss["loss"]), ("train_lm", train_loss["lm"]), ("train_align", train_loss["align"]),
                         ("val", val["loss"]), ("val_lm", val["lm"]), ("val_align", val["align"]), ("shuffle", shuf["loss"]),
                         ("zero", zero["loss"]), ("lr", lr), ("epoch_seconds", time.perf_counter() - t_epoch)):
                history[k].append(v)
            state = {k: v.detach().cpu() for k, v in qformer.state_dict().items()}
            improved = val["loss"] < best_val
            if improved:
                best_val, best_epoch = val["loss"], epoch
            last_epoch = epoch
            stop = (patience > 0 and epoch < epochs - 1 and epoch + 1 >= args.min_epochs and epoch - best_epoch >= patience)
            if args.save_every_epoch or epoch == epochs - 1 or stop:
                torch.save(state, run_dir / f"qformer_grounded_permissive_neuro_epoch-{epoch}.pt")
            if improved:
                torch.save(state, best_path)
                torch.save(state, run_dir / f"qformer_grounded_permissive_neuro_best_epoch-{epoch}.pt")
            log.info("Epoch %3d | train %.4f lm %.4f align %.4f | val %.4f lm %.4f align %.4f | shuffle gap %+.4f zero gap %+.4f | lr %.2e | %.0fs",
                     epoch + 1, train_loss["loss"], train_loss["lm"], train_loss["align"], val["loss"], val["lm"], val["align"],
                     shuf["loss"] - val["loss"], zero["loss"] - val["loss"], lr, history["epoch_seconds"][-1])
            write_json(run_dir / "history_13_qformer_pubmed.json", history)
            if args.probe_every > 0 and epoch % args.probe_every == 0:
                atlas_probe(epoch, 16 if args.smoke else 128)
            if stop:
                stopped_epoch = epoch
                log.info("early stop after epoch %d: no val improvement for %d epochs (best %.4f at epoch %d)",
                         epoch + 1, epoch - best_epoch, best_val, best_epoch + 1)
                break

    epochs_trained = last_epoch + 1
    final_path = run_dir / f"qformer_grounded_permissive_neuro_epoch-{last_epoch}.pt"
    if best_epoch < 0:
        raise RuntimeError("No grounded best checkpoint was produced (validation loss never improved).")
    qformer.load_state_dict(torch.load(best_path, map_location=device, weights_only=True), strict=True)
    qformer.eval()
    log.info("best epoch=%d val=%.4f -> %s", best_epoch, best_val, best_path)
    atlas_probe(epochs_trained, 16 if args.smoke else 128)
    probe_log.close()
    test = eval_rows(test_rows, "correct", max_batches=smoke_eval)

    # package like neurovlm/NeuroQformer-PubMed (fp32)
    package_src = final_path if package_choice == "final" else best_path
    package_state = torch.load(package_src, map_location="cpu", weights_only=True)
    q_fp32 = build_projector(args.projector, **arch_cfg)
    q_fp32.load_state_dict({k: v.float() for k, v in package_state.items()}, strict=True)
    packaged = NeuroQFormer(qformer=q_fp32, proj_head_image=ProjHead(384, 384, 384), canonical_banks=None, projection_temp=None,
                            canonical_basis="all", use_canonical_projection=False)
    packaged.proj_head_image.load_state_dict(proj_head_image.cpu().state_dict())
    write_projector_package(pkg_dir, packaged, {
        "architectures": ["neurovlm.qformer.NeuroQFormer"], "library_name": "neurovlm", "model_type": "neurovlm_qformer",
        "qformer": {**package_qformer_block(packaged), "projection_temp": None, "use_canonical_projection": False, "canonical_basis": "all"},
        "qformer_variant": "pubmed", "weights": "model.safetensors",
        "training_metadata": {"source_checkpoint": str(package_src), "packaged_checkpoint": package_choice, "best_checkpoint": str(best_path),
                              "best_epoch": best_epoch, "final_checkpoint": str(final_path), "epochs_trained": epochs_trained,
                              "epochs_scheduled": epochs, "early_stop_patience": patience, "seed": args.seed,
                              "image_projection": head_source["image"], "brain_latents": str(latent_path or "hf:pubmed_images"),
                              "source_script": "scripts/models/13_qformer_pubmed.py"},
    })
    log.info("packaged %s (%s projector, %s checkpoint %s; %d of %d epochs trained)",
             pkg_dir, args.projector, package_choice, package_src.name, epochs_trained, epochs)

    summary = {"epochs": epochs, "epochs_trained": epochs_trained, "best_epoch": best_epoch, "best_val_loss": best_val, "test": test,
               "early_stop": {"patience": patience, "min_epochs": args.min_epochs, "stopped": stopped_epoch is not None,
                              "stopped_after_epoch": None if stopped_epoch is None else stopped_epoch + 1},
               "final_checkpoint": str(final_path), "history": history,
               "seconds": t_train.seconds, "sec_per_epoch": t_train.seconds / epochs_trained,
               "sec_per_step": float(np.median(step_times)) if step_times else None, "steps_per_epoch": n_train_batches,
               "n_rows": {"train": len(train_rows), "val": len(val_rows), "test": len(test_rows)},
               "grad_ckpt": grad_ckpt, "bucket": bucket, "amp": args.amp, "max_tokens": args.max_tokens,
               "train_batch": train_batch, "eval_batch": args.eval_batch_size, "heads": head_source,
               "latents": str(latent_path or "hf:pubmed_images"), "device": str(device),
               "projector": args.projector, "projector_config": projector_cfg, "packaged_checkpoint": package_choice,
               "packaged_source": str(package_src), "seed": args.seed, "deterministic": args.deterministic}
    write_stage_summary(run_dir, STAGE, summary)
    if args.install:
        install_artifacts({best_path.name: best_path, final_path.name: final_path}, log)
    return 0


if __name__ == "__main__":
    sys.exit(main())
