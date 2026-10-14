#!/usr/bin/env python
"""Continued pretraining (DAPT) of Qwen/Qwen3-0.6B on PubMed titles + abstracts: NeuroQwen3-0.6B, without the papers
NeuroVLM is evaluated on.

Port of efficient_ai/project/notebooks/01_dapt.ipynb (the recipe on the neurovlm/NeuroQwen3-0.6B model card). Documents
are "<title>\\n\\n<abstract><|im_end|>", packed into 1,024-token sequences the way TRL's "wrapped" packing does (each
batch of 1,000 documents is concatenated and cut every 1,024 tokens), 16 sequences (16,384 tokens) per optimizer step, one
epoch, AdamW lr 5e-6 with 10 % linear warmup and cosine decay, weight decay 0.01, gradient clipping 0.3, 1 % of the
documents held out for validation (seed 12345). A plain loop does what TRL's SFTTrainer did, so no new dependency.
Precision: fp32 master weights and AdamW states with bf16 autocast (the export is bf16). The notebook trained the bf16
weights directly, where most lr-5e-6 updates round away (the published model differs from Qwen3-0.6B in 4.6 % of its
weights); --weights-bf16 restores that.

    .venv/bin/python scripts/models/00_neuroqwen_dapt.py --run-name <run> --corpus <pubmed_neuroimaging.parquet>
    .venv/bin/python scripts/models/00_neuroqwen_dapt.py --smoke     # 1 step on 256 documents, sequence length 128

Writes scripts/outputs/runs/<run>/neuroqwen/: NeuroQwen3-0.6B/ (HF model folder: model.safetensors, config, tokenizer),
eval_filter.json, eval_filter_removed.csv, history.jsonl, export_check.json, 00_neuroqwen_dapt.summary.json.
--resume continues from neuroqwen/checkpoint.pt (written every --save-every steps).
"""
from __future__ import annotations

import functools
import itertools
import json
import math
import os
import re
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))  # scripts/models/
from _common_models import *  # noqa: E402,F403
from common import hf_manifest, hf_pinned_dir, hf_split_pmids, load_eval_exclusions, norm_doi, provenance  # noqa: E402

STAGE = "00_neuroqwen_dapt"
log = get_logger(STAGE)

# notebook cell 2 (H100 run) and the model card
BASE_MODEL = "Qwen/Qwen3-0.6B"
SEED, VAL_FRAC, MAX_LENGTH, EPOCHS, LR = 12345, 0.01, 1024, 1, 5e-6
SEQS_PER_STEP = 16  # per-device batch 8 x gradient accumulation 2
# sequences per forward pass (16 per step either way). Each 1,024-token sequence needs ~4 GB of activations and logits on
# top of weights + grads + AdamW states (0.75 B parameters with the untied head: 12 GB in fp32, 5.6 GB in bf16)
MICRO_BATCH = {"fp32": 2, "bf16": 4}
WARMUP_RATIO, WEIGHT_DECAY, MAX_GRAD_NORM = 0.10, 0.01, 0.30
LOG_EVERY, EVAL_EVERY, SAVE_EVERY = 25, 1000, 1000
DOCS_PER_PACK = 1000  # TRL packs with datasets.map(batched=True), whose default batch is 1,000 documents
SMOKE = {"docs": 256, "max_length": 128, "batch_size": 2, "grad_accum": 2, "max_steps": 1, "lr": 1e-4, "warmup_ratio": 0.0}
CORPUS_NAME = "pubmed_neuroimaging.parquet"
DEFAULT_CORPUS = Path.home() / "efficient_ai" / "project" / "pubmed" / CORPUS_NAME
EXPORT_NAME = "NeuroQwen3-0.6B"
CHECK_PROMPT = "The default mode network"
MIN_TITLE_CHARS = 25


def load_corpus(args) -> tuple:
    """(DataFrame pmid/doi/title/abstract, source record). The corpus is pinned by sha256 in hf_manifest.json (not_on_hub)."""
    import pandas as pd

    path = Path(args.corpus or DEFAULT_CORPUS)
    if path.exists():
        pin = next(e for e in hf_manifest()["not_on_hub"] if e["name"] == CORPUS_NAME)
        digest = sha256_file(path)
        if digest != pin["sha256"] and not args.unpinned_corpus:
            raise SystemExit(f"{path}: sha256 {digest[:12]} is not the pinned {pin['sha256'][:12]} (hf_manifest.json not_on_hub); "
                             "pass --unpinned-corpus to train on it anyway")
        df = pd.read_parquet(path, columns=["pmid", "doi", "title", "abstract"])
        return df, {"path": str(path), "sha256": digest, "pinned": digest == pin["sha256"], "rows": len(df)}
    if not args.smoke:
        raise FileNotFoundError(f"{path} not found; pass --corpus (the {CORPUS_NAME} pinned in hf_manifest.json not_on_hub)")
    # smoke without the corpus: the pinned NeuroVLM papers stand in (name = title, description = abstract)
    src = hf_pinned("neurovlm/neuro_image_papers", "publications.parquet")
    df = pd.read_parquet(src, columns=["pmid", "doi", "name", "description"]).rename(columns={"name": "title", "description": "abstract"})
    log.warning("%s not found; smoke run on the pinned neuro_image_papers/publications.parquet instead", path)
    return df, {"path": str(src), "sha256": sha256_file(src), "pinned": True, "rows": len(df), "smoke_stand_in": True}


# ======================================================================================================================
# EVALUATION FILTER: drop every paper NeuroVLM is evaluated on (--no-eval-filter keeps them; the counts are still written)
#   - the validation and test papers of the pinned neuro_image_papers/publications.parquet (split flags, = pmids_split.pt),
#     by PMID or by title. Their DOIs are not used: some publications.parquet DOIs belong to another paper.
#   - the NeuroVault evaluation papers: the 312 released publications (neurovlm/embedded_text publications_neurovault.parquet)
#     plus scripts/eval_exclusions/neurovault_exclusions.csv (7 of them), by DOI or by title.
#   Titles match after lower-casing and keeping letters and digits only, from 25 characters on.
# ======================================================================================================================
def norm_title(s) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(s).lower()).strip()


def eval_filter(df) -> tuple:
    """(keep mask, removed rows with their rules, counts)."""
    import pandas as pd

    split = hf_split_pmids()
    held = np.concatenate([split["val"], split["test"]])
    pubs = pd.read_parquet(hf_pinned("neurovlm/neuro_image_papers", "publications.parquet"), columns=["pmid", "name"])
    held_titles = pubs.loc[pubs["pmid"].isin(held), "name"].map(norm_title)
    nv = pd.read_parquet(hf_pinned("neurovlm/embedded_text", "publications_neurovault.parquet"), columns=["doi", "title"])
    nv_csv = load_eval_exclusions()["neurovault"]
    nv_dois = {norm_doi(d) for d in pd.concat([nv["doi"], nv_csv["doi"]])} - {None}
    nv_pmids = {int(float(x)) for x in nv_csv["pmid"].dropna()}

    titles, dois = df["title"].map(norm_title), df["doi"].map(norm_doi)
    long_titles = lambda ts: {t for t in ts if len(t) >= MIN_TITLE_CHARS}  # noqa: E731
    rules = {
        "val_test_pmid": df["pmid"].isin(held),
        "val_test_title": titles.isin(long_titles(held_titles)),
        "neurovault_doi": dois.isin(nv_dois) | df["pmid"].isin(nv_pmids),
        "neurovault_title": titles.isin(long_titles(nv["title"].map(norm_title))),
    }
    drop = np.logical_or.reduce([m.to_numpy() for m in rules.values()])
    removed = df.loc[drop, ["pmid", "doi", "title"]].copy()
    removed["rules"] = [";".join(r for r, m in rules.items() if m.iloc[i]) for i in np.flatnonzero(drop)]
    in_corpus = set(df["pmid"])
    counts = {
        "corpus_documents": len(df), "removed": int(drop.sum()), "kept": int((~drop).sum()),
        "removed_by_rule": {r: int(m.sum()) for r, m in rules.items()},
        "test_papers": len(split["test"]), "test_papers_in_corpus_by_pmid": len(in_corpus & set(split["test"].tolist())),
        "val_papers": len(split["val"]), "val_papers_in_corpus_by_pmid": len(in_corpus & set(split["val"].tolist())),
        "neurovault_publications": int(nv["doi"].map(norm_doi).nunique()),
        "neurovault_exclusions_csv": len(nv_csv),
    }
    return ~drop, removed, counts
# ======================================================================================================================


def notebook_texts(df) -> np.ndarray:
    """Notebook cell 5: "<title>\\n\\n<abstract>", or whichever of the two is not empty."""
    out = []
    for title, abstract in zip(df["title"].fillna(""), df["abstract"].fillna("")):
        title, abstract = str(title).replace("\r\n", "\n").strip(), str(abstract).replace("\r\n", "\n").strip()
        out.append(f"{title}\n\n{abstract}" if title and abstract else (title or abstract))
    return np.array(out, dtype=object)


def permutation(n: int, seed: int) -> np.ndarray:
    """The row order of datasets' shuffle / train_test_split(seed=seed)."""
    return np.random.default_rng(seed).permutation(n)


def tokenize(texts, tok, eos_id: int, chunk: int = 20_000) -> tuple:
    """(flat uint32 token ids, int64 document offsets); each document ends with the EOS token (TRL adds it as text)."""
    os.environ["TOKENIZERS_PARALLELISM"] = "true"  # no forked workers in this script
    parts, lengths = [], []
    for i in range(0, len(texts), chunk):
        ids = tok(list(texts[i:i + chunk]), return_attention_mask=False)["input_ids"]
        lens = np.fromiter(map(len, ids), dtype=np.int64, count=len(ids)) + 1
        out = np.full(int(lens.sum()), eos_id, dtype=np.uint32)
        body = np.ones(len(out), dtype=bool)
        body[np.cumsum(lens) - 1] = False
        out[body] = np.fromiter(itertools.chain.from_iterable(ids), dtype=np.uint32, count=int(body.sum()))
        parts.append(out)
        lengths.append(lens)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    lengths = np.concatenate(lengths) if lengths else np.zeros(0, np.int64)
    return np.concatenate(parts) if parts else np.zeros(0, np.uint32), np.concatenate([[0], np.cumsum(lengths)]).astype(np.int64)


def pack_wrapped(offsets: np.ndarray, seq_len: int) -> tuple:
    """(starts, ends) of TRL's 'wrapped' packing: each run of 1,000 documents is concatenated and cut every seq_len
    tokens, so its last sequence is shorter."""
    n = len(offsets) - 1
    starts, ends = [], []
    for b in range(0, n, DOCS_PER_PACK):
        lo, hi = offsets[b], offsets[min(b + DOCS_PER_PACK, n)]
        s = np.arange(lo, hi, seq_len)
        starts.append(s)
        ends.append(np.minimum(s + seq_len, hi))
    return np.concatenate(starts), np.concatenate(ends)


def make_batch(ids, starts, ends, rows, pad_id: int, device):
    """(input_ids, attention_mask or None without padding, labels with -100 on padding, tokens, predicted tokens),
    right-padded like TRL's collator."""
    import torch

    lens = ends[rows] - starts[rows]
    width = int(lens.max())
    x = np.full((len(rows), width), pad_id, dtype=np.int64)
    for i, (s, n) in enumerate(zip(starts[rows], lens)):
        x[i, :n] = ids[s:s + n]
    x = torch.from_numpy(x)
    mask = torch.arange(width)[None, :] < torch.from_numpy(lens)[:, None]
    labels = x.masked_fill(~mask, -100)
    am = None if bool(mask.all()) else mask.long().to(device)
    return x.to(device), am, labels.to(device), int(lens.sum()), int((lens - 1).clip(min=0).sum())


def evaluate(model, data, batch_size, pad_id, device, autocast) -> float:
    """Mean token loss over the validation sequences."""
    import torch

    ids, starts, ends = data
    total, n_tok = 0.0, 0
    model.eval()
    with torch.no_grad():
        for b in range(0, len(starts), batch_size):
            x, am, y, _, n = make_batch(ids, starts, ends, np.arange(b, min(b + batch_size, len(starts))), pad_id, device)
            with autocast():
                total += float(model(input_ids=x, attention_mask=am, labels=y, num_items_in_batch=n).loss) * n
            n_tok += n
    model.train()
    return total / max(n_tok, 1)


def check_export(export_dir: Path, trained_head, base_dir: Path) -> dict:
    """Reload the export on the CPU: untied lm_head equal to the trained one, 10 greedy tokens, share of weights changed."""
    import torch
    from safetensors import safe_open
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model = AutoModelForCausalLM.from_pretrained(export_dir, dtype=torch.bfloat16)
    # transformers 4.57.6 flags every local folder it saved itself as a Mistral tokenizer; Qwen's regex is not affected
    tok = AutoTokenizer.from_pretrained(export_dir, fix_mistral_regex=False)
    sample = CHECK_PROMPT + ": fMRI at 3 T (n = 24), α-band, 5-HT2A."
    same_tok = tok(sample)["input_ids"] == AutoTokenizer.from_pretrained(base_dir)(sample)["input_ids"]
    head, emb = model.lm_head.weight, model.get_input_embeddings().weight
    enc = tok(CHECK_PROMPT, return_tensors="pt")
    out = model.generate(**enc, max_new_tokens=10, min_new_tokens=10, do_sample=False, temperature=None, top_p=None, top_k=None,
                         pad_token_id=tok.pad_token_id)
    new = out[0, enc["input_ids"].shape[1]:]
    changed = total = 0
    state = model.state_dict()
    with safe_open(str(base_dir / "model.safetensors"), "pt") as f:
        for k in f.keys():
            a = f.get_tensor(k)
            changed += int((a != state[k]).sum())
            total += a.numel()
    res = {"tie_word_embeddings": model.config.tie_word_embeddings, "lm_head_shares_embedding": head.data_ptr() == emb.data_ptr(),
           "lm_head_equals_trained_head": bool(torch.equal(head, trained_head)),
           "lm_head_vs_embedding_share_different": float((head != emb).float().mean()),
           "weights_changed_vs_base": changed / total, "tokenizer_matches_base": same_tok, "prompt": CHECK_PROMPT, "generated_tokens": int(len(new)),
           "generated_text": tok.decode(new)}
    if res["lm_head_shares_embedding"] or not res["lm_head_equals_trained_head"] or res["generated_tokens"] != 10 or not same_tok:
        raise RuntimeError(f"export check failed: {res}")
    return res


def main(argv=None):
    p = base_parser("NeuroQwen3-0.6B: DAPT of Qwen/Qwen3-0.6B on PubMed titles + abstracts, evaluation papers removed.", stage=STAGE)
    p.set_defaults(seed=SEED)
    p.add_argument("--run-name", default=None, help="Run directory under scripts/outputs/runs/ (default: timestamp; 'smoke' with --smoke).")
    p.add_argument("--corpus", type=Path, default=None, help=f"The pinned {CORPUS_NAME} (default: {DEFAULT_CORPUS}).")
    p.add_argument("--unpinned-corpus", action="store_true", help="Accept a corpus whose sha256 differs from the pin.")
    p.add_argument("--no-eval-filter", dest="eval_filter", action="store_false",
                   help="Keep the validation/test and NeuroVault evaluation papers (the published NeuroQwen3-0.6B corpus).")
    p.add_argument("--max-length", type=int, default=None, help=f"Packed sequence length (recipe {MAX_LENGTH}).")
    p.add_argument("--grad-accum", type=int, default=None, help=f"Micro-batches per step (default {SEQS_PER_STEP} / --batch-size).")
    p.add_argument("--lr", type=float, default=None, help=f"Peak learning rate (recipe {LR}).")
    p.add_argument("--warmup-ratio", type=float, default=None, help=f"Warmup share of the steps (recipe {WARMUP_RATIO}).")
    p.add_argument("--max-steps", type=int, default=None, help="Stop after this many optimizer steps (the schedule spans them).")
    p.add_argument("--eval-every", type=int, default=EVAL_EVERY, help="Validation loss every N steps (0 = only at the end).")
    p.add_argument("--save-every", type=int, default=SAVE_EVERY, help="Resumable checkpoint every N steps (0 = none).")
    p.add_argument("--resume", action="store_true", help="Continue from neuroqwen/checkpoint.pt.")
    p.add_argument("--weights-bf16", action="store_true",
                   help="Train the bf16 weights directly, no fp32 master copy (the notebook; most updates round away).")
    raw = p.parse_args(argv)
    if raw.smoke:
        for k, v in SMOKE.items():
            key = "max_samples" if k == "docs" else k
            if getattr(raw, key, None) is None:
                setattr(raw, key, v)
    args = finalize_args(raw)
    run_dir = resolve_run_dir(args)
    out_dir = run_dir / "neuroqwen"
    out_dir.mkdir(parents=True, exist_ok=True)
    export_dir = out_dir / EXPORT_NAME
    if artifacts_exist(export_dir / "model.safetensors", out_dir / f"{STAGE}.summary.json", force=args.force) and not args.resume:
        log.info("%s exists; skipping (use --force to retrain)", export_dir)
        return 0
    setup_torch(args.seed, args.deterministic)
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup

    device = select_device(args.device)
    max_length = args.max_length or MAX_LENGTH
    precision = "bf16" if args.weights_bf16 else "fp32"
    batch_size = args.batch_size or MICRO_BATCH[precision]
    grad_accum = args.grad_accum or SEQS_PER_STEP // batch_size
    lr = LR if args.lr is None else args.lr
    warmup_ratio = WARMUP_RATIO if args.warmup_ratio is None else args.warmup_ratio
    epochs = args.epochs or EPOCHS
    meta = dict(stage=STAGE, smoke=args.smoke, run=args.run_name)

    with timed("load_corpus", log, out=args.out, **meta):
        df, corpus = load_corpus(args)
        keep, removed, counts = eval_filter(df)
        counts["applied"] = bool(args.eval_filter)
        removed.to_csv(out_dir / "eval_filter_removed.csv", index=False)
        write_json(out_dir / "eval_filter.json", {**counts, "corpus": corpus})
        log.info("eval filter: %s %d of %d documents (%s); test papers in corpus %d/%d, val %d/%d -> %s",
                 "removed" if args.eval_filter else "would remove (--no-eval-filter)", counts["removed"], len(df),
                 counts["removed_by_rule"], counts["test_papers_in_corpus_by_pmid"], counts["test_papers"],
                 counts["val_papers_in_corpus_by_pmid"], counts["val_papers"], out_dir / "eval_filter.json")
        if args.eval_filter:
            df = df[keep].reset_index(drop=True)
        texts = notebook_texts(df)
        texts = texts[texts != ""]
        del df
        if args.max_samples and len(texts) > args.max_samples:  # notebook DEBUG_MAX_SAMPLES
            texts = texts[permutation(len(texts), args.seed)[: args.max_samples]]
        perm = permutation(len(texts), args.seed)
        n_val = math.ceil(VAL_FRAC * len(texts))
        train_texts, val_texts = texts[perm[n_val:]], texts[perm[:n_val]]
        log.info("documents: %d train / %d val", len(train_texts), len(val_texts))

    base_dir = hf_pinned_dir(BASE_MODEL)
    tok = AutoTokenizer.from_pretrained(base_dir, use_fast=True)
    pad_id, eos_id = tok.pad_token_id, tok.eos_token_id
    with timed("tokenize", log, out=args.out, **meta):
        train_ids, train_off = tokenize(train_texts, tok, eos_id)
        val_ids, val_off = tokenize(val_texts, tok, eos_id)
        del texts, train_texts, val_texts
        starts, ends = pack_wrapped(train_off, max_length)
        val_data = (val_ids, *pack_wrapped(val_off, max_length))
    n_tokens = int(train_off[-1])
    log.info("train: %d tokens (EOS included) in %d sequences of <= %d; val: %d tokens", n_tokens, len(starts), max_length, int(val_off[-1]))

    seqs_per_step = batch_size * grad_accum
    steps_per_epoch = math.ceil(len(starts) / seqs_per_step)
    total_steps = min(args.max_steps or epochs * steps_per_epoch, epochs * steps_per_epoch)
    orders = [torch.randperm(len(starts), generator=torch.Generator().manual_seed(args.seed + e)).numpy() for e in range(epochs)]
    recipe = {"base_model": BASE_MODEL, "base_revision": hf_pinned(BASE_MODEL), "max_length": max_length, "micro_batch": batch_size,
              "grad_accum": grad_accum, "sequences_per_step": seqs_per_step, "tokens_per_step": seqs_per_step * max_length,
              "epochs": epochs, "steps": total_steps, "lr": lr, "schedule": "cosine", "warmup_steps": math.ceil(warmup_ratio * total_steps),
              "weight_decay": WEIGHT_DECAY, "max_grad_norm": MAX_GRAD_NORM, "seed": args.seed, "val_frac": VAL_FRAC,
              "eval_filter": bool(args.eval_filter), "sequences": len(starts), "train_tokens": n_tokens,
              "weights": "bf16 (notebook)" if args.weights_bf16 else "fp32 master, bf16 autocast", "export_dtype": "bfloat16"}
    log.info("recipe: %s", recipe)

    with timed("load_model", log, out=args.out, **meta):
        dtype = torch.bfloat16 if args.weights_bf16 else torch.float32
        autocast = functools.partial(torch.autocast, device.type, dtype=torch.bfloat16, enabled=not args.weights_bf16)
        # tie_word_embeddings=False: the published run (transformers 5) trained lm_head and embed_tokens as two matrices
        model = AutoModelForCausalLM.from_pretrained(base_dir, dtype=dtype, tie_word_embeddings=False, attn_implementation="sdpa").to(device)
        model.config.pad_token_id = model.generation_config.pad_token_id = pad_id
        model.config.use_cache = False
        lm_sdpa_backends(args.deterministic, log)
        decay = [prm for prm in model.parameters() if prm.ndim >= 2]  # Trainer: no decay on norms (Qwen3 has no biases)
        no_decay = [prm for prm in model.parameters() if prm.ndim < 2]
        opt = torch.optim.AdamW([{"params": decay, "weight_decay": WEIGHT_DECAY}, {"params": no_decay, "weight_decay": 0.0}],
                                lr=lr, betas=(0.9, 0.999), eps=1e-8, fused=device.type == "cuda")
        sched = get_cosine_schedule_with_warmup(opt, recipe["warmup_steps"], total_steps)
    ckpt_path, start_step = out_dir / "checkpoint.pt", 0
    if args.resume and ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        if ckpt["recipe"] != recipe:
            raise SystemExit(f"{ckpt_path} was written with another recipe: {ckpt['recipe']}")
        model.load_state_dict(ckpt["model"])
        opt.load_state_dict(ckpt["optimizer"])
        sched.load_state_dict(ckpt["scheduler"])
        start_step = ckpt["step"]
        del ckpt
        log.info("resumed from %s at step %d", ckpt_path, start_step)

    hist_path = out_dir / "history.jsonl"
    if start_step == 0:
        hist_path.unlink(missing_ok=True)
    model.train()
    window, tok_seen, val_loss = [], 0, None
    t0 = time.perf_counter()
    with timed("train", log, out=args.out, steps=total_steps - start_step, device=str(device), **meta) as t_train:
        for step in range(start_step, total_steps):
            epoch, i = divmod(step, steps_per_epoch)
            rows = orders[epoch][i * seqs_per_step:(i + 1) * seqs_per_step]
            batches = [make_batch(train_ids, starts, ends, rows[m:m + batch_size], pad_id, device) for m in range(0, len(rows), batch_size)]
            n_items = sum(b[4] for b in batches)  # predicted tokens of the whole step (Trainer's num_items_in_batch)
            step_loss = 0.0
            for x, am, y, n_tok, _ in batches:
                with autocast():
                    loss = model(input_ids=x, attention_mask=am, labels=y, num_items_in_batch=n_items).loss
                loss.backward()
                step_loss += loss.detach()
                tok_seen += n_tok
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
            lr_now = sched.get_last_lr()[0]
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            window.append((float(step_loss), float(grad_norm), lr_now))
            done = step + 1
            if args.eval_every and done % args.eval_every == 0 and done < total_steps:
                val_loss = evaluate(model, val_data, batch_size, pad_id, device, autocast)
            if done % LOG_EVERY == 0 or done == total_steps or val_loss is not None:
                secs = time.perf_counter() - t0
                row = {"step": done, "loss": float(np.mean([w[0] for w in window])), "grad_norm": float(np.mean([w[1] for w in window])),
                       "lr": window[-1][2], "tokens_per_s": tok_seen / secs, "elapsed_s": secs, "val_loss": val_loss}
                with hist_path.open("a") as fh:
                    fh.write(json.dumps(row) + "\n")
                log.info("step %d/%d loss %.4f grad_norm %.3f lr %.2e %.0f tok/s%s eta %.1f h", done, total_steps, row["loss"],
                         row["grad_norm"], row["lr"], row["tokens_per_s"], "" if val_loss is None else f" val_loss {val_loss:.4f}",
                         (total_steps - done) * secs / max(done - start_step, 1) / 3600)
                window, val_loss = [], None
            if args.save_every and done % args.save_every == 0 and done < total_steps:
                tmp = ckpt_path.with_suffix(".tmp")
                torch.save({"model": model.state_dict(), "optimizer": opt.state_dict(), "scheduler": sched.state_dict(),
                            "step": done, "recipe": recipe}, tmp)
                os.replace(tmp, ckpt_path)
    tokens_per_s = tok_seen / max(t_train.seconds, 1e-9)
    peak_gb = torch.cuda.max_memory_allocated(device) / 2**30 if device.type == "cuda" else None
    log.info("training: %.0f tokens/s%s", tokens_per_s, "" if peak_gb is None else f", peak GPU memory {peak_gb:.1f} GB")

    with timed("final_eval", log, out=args.out, **meta):
        val_loss = evaluate(model, val_data, batch_size, pad_id, device, autocast)
    log.info("final validation loss %.4f", val_loss)

    with timed("export", log, out=args.out, **meta):
        # tie_word_embeddings=False in config.json: transformers 4.57 would otherwise replace the trained lm_head by
        # embed_tokens on load (that is what happens to the published NeuroQwen3-0.6B, whose config says true)
        model = model.to(torch.bfloat16).cpu()
        model.config.tie_word_embeddings = False
        model.config.use_cache = True
        model.save_pretrained(export_dir, safe_serialization=True)
        tok.save_pretrained(export_dir)
        trained_head = model.lm_head.weight.detach().clone()
        del model, opt
        check = check_export(export_dir, trained_head, base_dir)
        write_json(out_dir / "export_check.json", check)
    log.info("exported %s; reload: lm_head untied and trained %s, %d tokens generated (%r), %.1f %% of weights differ from %s",
             export_dir, check["lm_head_equals_trained_head"], check["generated_tokens"], check["generated_text"],
             100 * check["weights_changed_vs_base"], BASE_MODEL)
    ckpt_path.unlink(missing_ok=True)

    write_json(out_dir / f"{STAGE}.summary.json", {
        "stage": STAGE, "smoke": args.smoke, "export": str(export_dir), "recipe": recipe, "corpus": corpus,
        "eval_filter": counts, "train_documents": int(len(train_off) - 1), "val_documents": int(len(val_off) - 1),
        "final_val_loss": val_loss, "train_seconds": t_train.seconds, "tokens_per_s": tokens_per_s, "peak_gpu_gb": peak_gb, "device": str(device),
        "export_check": check, "provenance": provenance(), "ts": time.strftime("%Y-%m-%dT%H:%M:%S")})
    return 0


if __name__ == "__main__":
    sys.exit(main())
