#!/usr/bin/env python
"""Context-free baseline (reviewer 3.3): swap the frozen text encoder and retrain only the projection heads on the selected
model's autoencoder latents (--model-dir, default runs/final), then score retrieval and text-to-brain on the PubMed test split.

Recipes as scripts/models/09_projection_head.py: stage A (BCE through the frozen decoder) initialises the InfoNCE text head;
the text-to-brain head is head v2 (the final model's), or stage A with --released (the submitted model's). So the SPECTER2 row
retrains the selected model's own heads, and the reference row scores those heads directly.  Embeddings cached in outputs/cache/.

Usage: run_analysis.py --device cuda [--encoders specter2,word2vec] [--eval-only]   (smoke: --smoke --device cpu)
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _final_model as M  # noqa: E402
from _a5_common import (  # noqa: E402
    Timings, count_params, eval_heads_retrieval, eval_text_to_brain_head, exclude_heldout_rows, get_logger, load_pubmed_latents,
    load_pubmed_maps, load_pubmed_text_rows, load_specter_latents, read_json, same_exclusions, seed_all, select_device,
    set_offline_env, sha_key, split_masks, subsample_masks, train_infonce_heads, train_text_to_brain_head, train_text_to_brain_v2,
    write_json,
)

set_offline_env()
log = get_logger("a5_encoder_swap")

MSE_EPOCHS, MSE_BATCH, MSE_LR = 201, 1024, 5e-5
NCE_EPOCHS, NCE_BATCH, NCE_LR = 301, 2048, 1e-5
T2B_EPOCHS = 80  # head v2
HEAD_SEED = 123
PAPER_DICE_PCT = 70  # paper Dice: both maps thresholded at their 70th percentile
GPU_RATES = {"mse_sec_per_epoch": 0.093, "infonce_sec_per_epoch": 0.017, "head_v2_sec_per_epoch": 0.27}  # 09 runs (scripts/README.md)
GPU_EST_PER_ENCODER = (MSE_EPOCHS * GPU_RATES["mse_sec_per_epoch"] + NCE_EPOCHS * GPU_RATES["infonce_sec_per_epoch"]
                       + T2B_EPOCHS * GPU_RATES["head_v2_sec_per_epoch"])
CACHE_VERSION = "v2"  # part of the embedding-cache key
# encoders: specter2 = the paper's (cached corpus embeddings; also the reference row), specter2_base /
# specter2_prox = SPECTER2 without adapter / with the proximity adapter, word2vec = mean word vectors (the baseline asked for),
# tfidf[<k>] = TF-IDF + SVD fitted on the train abstracts (300 or k dims), minilm = all-MiniLM-L6-v2, large = --large-encoder.
# Only specter2 and word2vec are used by the response; the rest are opt-in.
ENCODER_INFO = {
    "specter2": {"label": "SPECTER2 (paper)", "contextual": True, "params_M": 110.6},
    "specter2_base": {"label": "SPECTER2 base, no adapter", "contextual": True, "params_M": 110.1},
    "specter2_prox": {"label": "SPECTER2 proximity adapter", "contextual": True, "params_M": 110.6},
    "word2vec": {"label": "mean word2vec", "contextual": False},
    "tfidf": {"label": "TF-IDF + SVD", "contextual": False},
    "minilm": {"label": "all-MiniLM-L6-v2", "contextual": True},
    "large": {"label": "large encoder", "contextual": True},
}
SMOKE = {"max_samples": 64, "epochs": 1, "infonce_epochs": 1, "max_batches": 2}
TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z\-]+")
SPECTER_VARIANTS = {  # encoder -> (neurovlm.models.Specter model, adapter)
    "specter2": ("allenai/specter2_aug2023refresh", "adhoc_query"),
    "specter2_base": ("allenai/specter2_aug2023refresh", None),
    "specter2_prox": ("allenai/specter2", "proximity"),
}


def tfidf_dim(encoder: str) -> int:
    """SVD components: ``tfidf`` = 300, ``tfidf<k>`` = k."""
    return int(encoder[5:]) if encoder[5:].isdigit() else 300


def specter_texts(rows) -> list[str]:
    """The paper's SPECTER2 input (scripts/data/02_pubmed.py): title + "[SEP]" + abstract."""
    return [f"{n}[SEP]{d}" for n, d in zip(rows["name"], rows["description"])]


def load_specter_variant(encoder: str, device):
    """``neurovlm.models.Specter`` (no revision argument; it reads the HF cache first) once the cache serves the pinned
    revisions of its base model and adapter."""
    from neurovlm.models import Specter
    from _hf_pins import SPECTER_REPOS, hf_cache_pinned

    model, adapter = SPECTER_VARIANTS[encoder]
    if encoder == "specter2_prox":  # opt-in pair, manifest group "extra"
        hf_cache_pinned("allenai/specter2_base", "allenai/specter2", group="extra")
    else:
        hf_cache_pinned(*SPECTER_REPOS)
    return Specter(model, adapter=adapter, device=str(device))


def specter_encode(specter, texts: list[str], batch_size: int = 32):
    """Length-sorted batches through a ``neurovlm.models.Specter`` wrapper -> float32 [N, 768] in input order."""
    import torch

    order = np.argsort([len(t) for t in texts])
    out = torch.empty((len(texts), 768), dtype=torch.float32)
    with torch.inference_mode():
        for s in range(0, len(texts), batch_size):
            idx = order[s:s + batch_size]
            out[torch.as_tensor(idx)] = specter([texts[i] for i in idx]).detach().float().cpu()
    return out


def corpus_texts(rows) -> list[str]:
    return [f"{n}. {d}".strip(" .") for n, d in zip(rows["name"], rows["description"])]


def load_word_vectors(model: str, limit: int | None):
    import gensim.downloader as api
    from gensim.models import KeyedVectors

    local = Path(api.base_dir) / model / f"{model}.gz"
    if model == "word2vec-google-news-300" and local.exists():
        return KeyedVectors.load_word2vec_format(str(local), binary=True, limit=limit)
    return api.load(model)  # ~/gensim-data; raises offline if not cached


def embed_word2vec(texts: list[str], *, model: str, limit: int | None, kv=None) -> tuple[np.ndarray, dict]:
    """Mean of in-vocabulary word vectors (case-sensitive lookup first, then lower case)."""
    kv = kv if kv is not None else load_word_vectors(model, limit)
    dim = kv.vector_size
    out = np.zeros((len(texts), dim), dtype=np.float32)
    hit = total = empty = 0
    for i, t in enumerate(texts):
        vecs = []
        for tok in TOKEN_RE.findall(t):
            total += 1
            for cand in (tok, tok.lower()):
                if cand in kv.key_to_index:
                    vecs.append(kv[cand])
                    hit += 1
                    break
        if vecs:
            out[i] = np.mean(vecs, axis=0)
        else:
            empty += 1
    n_vocab = len(kv.key_to_index)
    return out, {"method": "mean word vectors", "vectors": model, "vocab_limit": limit, "vocab_size": n_vocab, "token_coverage": hit / max(total, 1),
                 "rows_without_known_token": empty, "params_M": n_vocab * dim / 1e6, "params_note": "lookup-table entries (frozen, no context)"}


def fit_tfidf_svd(texts: list[str], train_mask: np.ndarray, *, dim: int = 300, seed: int = 0):
    """TF-IDF + truncated SVD fitted on the train texts only."""
    from sklearn.decomposition import TruncatedSVD
    from sklearn.feature_extraction.text import TfidfVectorizer

    tr = [t for t, m in zip(texts, train_mask) if m]
    tfidf = TfidfVectorizer(min_df=2, max_df=0.9, sublinear_tf=True, stop_words="english")
    X_tr = tfidf.fit_transform(tr)
    k = max(2, min(dim, X_tr.shape[1] - 1, X_tr.shape[0] - 1))
    svd = TruncatedSVD(n_components=k, random_state=seed).fit(X_tr)
    return tfidf, svd, {"n_fit_docs": len(tr)}


def embed_tfidf_svd(texts: list[str], train_mask: np.ndarray, *, dim: int = 300, seed: int = 0, save_to: Path | None = None) -> tuple[np.ndarray, dict]:
    tfidf, svd, fit = fit_tfidf_svd(texts, train_mask, dim=dim, seed=seed)
    if save_to is not None:
        import joblib

        joblib.dump({"tfidf": tfidf, "svd": svd}, save_to)
    k = int(svd.n_components)
    Z = svd.transform(tfidf.transform(texts)).astype(np.float32)
    return Z, {"method": "tfidf+svd (fit on train abstracts)", "vocab_size": int(len(tfidf.vocabulary_)), "components": k,
               "n_fit_docs": fit["n_fit_docs"], "explained_variance": float(svd.explained_variance_ratio_.sum()),
               "params_M": len(tfidf.vocabulary_) * k / 1e6, "params_note": "vocabulary x SVD components (frozen, no context)"}


def embed_sentence_transformer(texts: list[str], name: str, device, batch_size: int = 64) -> tuple[np.ndarray, dict]:
    from _hf_pins import sentence_transformer

    st = sentence_transformer(name, device)  # at the revision pinned in scripts/hf_manifest.json
    emb = st.encode(texts, batch_size=batch_size, convert_to_numpy=True, normalize_embeddings=False, show_progress_bar=False)
    n_params = sum(p.numel() for p in st.parameters())
    return emb.astype(np.float32), {"method": "sentence-transformers", "model": name, "params_M": n_params / 1e6, "max_seq_length": int(st.max_seq_length)}


def corpus_embeddings(encoder: str, rows, pmids, train_mask, *, device, cache_dir: Path, smoke: bool, large_name: str | None, w2v_model: str,
                      timings: Timings, cache_tag: str = ""):
    """Title + abstract embeddings of one encoder, L2-normalised and cached (key: encoder, pmid set, train mask, ``cache_tag``)."""
    import torch
    import torch.nn.functional as F

    cache_dir.mkdir(parents=True, exist_ok=True)
    pm = np.asarray(pmids, dtype=np.int64)
    key = sha_key(CACHE_VERSION, encoder, large_name or "", w2v_model if encoder == "word2vec" else "", smoke, len(pm), int(pm.sum() % 2**31),
                  int((pm * np.asarray(train_mask)).sum() % 2**31))
    path = cache_dir / f"emb_{encoder}_{key}{cache_tag}.pt"
    if path.exists():
        payload = torch.load(path, weights_only=False)
        return payload["emb"].float(), payload["meta"]
    texts = corpus_texts(rows)
    with timings.time(f"embed_{encoder}", n=len(pm), device=str(device)):
        if encoder == "specter2":
            emb = load_specter_latents(pmids)
            meta = {"method": "cached SPECTER2 adhoc_query embeddings (latent_specter2_adhoc.pt)", "params_M": ENCODER_INFO["specter2"]["params_M"]}
        elif encoder in SPECTER_VARIANTS:
            sp = load_specter_variant(encoder, device)
            emb = specter_encode(sp, specter_texts(rows))
            model, adapter = SPECTER_VARIANTS[encoder]
            meta = {"method": f"neurovlm.models.Specter({model!r}, adapter={adapter!r}) on title[SEP]abstract (CLS, orthogonalised)",
                    "params_M": sum(p.numel() for p in sp.specter.parameters()) / 1e6}
            del sp
        else:
            if encoder == "word2vec":
                e, meta = embed_word2vec(texts, model=w2v_model, limit=200_000 if smoke else None)
            elif encoder.startswith("tfidf"):
                # the fitted transform is saved so that other texts are embedded identically
                e, meta = embed_tfidf_svd(texts, np.asarray(train_mask), dim=tfidf_dim(encoder), save_to=path.with_suffix(".joblib"))
            elif encoder == "minilm":
                e, meta = embed_sentence_transformer(texts, "sentence-transformers/all-MiniLM-L6-v2", device)
            elif encoder == "large" and large_name:
                e, meta = embed_sentence_transformer(texts, large_name, device)
            else:
                raise ValueError(f"unknown encoder {encoder!r} ('large' needs --large-encoder)")
            emb = torch.from_numpy(e)
    emb = emb.float()
    meta["raw_norm_mean"] = float(emb.norm(dim=1).mean())
    meta["zero_rows"] = int((emb.norm(dim=1) == 0).sum())
    emb = F.normalize(emb, dim=1, eps=1e-8)
    meta["l2_normalised"] = True
    torch.save({"emb": emb, "meta": meta, "pmids": pm}, path)
    return emb, meta


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path, default=Path(__file__).resolve().parent / "outputs")
    M.add_model_args(p)
    p.add_argument("--device", default="auto")
    p.add_argument("--seed", type=int, default=HEAD_SEED)
    p.add_argument("--encoders", default="specter2,word2vec",
                   help="Comma list of specter2,specter2_base,specter2_prox,minilm,word2vec,tfidf,tfidf<k>[,large].")
    p.add_argument("--w2v-model", default="word2vec-google-news-300", help="gensim vectors for word2vec (e.g. glove-wiki-gigaword-300).")
    p.add_argument("--large-encoder", default=None, help="HF / sentence-transformers id for an extra 'large' row (pin it in scripts/hf_manifest.json first).")
    p.add_argument("--epochs", type=int, default=None, help="Stage-A epochs (default 201; 1 in --smoke).")
    p.add_argument("--infonce-epochs", type=int, default=None, help="Stage-B epochs (default 301; 1 in --smoke).")
    p.add_argument("--max-samples", type=int, default=None, help="Cap rows (split-proportional prefix).")
    p.add_argument("--max-batches", type=int, default=None, help="Cap batches per epoch.")
    p.add_argument("--skip-reference", action="store_true", help="Skip the reference row (the selected model's own heads).")
    p.add_argument("--smoke", action="store_true", help="64 rows, 1 epoch x 2 batches per stage, 200k-word vocabulary.")
    p.add_argument("--force", action="store_true", help="Retrain encoders that already have results.")
    p.add_argument("--eval-only", action="store_true", help="Re-evaluate the saved heads; no training.")
    p.add_argument("--no-exclusions", action="store_true", help="Keep the leakage-audit papers in val/test.")
    args = p.parse_args(argv)
    if args.smoke:
        for k, v in SMOKE.items():
            setattr(args, k, getattr(args, k) or v)
    mse_epochs, nce_epochs = args.epochs or MSE_EPOCHS, args.infonce_epochs or NCE_EPOCHS
    t2b_epochs = 1 if args.smoke else T2B_EPOCHS
    encoders = [e.strip() for e in args.encoders.split(",") if e.strip()]
    if args.large_encoder and "large" not in encoders:
        encoders.append("large")
    out = args.out
    M.from_args(args)
    t2b_recipe = "v1" if args.released else "v2"  # the selected model's text-to-brain head
    ck_dir = out / "checkpoints" / M.model_tag()
    ck_dir.mkdir(parents=True, exist_ok=True)
    timings = Timings(out / "timings.json")
    seed_all(args.seed)
    import torch

    device = select_device(args.device)
    log.info("device %s | model %s | encoders %s | epochs A/B %d/%d | text-to-brain head %s | smoke %s", device, M.model_tag(), encoders,
             mse_epochs, nce_epochs, t2b_recipe, args.smoke)

    with timings.time("load_data"):
        latents, pmids = load_pubmed_latents()
        # exclusions before the --max-samples prefix
        masks, excl = exclude_heldout_rows(split_masks(pmids), pmids, enabled=not args.no_exclusions, log=log.info)
        masks = subsample_masks(masks, args.max_samples)
        keep = masks["train"] | masks["val"] | masks["test"]
        latents, pmids = latents[torch.from_numpy(keep)], pmids[keep]
        masks = {k: v[keep] for k, v in masks.items()}
        rows = load_pubmed_text_rows(pmids)
        maps = load_pubmed_maps(pmids)
        ae = M.component("autoencoder", device)
        ref_image_head = M.component("proj_head_image_infonce", device)
        log.info("rows train/val/test = %d/%d/%d | maps %s", masks["train"].sum(), masks["val"].sum(), masks["test"].sum(), tuple(maps.shape))
    tr, va, te = (torch.from_numpy(masks[k]) for k in ("train", "val", "test"))
    emb_kw = dict(device=device, cache_dir=out / "cache", smoke=args.smoke, large_name=args.large_encoder, w2v_model=args.w2v_model, timings=timings,
                  cache_tag="_excl" if excl["enabled"] else "")

    prev = read_json(out / "metrics.json") if (out / "metrics.json").exists() else {}
    reuse = (not args.force and bool(prev.get("smoke")) == bool(args.smoke) and same_exclusions(prev, excl["enabled"])
             and prev.get("model_dir") == M.model_info()["model_dir"] and prev.get("text_to_brain_head") == t2b_recipe)
    results = prev.get("results", {}) if reuse else {}
    run_meta = {"smoke": args.smoke, "exclusions": excl, "model_dir": M.model_info()["model_dir"], "text_to_brain_head": t2b_recipe}

    # reference: the selected model's own heads on SPECTER2, no training
    if not args.skip_reference and "reference" not in results:
        with timings.time("eval_reference"):
            emb, _ = corpus_embeddings("specter2", rows, pmids, masks["train"], **emb_kw)
            th, ih, mh = (M.component(k, device) for k in ("proj_head_text_infonce", "proj_head_image_infonce", "proj_head_text_mse"))
            r = {"encoder": "specter2", "label": f"SPECTER2 + {'released' if args.released else 'final'} heads", "trained_here": False, "dim": 768,
                 "encoder_params_M": ENCODER_INFO["specter2"]["params_M"], "contextual": True,
                 "heads_params": count_params(th) + count_params(ih) + count_params(mh)}
            r["retrieval"] = eval_heads_retrieval(th, ih, emb[te], latents[te], device)
            r["text_to_brain"] = eval_text_to_brain_head(mh, ae, emb[te], maps[te], device, text_va=emb[va], image_head=ref_image_head,
                                                         pct=PAPER_DICE_PCT)
            results["reference"] = r
        write_json(out / "metrics.json", {"results": results, **run_meta})

    for enc in encoders:
        if enc in results and args.eval_only:
            from neurovlm.models import ProjHead

            emb, _ = corpus_embeddings(enc, rows, pmids, masks["train"], **emb_kw)
            dim, heads = int(emb.shape[1]), {}
            for nm, (i, h, o) in (("text_mse", (dim, 512, 384)), ("text_infonce", (dim, 512, 384)), ("image_infonce", (384, 384, 384))):
                heads[nm] = ProjHead(seed=0, latent_in_dim=i, hidden_dim=h, latent_out_dim=o)
                heads[nm].load_state_dict(torch.load(ck_dir / f"{enc}_proj_head_{nm}.pt", map_location="cpu"))
                heads[nm] = heads[nm].to(device).eval()
            with torch.no_grad(), timings.time(f"eval_{enc}"):
                results[enc]["retrieval"] = eval_heads_retrieval(heads["text_infonce"], heads["image_infonce"], emb[te], latents[te], device)
                results[enc]["text_to_brain"] = eval_text_to_brain_head(heads["text_mse"], ae, emb[te], maps[te], device, text_va=emb[va],
                                                                        image_head=ref_image_head, pct=PAPER_DICE_PCT)
            r = results[enc]
            log.info("%s (eval only) | t2i AUC %.4f i2t AUC %.4f | dice %.4f | pearson %.4f", enc, r["retrieval"]["t2i_auc"], r["retrieval"]["i2t_auc"],
                     r["text_to_brain"]["dice_matched"], r["text_to_brain"]["pearson_matched"])
            write_json(out / "metrics.json", {**prev, "results": results, **run_meta})
            continue
        if enc in results:
            log.info("%s: done earlier, skipping (use --force)", enc)
            continue
        emb, meta = corpus_embeddings(enc, rows, pmids, masks["train"], **emb_kw)
        dim = int(emb.shape[1])
        info = dict(ENCODER_INFO.get(enc) or {"label": f"TF-IDF + SVD ({tfidf_dim(enc)}-d)", "contextual": False})
        if enc == "word2vec" and "glove" in meta.get("vectors", ""):
            info["label"] = "mean GloVe"
        elif enc == "large":
            info["label"] = args.large_encoder
        log.info("encoder %s: %s | dim %d | params %.1f M", enc, meta.get("method"), dim, meta.get("params_M", float("nan")))
        seed_all(args.seed)
        with timings.time(f"train_text_to_brain_{enc}", epochs=mse_epochs, dim=dim, device=str(device)) as ta:
            head_a, sum_a = train_text_to_brain_head(emb[tr], maps[tr], emb[va], maps[va], decoder=ae.decoder, in_dim=dim, device=device, epochs=mse_epochs,
                                                     batch_size=MSE_BATCH, lr=MSE_LR, seed=args.seed, max_batches=args.max_batches, log=log)
            ta.update({k: sum_a[k] for k in ("sec_per_epoch", "sec_per_step", "steps")})
        with timings.time(f"train_infonce_{enc}", epochs=nce_epochs, dim=dim, device=str(device)) as tb:
            head_t, head_i, sum_b = train_infonce_heads(emb[tr], latents[tr], emb[va], latents[va], text_head_init=head_a, in_dim=dim, device=device, epochs=nce_epochs,
                                                        batch_size=NCE_BATCH, lr=NCE_LR, seed=args.seed, max_batches=args.max_batches, log=log)
            tb.update({k: sum_b[k] for k in ("sec_per_epoch", "sec_per_step", "steps")})
        head_mse, sum_v2 = head_a, None
        if t2b_recipe == "v2":
            seed_all(args.seed)
            with timings.time(f"train_head_v2_{enc}", epochs=t2b_epochs, dim=dim, device=str(device)) as tc:
                head_mse, sum_v2 = train_text_to_brain_v2(emb[tr], latents[tr], emb[va], latents[va], decoder=ae.decoder, in_dim=dim, device=device,
                                                          epochs=t2b_epochs, head_seed=args.seed, max_batches=args.max_batches, log=log)
                tc.update({k: sum_v2[k] for k in ("sec_per_epoch", "sec_per_step", "steps")})
        for nm, mod in (("text_bce", head_a), ("text_mse", head_mse), ("text_infonce", head_t), ("image_infonce", head_i)):
            torch.save(mod.state_dict(), ck_dir / f"{enc}_proj_head_{nm}.pt")
        with timings.time(f"eval_{enc}"):
            r = {"encoder": enc, "label": info["label"], "trained_here": True, "dim": dim, "encoder_params_M": meta.get("params_M"),
                 "contextual": info["contextual"], "embedding_meta": meta, "heads_params": count_params(head_mse) + count_params(head_t) + count_params(head_i),
                 "stage_a": sum_a, "stage_b": sum_b, "head_v2": sum_v2}
            r["retrieval"] = eval_heads_retrieval(head_t, head_i, emb[te], latents[te], device)
            r["text_to_brain"] = eval_text_to_brain_head(head_mse, ae, emb[te], maps[te], device, text_va=emb[va], image_head=ref_image_head,
                                                         pct=PAPER_DICE_PCT)
            r["runtime_estimate_gpu_seconds"] = GPU_EST_PER_ENCODER
        results[enc] = r
        log.info("%s | t2i AUC %.4f i2t AUC %.4f | dice %.4f (null %.4f) | pearson %.4f (null %.4f)", enc, r["retrieval"]["t2i_auc"], r["retrieval"]["i2t_auc"],
                 r["text_to_brain"]["dice_matched"], r["text_to_brain"]["dice_null"], r["text_to_brain"]["pearson_matched"], r["text_to_brain"]["pearson_null"])
        write_json(out / "metrics.json", {"results": results, **run_meta})

    import pandas as pd

    table = pd.DataFrame([{
        "row": k, "encoder": r["label"], "contextual": r["contextual"], "dim": r["dim"], "encoder_params_M": r["encoder_params_M"], "heads_params_K": r["heads_params"] / 1e3,
        "trained_here": r["trained_here"], "t2i_auc": r["retrieval"]["t2i_auc"], "i2t_auc": r["retrieval"]["i2t_auc"], "mean_auc": r["retrieval"]["mean_auc"],
        "mean_recall@10": r["retrieval"]["mean_recall@10"], "dice_pct": r["text_to_brain"]["dice_pct"], "dice_matched": r["text_to_brain"]["dice_matched"],
        "dice_null": r["text_to_brain"]["dice_null"],
        "pearson_matched": r["text_to_brain"]["pearson_matched"], "pearson_null": r["text_to_brain"]["pearson_null"],
        "map_recall_auc": r["text_to_brain"].get("map_recall_auc"), "n_test": r["retrieval"]["n"]} for k, r in results.items()])
    table.to_csv(out / "table.csv", index=False)
    write_json(out / "metrics.json", {"results": results, **run_meta, "model_sources": M.model_sources(), "device": str(device),
                                      "epochs": {"stage_a": mse_epochs, "stage_b": nce_epochs, "head_v2": t2b_epochs},
                                      "n_rows": {k: int(v.sum()) for k, v in masks.items()}, "gpu_rates_sec_per_epoch": GPU_RATES,
                                      "runtime_estimate_per_encoder_gpu_seconds": GPU_EST_PER_ENCODER, "w2v_model": args.w2v_model, "large_encoder": args.large_encoder})
    print(table.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
