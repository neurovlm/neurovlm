#!/usr/bin/env python
"""n-gram vocabulary, binary document matrix and SPECTER2 n-gram embeddings (port of docs/02_data/06_n_grams.ipynb).
The notebook's hand-curated corpus.txt is not in the repo: pass it (or the notebook's ngram_labels.npy) with --corpus.

    .venv/bin/python scripts/data/06_n_grams.py [--smoke] [--force] [--corpus ~/.cache/neurovlm/ngram_labels.npy]
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # scripts/
from common import artifacts_exist, base_parser, finalize_args, get_logger, select_device, setup_torch, timed  # noqa: E402
from data_utils import SPECTER_REPOS, add_data_args, encode_texts, hf_cache_pinned, resolve_artifact_dir  # noqa: E402

STAGE = "data_06_n_grams"
MIN_COUNT = 100
FORCE_KEEP = ["precuneus", "working memory", "putamen"]
log = get_logger(STAGE)

# manual cleaning (cell 2)
DROP_SUBSTRINGS = [
    # study-like language
    "study", "studies", "result", "indicate", "show", "related",
    "differences", "significant", "effect", "role", "measure",
    "displayed", "involved", "examined", "associated", "altered",
    "performed", "demonstrated", "conclus", "correlate", "individuals",
    "common", "prior",
    # too general
    "brain", "neural", "neuroimaging", "mri", "fmri", "connectivity",
    "diagnosed", "patients", "little", "known", "activation", "blood",
    "alterations", "neuroscience", "people",
]
DROP_REGEXES = [
    r"^cortex",
    # general single terms
    r"^ventral$", r"^frontal$", r"^neuronal$", r"^cognitive$",
    r"^cerebral$", r"^resting_state$", r"^disorder$",
    r"^neuropsychological$", r"^cognition$", r"^stimulus$",
    r"^dysfunction$", r"^imaging$", r"^functional$",
    r"^functional imaging$", r"^task performance$", r"^impairments$",
    r"^traits$", r"^dysfunction$", r"^cognitive abilities$", r"^imaging dti$",
    # [SEP] token
    r"\bsep\b",
]
DROP_PATTERN = "|".join([re.escape(s) for s in DROP_SUBSTRINGS] + DROP_REGEXES)

# vocabulary cleaning (cell 3)
DROP_ANY_TOKENS = {"magnetic", "resonance", "imaging", "mri", "fmri", "functional", "scan", "scanner", "image", "data"}
DROP_FIRST_TOKENS = {
    "use", "using", "undergo", "undergoing", "show", "showing", "change", "changes",
    "investigate", "investigating", "examine", "examining", "perform", "performed",
}
FUNCTION_WORDS = {"of", "to", "in", "on", "with", "by", "for", "from", "as", "at", "and", "or", "the", "a", "an"}


def extract_ngrams_sparse(docs, *, ngram_range=(1, 3), min_count: int = MIN_COUNT):
    """One CountVectorizer pass for all n (same features and counts as the notebook's one pass per n).

    Returns (X csc [n_docs, n_kept] int32, feature names) for features with corpus count >= min_count.
    """
    from sklearn.feature_extraction.text import CountVectorizer

    counts = CountVectorizer(ngram_range=ngram_range, stop_words="english", min_df=1, dtype=np.int32)
    X = counts.fit_transform(docs)
    feature_names = counts.get_feature_names_out()
    mask = np.asarray(X.sum(axis=0)).ravel() >= min_count
    X = X.tocsc()[:, mask]
    return X, feature_names[mask]


def clean_vocab_and_remap_X(X, terms, *, force_keep=()):
    """Notebook cell 3: keep "nouny" terms (+ force_keep), collapse sub-phrases into the longest kept phrase,
    OR-merge their columns. Returns (bool X [n_docs, n_new], new terms sorted by (n tokens, term))."""
    import scipy.sparse as sp

    def norm(s: str) -> str:
        s = s.lower().strip()
        s = re.sub(r"[^a-z0-9\s\-]", " ", s)
        return re.sub(r"\s+", " ", s)

    force_keep = {norm(w) for w in force_keep}
    terms = list(terms)
    n_terms = len(terms)

    # canonical form of every kept term
    canon_terms = [None] * n_terms
    for j, t in enumerate(terms):
        nt = norm(t)
        tt = [w for w in nt.split() if w.isalpha()]
        if nt in force_keep:
            canon_terms[j] = nt
        elif len(tt) == 1:
            # the notebook's acronym test (^[a-z]{2,6}$) keeps any 2-6 letter unigram
            if 2 <= len(tt[0]) <= 6:
                canon_terms[j] = tt[0]
        elif len(tt) > 1:
            if tt[0] in DROP_FIRST_TOKENS or any(w in DROP_ANY_TOKENS or w in FUNCTION_WORDS for w in tt):
                continue
            if any(w.endswith(("ing", "ed")) for w in tt):
                continue
            canon_terms[j] = " ".join(tt)

    # collapse contiguous sub-phrases into the longest phrase that contains them
    tok_tuples = [tuple(ct.split()) if ct else () for ct in canon_terms]
    keep_toks = {tt for tt in tok_tuples if tt}
    sub_to_super = {}
    # longest first; the notebook sorted by length only (ties in set order), the phrase makes it deterministic
    for super_tt in sorted(keep_toks, key=lambda t: (-len(t), t)):
        L = len(super_tt)
        for i in range(L):
            for k in range(i + 2, L + 1):
                sub_tt = super_tt[i:k]
                if sub_tt != super_tt and sub_tt in keep_toks and sub_tt not in sub_to_super:
                    sub_to_super[sub_tt] = super_tt

    new_terms = sorted((" ".join(tt) for tt in keep_toks - set(sub_to_super)), key=lambda s: (len(s.split()), s))
    new_index = {t: i for i, t in enumerate(new_terms)}
    old_to_new_idx = np.full(n_terms, -1, dtype=np.int32)
    for j, tt in enumerate(tok_tuples):
        if tt:
            old_to_new_idx[j] = new_index[" ".join(sub_to_super.get(tt, tt))]

    # OR-merge columns: X_new[:, new] = any(X[:, old] for old mapped to new)
    kept = old_to_new_idx >= 0
    merge = sp.csr_matrix(
        (np.ones(int(kept.sum()), dtype=np.int32), (np.flatnonzero(kept), old_to_new_idx[kept])),
        shape=(n_terms, len(new_terms)),
    )
    Xb = sp.csr_matrix(X).astype(bool).astype(np.int32)
    X_new = np.asarray((Xb @ merge).todense()) > 0
    return X_new, new_terms


def load_whitelist(corpus: Path | None, whitelist: str):
    """Terms to keep (cell 7): --corpus file, else the released ngram labels (--whitelist release), else None."""
    if corpus is not None:
        corpus = Path(corpus)
        if corpus.suffix == ".npy":
            terms = np.load(corpus, allow_pickle=True).astype(str)
        else:
            df = pd.read_csv(corpus)
            col = next((c for c in ("terms", "term", "phrase") if c in df.columns), df.columns[0])
            terms = df[col].astype(str).to_numpy()
        return pd.Series(terms), f"{corpus} ({len(terms)} terms)"
    if whitelist == "release":
        from neurovlm.data import load_dataset

        hf_cache_pinned("neurovlm/embedded_text")  # load_dataset takes no revision
        terms = load_dataset("ngrams")["term"].astype(str)
        return terms, f"released load_dataset('ngrams') ({len(terms)} terms)"
    return None, "none (manual filter skipped)"


def main(argv=None):
    p = base_parser("n-gram vocabulary + document matrix + SPECTER2 embeddings of PubMed text.", stage=STAGE)
    add_data_args(p)
    p.add_argument("--min-count", type=int, default=None, help=f"Keep n-grams with corpus count >= this (default {MIN_COUNT}; 2 in --smoke).")
    p.add_argument("--corpus", type=Path, default=None, help="Hand-curated corpus.txt (CSV with a 'terms' column) or .npy of terms to keep.")
    p.add_argument("--whitelist", choices=["none", "release"], default="none",
                   help="Without --corpus: none = automatic vocabulary; release = intersect with load_dataset('ngrams').")
    args = finalize_args(p.parse_args(argv))
    art = resolve_artifact_dir(args)
    setup_torch(args.seed)
    import scipy.sparse as sp
    import torch

    device = select_device(args.device)
    batch_size = args.batch_size or 512
    min_count = args.min_count if args.min_count is not None else (2 if args.smoke else MIN_COUNT)
    meta = dict(stage=STAGE, smoke=args.smoke)
    mat_path, lab_path, emb_path = art / "ngram_matrix.npy", art / "ngram_labels.npy", art / "ngram_emb.pt"
    if artifacts_exist(mat_path, lab_path, emb_path, force=args.force):
        log.info("artifacts exist (%s, %s, %s); use --force to recompute", mat_path, lab_path, emb_path)
        return 0

    from neurovlm.data import load_dataset
    from neurovlm.models import Specter

    hf_cache_pinned(*SPECTER_REPOS, "neurovlm/neuro_image_papers", "neurovlm/embedded_text")  # the neurovlm loaders below take no revision

    with timed("load_dataset", log, out=args.out, **meta):
        df = load_dataset("pubmed_text")
        if args.max_samples:
            df = df.head(args.max_samples).reset_index(drop=True)
        text = df["name"] + " [SEP] " + df["description"]
    log.info("%d documents, min_count=%d", len(df), min_count)

    # 1-3-gram counts (cell 4), cached
    cache_dir = art / "ngram_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    counts_cache = cache_dir / f"counts_1to3_{len(df)}_{min_count}.npz"
    feats_cache = cache_dir / f"features_1to3_{len(df)}_{min_count}.npy"
    if artifacts_exist(counts_cache, feats_cache, force=args.force):
        log.info("reusing cached counts %s", counts_cache)
        X = sp.load_npz(counts_cache).tocsc()
        features = np.load(feats_cache, allow_pickle=True).astype(str)
    else:
        with timed("count_vectorize", log, out=args.out, n_docs=len(df), min_count=min_count, **meta):
            X, features = extract_ngrams_sparse(text.tolist(), ngram_range=(1, 3), min_count=min_count)
            sp.save_npz(counts_cache, X.tocsr())
            np.save(feats_cache, features)
    log.info("%d n-grams with count >= %d (nnz=%d)", len(features), min_count, X.nnz)

    # regex drop, rows in ascending pmid order, vocabulary cleaning (cell 5)
    with timed("clean_vocab", log, out=args.out, n_features=len(features), **meta):
        mask = ~pd.Series(features).str.contains(DROP_PATTERN, case=False, na=False, regex=True).to_numpy()
        features = features[mask]
        X = X[:, mask]
        X = X.tocsr()[df["pmid"].argsort().to_numpy()]
        X, features = clean_vocab_and_remap_X(X, features, force_keep=FORCE_KEEP)
        features = np.array(features)
    log.info("%d terms after regex drop + cleaning", len(features))

    with timed("manual_filter", log, out=args.out, **meta):
        whitelist, source = load_whitelist(args.corpus, args.whitelist)
        if whitelist is not None:
            mask = pd.Series(features).isin(whitelist).to_numpy()
            features = features[mask]
            X = X[:, mask]
    log.info("manual term list: %s -> %d terms kept; matrix %s", source, len(features), X.shape)
    if len(features) == 0:
        raise RuntimeError("no n-gram survived the filters; check --corpus/--whitelist/--min-count")

    np.save(mat_path, X)
    np.save(lab_path, features)
    log.info("saved %s %s and %s", mat_path, X.shape, lab_path)

    # unit-norm SPECTER2 embeddings of the terms (cell 9)
    with timed("load_specter", log, out=args.out, **meta):
        specter = Specter(device=str(device))
    with timed("encode_ngrams", log, out=args.out, n_terms=len(features), batch_size=batch_size, amp=args.amp_dtype, device=str(device), **meta):
        ngram_emb = encode_texts(specter, features.tolist(), batch_size=batch_size, amp_dtype=args.amp_dtype, desc="specter ngrams")
        ngram_emb = ngram_emb / ngram_emb.norm(dim=1)[:, None]
    torch.save(ngram_emb, emb_path)
    log.info("saved %s %s", emb_path, tuple(ngram_emb.shape))
    return 0


if __name__ == "__main__":
    sys.exit(main())
