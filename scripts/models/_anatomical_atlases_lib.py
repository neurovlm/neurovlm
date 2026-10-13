"""N-gram miner, spatial-coherence scores and parcel n-gram statistics for 11_anatomical_atlases.py.

Cells 6, 7 and 14 of docs/03_models/11_anatomical_atlases.ipynb, without the options the stage does not use.
"""
from __future__ import annotations

import re
import unicodedata
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
import xxhash
from joblib import Parallel, delayed
from scipy.sparse import coo_matrix, csr_matrix

TOKEN_RE = re.compile(r"[a-z0-9]+(?:'[a-z0-9]+)?", flags=re.IGNORECASE)

# an n-gram may not start or end with one of these
BOUNDARY_STOPWORDS = frozenset({
    "a", "an", "and", "are", "as", "at", "be", "been", "being", "by",
    "for", "from", "has", "have", "having", "in", "into", "is", "it",
    "its", "of", "on", "or", "our", "that", "the", "their", "these",
    "this", "those", "to", "was", "were", "with", "without",
})

# generic one-word labels dropped from the n-gram features
BAD_LABEL_RE = re.compile(
    r"^(?:"
    r"study|studies|results|result|method|methods|analysis|analyses|"
    r"significant|significantly|activation|activations|activated|"
    r"increase|increased|decrease|decreased|effect|effects|"
    r"task|tasks|control|controls|condition|conditions|"
    r"subjects|patients|participants|group|groups|"
    r"brain|cerebral|regional|local"
    r")$"
)


def _normalize_text(text):
    text = unicodedata.normalize("NFKC", text)
    for ch in "‐–—-/\\":  # hyphens, dashes and slashes -> space
        text = text.replace(ch, " ")
    return text.casefold()


def _splitmix64(x):
    z = np.uint64(x) + np.uint64(0x9E3779B97F4A7C15)
    z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    return z ^ (z >> np.uint64(31))


def _parallel(n_jobs):
    return Parallel(n_jobs=n_jobs, backend="loky", batch_size=1, prefer="processes")


class FastNgramMiner:
    """Hashed n-grams: document frequencies, example phrases and a document x n-gram incidence matrix."""

    def __init__(self, *, n_min=1, n_max=7):
        self.n_min, self.n_max = n_min, n_max
        self._position_salts = np.array([_splitmix64(i + 1) for i in range(n_max)], dtype=np.uint64)
        self._n_salts = np.array([_splitmix64(10_000 + i) for i in range(n_max + 1)], dtype=np.uint64)

    def tokenize(self, text):
        return [t for t in TOKEN_RE.findall(_normalize_text(text)) if len(t) >= 2]

    def _windows(self, tokens):
        """Yield (n, hash of every n-token window, window has valid boundary words) for each n."""
        # xxhash >= 3 needs bytes; the hashes only need to be stable within a run
        token_hashes = np.fromiter((xxhash.xxh3_64_intdigest(t.encode("utf-8")) for t in tokens),
                                   dtype=np.uint64, count=len(tokens))
        ok = np.fromiter((t not in BOUNDARY_STOPWORDS and not t.isdigit() for t in tokens), dtype=bool, count=len(tokens))
        for n in range(self.n_min, min(self.n_max, len(tokens)) + 1):
            windows = np.lib.stride_tricks.sliding_window_view(token_hashes, n)
            hashes = (windows * self._position_salts[:n]).sum(axis=1, dtype=np.uint64) ^ self._n_salts[n]
            yield n, hashes, ok[:len(tokens) - n + 1] & ok[n - 1:]

    def ngram_hashes(self, text):
        """Sorted unique n-gram hashes of one document."""
        parts = [hashes[valid] for _, hashes, valid in self._windows(self.tokenize(text))]
        return np.unique(np.concatenate(parts)) if parts else np.empty(0, dtype=np.uint64)

    def iter_ngram_hits(self, text):
        """Yield (hash, phrase, n, start_token, end_token) for every n-gram of one document."""
        tokens = self.tokenize(text)
        for n, hashes, valid in self._windows(tokens):
            for start in np.flatnonzero(valid):
                yield int(hashes[start]), " ".join(tokens[start:start + n]), n, int(start), int(start + n)

    def count_document_frequency_parallel(self, records, *, n_jobs=-1, chunk_docs=20_000):
        """[{"hash", "df"}, ...] over (doc_id, text) records, most frequent first."""
        chunks = [records[i:i + chunk_docs] for i in range(0, len(records), chunk_docs)]
        counts = Counter()
        for values, freqs in _parallel(n_jobs)(delayed(_count_df_chunk)(self, chunk) for chunk in chunks):
            counts.update({int(h): int(c) for h, c in zip(values, freqs)})
        return [{"hash": int(h), "df": int(c)} for h, c in counts.most_common()]

    def collect_phrase_examples_parallel(self, records, candidate_hashes, *, n_jobs=-1, chunk_docs=20_000,
                                         max_examples_per_hash=3):
        """Up to ``max_examples_per_hash`` example phrases (dicts) per candidate hash."""
        candidate_set = frozenset(int(h) for h in candidate_hashes)
        chunks = [records[i:i + chunk_docs] for i in range(0, len(records), chunk_docs)]
        merged = defaultdict(list)
        for chunk_examples in _parallel(n_jobs)(delayed(_collect_examples_chunk)(self, chunk, candidate_set, max_examples_per_hash)
                                                for chunk in chunks):
            for h, vals in chunk_examples.items():
                merged[h].extend(vals[:max_examples_per_hash - len(merged[h])])
        return [ex for vals in merged.values() for ex in vals]

    def build_candidate_incidence_parallel(self, texts, candidate_hashes, *, n_jobs=-1, chunk_docs=5_000):
        """Sparse float32 document x candidate matrix, 1 where the candidate occurs in the document."""
        candidate_hashes = np.asarray(candidate_hashes, dtype=np.uint64)
        order = np.argsort(candidate_hashes)
        sorted_hashes = candidate_hashes[order]
        sorted_cols = np.arange(len(candidate_hashes), dtype=np.int64)[order]
        results = _parallel(n_jobs)(delayed(_candidate_incidence_chunk)(self, texts[i:i + chunk_docs], i, sorted_hashes, sorted_cols)
                                    for i in range(0, len(texts), chunk_docs))
        results = [(r, c) for r, c in results if r.size]
        if not results:
            return csr_matrix((len(texts), len(candidate_hashes)), dtype=np.float32)
        rows = np.concatenate([r for r, _ in results])
        cols = np.concatenate([c for _, c in results])
        mat = coo_matrix((np.ones(len(rows), dtype=np.float32), (rows, cols)),
                         shape=(len(texts), len(candidate_hashes)), dtype=np.float32).tocsr()
        mat.sum_duplicates()
        mat.data[:] = 1
        return mat


def _count_df_chunk(miner, chunk):
    return np.unique(np.concatenate([miner.ngram_hashes(text) for _, text in chunk]), return_counts=True)


def _collect_examples_chunk(miner, chunk, candidate_hashes, max_examples_per_hash):
    examples = defaultdict(list)
    for doc_id, text in chunk:
        seen_in_doc = set()
        for h, phrase, n, start, end in miner.iter_ngram_hits(text):
            if h not in candidate_hashes or h in seen_in_doc or len(examples[h]) >= max_examples_per_hash:
                continue
            examples[h].append({"hash": h, "phrase": phrase, "n": n, "doc_id": doc_id, "start_token": start, "end_token": end})
            seen_in_doc.add(h)
    return dict(examples)


def _candidate_incidence_chunk(miner, texts, offset, sorted_hashes, sorted_cols):
    rows, cols = [], []
    for i, text in enumerate(texts):
        hashes = miner.ngram_hashes(text)
        pos = np.searchsorted(sorted_hashes, hashes)
        valid = pos < len(sorted_hashes)
        pos, hashes = pos[valid], hashes[valid]
        c = sorted_cols[pos[sorted_hashes[pos] == hashes]]
        rows.append(np.full(len(c), offset + i, dtype=np.int64))
        cols.append(c)
    return np.concatenate(rows), np.concatenate(cols)


def score_spatial_coherence_from_incidence(*, incidence, maps, candidate_hashes, candidate_block_size=2048, eps=1e-8):
    """Score candidate n-grams by how alike the maps of the documents containing them are.

    coherence = ||sum of the L2-normalised maps of the documents with the n-gram|| / df; score = coherence * log(1 + df).
    """
    maps = np.asarray(maps, dtype=np.float32)
    maps_norm = maps / np.maximum(np.linalg.norm(maps, axis=1, keepdims=True), eps)
    candidate_hashes = np.asarray(candidate_hashes, dtype=np.uint64)
    out = []
    for start in range(0, incidence.shape[1], candidate_block_size):
        block = incidence[:, start:start + candidate_block_size]
        dfs = np.asarray(block.sum(axis=0)).ravel().astype(np.int64)
        sum_norms = np.linalg.norm(np.asarray(block.T @ maps_norm, dtype=np.float32), axis=1)
        coherence = np.divide(sum_norms, np.maximum(dfs, 1), out=np.zeros_like(sum_norms, dtype=np.float32), where=dfs > 0)
        score = coherence * np.log1p(dfs)
        for j, (df, norm, coh, s) in enumerate(zip(dfs, sum_norms, coherence, score)):
            out.append({"hash": int(candidate_hashes[start + j]), "df": int(df), "sum_norm": float(norm),
                        "coherence": float(coh), "score": float(s)})
    out.sort(key=lambda r: (r["score"], r["coherence"], r["df"]), reverse=True)
    return out


def _best_label(examples):
    """Most frequent example phrase (ties: more words, then longer) and its number of tokens."""
    counts, n_tokens = defaultdict(int), {}
    for ex in examples:
        phrase = str(ex["phrase"]).strip()
        if phrase:
            counts[phrase] += 1
            n_tokens[phrase] = int(ex["n"])
    if not counts:
        return "", 0
    label = max(counts, key=lambda p: (counts[p], len(p.split()), len(p)))
    return label, n_tokens[label]


def _valid_label(label, min_chars=3, max_chars=120):
    label = label.strip()
    return min_chars <= len(label) <= max_chars and not (len(label.split()) == 1 and BAD_LABEL_RE.match(label))


def make_filtered_ngram_matrix(*, scores, examples, incidence, candidate_hashes, min_df, max_df_frac, top_k, min_n, max_n):
    """Top-k scored n-grams -> (labels, bool document x label csr matrix, one metadata dict per label)."""
    n_docs = incidence.shape[0]
    hash_to_col = {int(h): i for i, h in enumerate(np.asarray(candidate_hashes, dtype=np.uint64))}
    examples_by_hash = defaultdict(list)
    for ex in examples:
        examples_by_hash[int(ex["hash"])].append(ex)
    max_df = int(np.floor(max_df_frac * n_docs))

    rows, seen_labels = [], set()
    for row in sorted(scores, key=lambda r: (r["score"], r["coherence"], r["df"]), reverse=True):
        h, df = int(row["hash"]), int(row["df"])
        if df < min_df or df > max_df:
            continue
        label, n = _best_label(examples_by_hash.get(h, []))
        if not label or not min_n <= n <= max_n or not _valid_label(label) or label.casefold() in seen_labels:
            continue
        seen_labels.add(label.casefold())
        rows.append({"hash": h, "label": label, "n": n, "df": df, "coherence": float(row["coherence"]),
                     "score": float(row["score"]), "original_col": hash_to_col[h]})
        if len(rows) >= top_k:
            break

    labels = np.array([r["label"] for r in rows], dtype=object)
    selected_cols = np.array([r["original_col"] for r in rows], dtype=np.int64)
    if selected_cols.size == 0:
        return labels, csr_matrix((n_docs, 0), dtype=bool), rows
    X_bool = incidence[:, selected_cols].astype(bool)
    X_bool.eliminate_zeros()
    for j, row in enumerate(rows):
        row["column"] = j
    return labels, X_bool, rows


def build_paper_parcel_overlap_matrix(maps, masks, *, min_overlap, top_k_parcels_per_paper, chunk_docs=2048, eps=1e-8):
    """Paper x parcel csr matrix: fraction of each (positive) map's mass inside each of its top-k parcels, >= min_overlap."""
    maps = np.maximum(np.asarray(maps, dtype=np.float32), 0.0)
    masks_t = np.ascontiguousarray(masks.T)
    n_docs = maps.shape[0]
    rows, cols, vals = [], [], []
    for start in range(0, n_docs, chunk_docs):
        block = maps[start:start + chunk_docs]
        scores = np.asarray((block @ masks_t) / block.sum(axis=1, keepdims=True).clip(min=eps), dtype=np.float32)
        k = min(top_k_parcels_per_paper, scores.shape[1])
        keep_idx = np.argpartition(scores, kth=scores.shape[1] - k, axis=1)[:, -k:]
        keep = np.zeros(scores.shape, dtype=bool)
        keep[np.arange(scores.shape[0])[:, None], keep_idx] = True
        scores = np.where(keep, scores, 0.0)
        hit_r, hit_c = np.nonzero(scores >= min_overlap)
        rows.append(hit_r.astype(np.int64) + start)
        cols.append(hit_c.astype(np.int64))
        vals.append(scores[hit_r, hit_c].astype(np.float32))
    out = csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                     shape=(n_docs, masks.shape[0]), dtype=np.float32)
    out.sum_duplicates()
    return out


def _top_ngrams_for_parcel(row_counts, row_weighted, labels, global_ngram_df, n_docs_total, n_docs_parcel, top_k, alpha=1.0):
    """The parcel's top-k n-grams by count x max(log lift, 0), with their stats."""
    term_cols = row_counts.indices
    counts = row_counts.data.astype(np.float32)
    weighted_lookup = dict(zip(row_weighted.indices.tolist(), row_weighted.data.tolist()))
    weighted_counts = np.array([weighted_lookup.get(int(c), 0.0) for c in term_cols], dtype=np.float32)
    # add-alpha smoothed P(n-gram | parcel papers) / P(n-gram | all papers)
    p_region = (counts + alpha) / (float(n_docs_parcel) + 2.0 * alpha)
    p_global = (global_ngram_df[term_cols].astype(np.float32) + alpha) / (float(n_docs_total) + 2.0 * alpha)
    lift = p_region / np.maximum(p_global, 1e-12)
    log_lift = np.log(lift)
    rank_score = counts * np.maximum(log_lift, 0.0)

    ngrams, stats = [], []
    for idx in np.argsort(rank_score)[::-1][:top_k]:
        col = int(term_cols[idx])
        label = str(labels[col])
        ngrams.append(label)
        stats.append({
            "ngram": label,
            "ngram_col": col,
            "count": int(counts[idx]),
            "weighted_count": float(weighted_counts[idx]),
            "global_df": int(global_ngram_df[col]),
            "lift": float(lift[idx]),
            "log_lift": float(log_lift[idx]),
            "rank_score": float(rank_score[idx]),
        })
    return ngrams, stats


def make_parcel_ngram_dataframe(*, brain_maps, atlas_masks, atlas_parcel_names, labels, X_bool, pmids, min_overlap,
                                top_k_parcels_per_paper, top_k_ngrams_per_parcel, max_pmids_per_parcel):
    """One row per parcel: its top n-grams (with stats) and the PMIDs of its best-overlapping papers."""
    maps = np.asarray(brain_maps, dtype=np.float32)
    masks = (np.asarray(atlas_masks) > 0).astype(np.float32)
    scores = build_paper_parcel_overlap_matrix(maps, masks, min_overlap=min_overlap,
                                               top_k_parcels_per_paper=top_k_parcels_per_paper)
    in_parcel = scores.astype(bool)
    in_parcel.eliminate_zeros()

    # numeric sparse products, not bool @ bool
    X_count = X_bool.astype(np.float32)
    count_matrix = (in_parcel.astype(np.float32).T @ X_count).tocsr()
    count_matrix.sum_duplicates()
    weighted_matrix = (scores.T @ X_count).tocsr()
    weighted_matrix.sum_duplicates()

    parcel_doc_counts = np.asarray(in_parcel.sum(axis=0)).ravel().astype(np.int64)
    global_ngram_df = np.asarray(X_bool.sum(axis=0)).ravel().astype(np.int64)
    mask_sizes = masks.sum(axis=1).astype(np.int64)
    scores_csc = scores.tocsc()

    rows = []
    for j, name in enumerate(atlas_parcel_names):
        n_docs = int(parcel_doc_counts[j])
        row_counts = count_matrix.getrow(j)
        ngrams, stats, top_pmids = [], [], []
        if n_docs:
            ngrams, stats = _top_ngrams_for_parcel(row_counts, weighted_matrix.getrow(j), labels, global_ngram_df,
                                                   maps.shape[0], n_docs, top_k_ngrams_per_parcel)
            col = scores_csc.getcol(j)
            top_pmids = [pmids[int(col.indices[i])] for i in np.argsort(col.data)[::-1][:max_pmids_per_parcel]]
        rows.append({
            "atlas_parcel_names": name,
            "list_of_ngrams": ngrams,
            "list_of_pmids": top_pmids,
            "n_parcel_pmids": n_docs,
            "n_ngrams": len(ngrams),
            "n_candidate_ngram_hits": int(row_counts.nnz),
            "max_ngram_count": float(row_counts.data.max()) if row_counts.nnz else 0.0,
            "ngram_stats": stats,
            "empty_mask": bool(mask_sizes[j] == 0),
        })
    return pd.DataFrame(rows)
