"""Semantic evaluation helpers for NeuroVLM experiments.

The functions in this module keep exact PMID retrieval available as a
diagnostic while making semantic ranking evaluations easy to reuse across the
pretrained NeuroVLM MLP baseline and newer ALE/CNN/GNN models.
"""

from __future__ import annotations

import gzip
import json
import math
import pickle
import random
import warnings
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import nibabel as nib
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from nilearn.image import resample_img
from tqdm import tqdm

from neurovlm.metrics import normalized_recall_curve_auc, retrieval_ranks


DEFAULT_NETWORK_LABEL_DEFINITIONS: dict[str, tuple[str, str]] = {
    "visual": (
        "Visual",
        "Visual network including occipital visual cortex and extrastriate regions for visual perception, object recognition, motion, and visuospatial processing.",
    ),
    "motor": (
        "Motor / Sensorimotor",
        "Motor and sensorimotor network including precentral, postcentral, supplementary motor, and somatosensory regions for movement and bodily sensation.",
    ),
    "auditory": (
        "Auditory",
        "Auditory network including primary auditory cortex, superior temporal regions, and insula for sound perception, pitch, localization, and speech-sound processing.",
    ),
    "language": (
        "Language",
        "Language network including inferior frontal, temporal, and temporoparietal regions for semantic, syntactic, phonological, and sentence-level processing.",
    ),
    "default_mode": (
        "Default Mode",
        "Default mode network including medial prefrontal cortex, posterior cingulate, precuneus, and angular gyrus for memory, self-reference, future simulation, and social cognition.",
    ),
    "frontoparietal_control": (
        "Frontoparietal Control",
        "Frontoparietal control network including lateral prefrontal and posterior parietal regions for executive control, working memory, flexible task rules, and goal-directed cognition.",
    ),
    "attention": (
        "Attention",
        "Dorsal and ventral attention network including intraparietal sulcus, superior parietal lobule, frontal eye fields, and temporoparietal regions for selective attention and orienting.",
    ),
    "cingulo_opercular": (
        "Cingulo-Opercular / Salience",
        "Cingulo-opercular and salience network including dorsal anterior cingulate, anterior insula, frontal operculum, and thalamus for salience detection, performance monitoring, interoception, and stable task control.",
    ),
}

NETWORK_TERM_COLUMNS: tuple[str, ...] = ("network_name", "cognitive_terms", "region_terms")
DEFAULT_MESH_RANKING_NODE_TYPES: tuple[str, ...] = (
    "cognitive_construct",
    "biological_process",
    "anatomical_region",
    "disorder",
)
EXCLUDED_MESH_RANKING_NODE_TYPES: tuple[str, ...] = (
    "molecular",
    "organism",
    "other",
    "method",
    "demographic",
)


def normalize_network_term(term: str) -> str:
    """Normalize a network/cognitive/region term for deduplication and lookup."""

    import re

    text = re.sub(r"\([^)]*\)", " ", str(term).lower())
    text = text.replace("&", " and ")
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def split_network_term_cell(value: Any) -> list[str]:
    """Split semicolon-separated network term cells while preserving display text."""

    if value is None or (isinstance(value, float) and math.isnan(value)):
        return []
    if pd.isna(value):
        return []
    out: list[str] = []
    for piece in str(value).replace("|", ";").split(";"):
        term = " ".join(piece.strip().split())
        if term and term.lower() not in {"nan", "none", "unknown"}:
            out.append(term)
    return out


def _as_bool_split_value(value: Any) -> bool:
    if pd.isna(value):
        return False
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer, float, np.floating)):
        return bool(int(value))
    return str(value).strip().lower() in {"1", "true", "t", "yes", "y"}


def save_official_pubmed_splits(
    pubmed_df: pd.DataFrame,
    aligned_pmids: Sequence[str],
    out_dir: str | Path,
    *,
    random_state: int = 0,
    random_val_frac: float = 0.1,
    random_test_frac: float = 0.1,
) -> dict[str, np.ndarray]:
    """Resolve train/val/test PMIDs from publication split columns and save them.

    The official ``train``, ``val``, and ``test`` columns are used whenever
    available. If all split columns are missing, a deterministic random split is
    used and a warning is emitted. If only some columns are missing, the present
    official columns are used and missing split manifests are written as empty
    lists, because the official test set should remain untouched.
    """

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    pmids = np.asarray(aligned_pmids).astype(str)
    df = pubmed_df.copy()
    if "pmid" not in df.columns:
        raise KeyError("pubmed_df must contain a 'pmid' column.")
    df["pmid"] = df["pmid"].astype(str)
    split_columns = [c for c in ("train", "val", "test") if c in df.columns]

    splits: dict[str, np.ndarray] = {"train": np.array([], dtype=str), "val": np.array([], dtype=str), "test": np.array([], dtype=str)}
    if split_columns:
        by_pmid = df.set_index("pmid")
        for split in ("train", "val", "test"):
            if split not in by_pmid.columns:
                warnings.warn(f"PubMed split column '{split}' is missing; writing an empty {split} split manifest.")
                continue
            flags = np.asarray([_as_bool_split_value(by_pmid.at[p, split]) if p in by_pmid.index else False for p in pmids])
            splits[split] = pmids[flags]
    else:
        warnings.warn(
            "PubMed dataframe has no train/val/test columns. Falling back to a deterministic random split; "
            "do not compare these metrics against official-split runs."
        )
        rng = np.random.default_rng(random_state)
        order = rng.permutation(len(pmids))
        n_test = int(round(len(pmids) * random_test_frac))
        n_val = int(round(len(pmids) * random_val_frac))
        splits["test"] = pmids[order[:n_test]]
        splits["val"] = pmids[order[n_test : n_test + n_val]]
        splits["train"] = pmids[order[n_test + n_val :]]

    # Ensure no accidental overlap when official boolean columns are clean.
    for split, split_pmids in splits.items():
        with (out_path / f"{split}_pmids.json").open("w") as f:
            json.dump(split_pmids.astype(str).tolist(), f, indent=2)

    counts = {split: int(len(values)) for split, values in splits.items()}
    print("PubMed split counts:", counts)
    if len(splits["test"]) == 0:
        raise RuntimeError("No held-out PubMed test PMIDs were resolved.")
    return splits


def official_split_positions(
    pubmed_df: pd.DataFrame,
    aligned_pmids: Sequence[str],
    out_dir: str | Path | None = None,
    *,
    random_state: int = 0,
    random_val_frac: float = 0.1,
    random_test_frac: float = 0.1,
) -> dict[str, np.ndarray]:
    """Return train/val/test positions for ``aligned_pmids``.

    Uses publication dataframe split columns when present. If no official split
    columns exist, falls back to the same deterministic random split used by
    :func:`save_official_pubmed_splits`.
    """

    pmids = np.asarray(aligned_pmids).astype(str)
    if out_dir is None:
        import tempfile

        tmp = tempfile.TemporaryDirectory()
        splits = save_official_pubmed_splits(
            pubmed_df,
            pmids,
            tmp.name,
            random_state=random_state,
            random_val_frac=random_val_frac,
            random_test_frac=random_test_frac,
        )
        tmp.cleanup()
    else:
        splits = save_official_pubmed_splits(
            pubmed_df,
            pmids,
            out_dir,
            random_state=random_state,
            random_val_frac=random_val_frac,
            random_test_frac=random_test_frac,
        )

    pos: dict[str, np.ndarray] = {}
    for split, split_pmids in splits.items():
        split_set = set(np.asarray(split_pmids).astype(str).tolist())
        pos[split] = np.flatnonzero(np.asarray([p in split_set for p in pmids]))
    return pos


@torch.no_grad()
def exact_pmid_retrieval_metrics(similarity: torch.Tensor, ks: Sequence[int] = (1, 5, 10, 50)) -> dict[str, float]:
    """Compute strict diagonal exact-paper retrieval metrics."""

    sim = similarity.float()
    ranks = retrieval_ranks(sim).float()
    n = float(sim.shape[0])
    auc = float(_recall_curve_auc_from_ranks(ranks, int(n)))
    out: dict[str, float] = {
        "paper_recall_curve_auc": auc,
        "normalized_k_recall_curve_auc": auc,
        "mrr": float((1.0 / ranks).mean().item()),
        "median_rank": float(ranks.median().item()),
        "mean_rank": float(ranks.mean().item()),
    }
    for k in ks:
        out[f"recall@{k}"] = float((ranks <= k).float().mean().item())
        out[f"random_recall@{k}"] = min(float(k) / n, 1.0)
    return out


def _recall_curve_auc_from_ranks(ranks: torch.Tensor, n: int) -> float:
    counts = torch.bincount((ranks.long() - 1), minlength=n)
    curve = counts.cumsum(0).float() / float(n)
    return normalized_recall_curve_auc(curve)


def exact_recall_curve(similarity: torch.Tensor) -> torch.Tensor:
    ranks = retrieval_ranks(similarity.float()).long()
    counts = torch.bincount(ranks - 1, minlength=similarity.shape[0])
    return counts.cumsum(0).float() / float(similarity.shape[0])


def load_network_maps(path: str | Path | None = None) -> list[dict[str, Any]]:
    """Load raw network arrays from ``networks_arrays.pkl.gz``.

    Returns records with ``atlas``, ``network_label``, and ``image`` fields.
    """

    if path is None:
        from neurovlm.retrieval_resources import _load_networks

        networks = _load_networks()
    else:
        with gzip.open(path, "rb") as f:
            networks = pickle.load(f)

    records: list[dict[str, Any]] = []
    for atlas, atlas_payload in networks.items():
        if isinstance(atlas_payload, Mapping) and "array" in atlas_payload:
            atlas_payload = {str(atlas): atlas_payload}
        for label, payload in atlas_payload.items():
            if not isinstance(payload, Mapping) or "array" not in payload or "affine" not in payload:
                continue
            records.append(
                {
                    "atlas": str(atlas),
                    "network_label": str(label),
                    "image": nib.Nifti1Image(np.asarray(payload["array"]), affine=np.asarray(payload["affine"])),
                }
            )
    return records


def preprocess_network_maps(
    network_records: Sequence[Mapping[str, Any]],
    masker: Any,
    *,
    exclude_atlases: Iterable[str] = (),
) -> list[dict[str, Any]]:
    """Resample and threshold raw network maps using the NeuroVLM reference logic."""

    exclude = {str(x) for x in exclude_atlases}
    out: list[dict[str, Any]] = []
    for rec in tqdm(network_records, total=len(network_records), desc="Preprocessing network maps"):
        if str(rec["atlas"]) in exclude:
            continue
        img = rec["image"]
        arr = img.get_fdata()
        if len(np.unique(arr)) == 2:
            img_resampled = resample_img(
                img,
                target_affine=masker.affine_,
                interpolation="nearest",
                force_resample=True,
                copy_header=True,
            )
        else:
            img_resampled = resample_img(
                img,
                target_affine=masker.affine_,
                force_resample=True,
                copy_header=True,
            )
            arr_resampled = img_resampled.get_fdata()
            arr_resampled[arr_resampled < 0] = 0.0
            thresh = np.percentile(arr_resampled.flatten(), 95)
            arr_resampled[arr_resampled < thresh] = 0.0
            arr_resampled[arr_resampled >= thresh] = 1.0
            img_resampled = nib.Nifti1Image(arr_resampled, affine=masker.affine_)
        out.append({**rec, "image": img_resampled})
    return out


@torch.no_grad()
def encode_network_maps(
    model_name: str,
    networks_resampled: Sequence[Mapping[str, Any]],
    model_bundle: Mapping[str, Any],
    preprocess_config: Mapping[str, Any] | None = None,
    *,
    batch_size: int = 128,
    device: str | torch.device = "cpu",
) -> torch.Tensor:
    """Encode resampled networks with a model-specific encoder.

    For the ``neurovlm_mlp`` baseline, ``model_bundle`` should contain
    ``masker``, ``autoencoder``, and ``proj_head_image``. For newer models,
    pass an ``encode_fn`` callable that accepts ``(image, model_bundle,
    preprocess_config)`` and returns one shared-space embedding.
    """

    name = model_name.lower()
    cfg = dict(preprocess_config or {})
    dev = torch.device(device)
    embeddings: list[torch.Tensor] = []

    if name in {"neurovlm_mlp", "mlp", "baseline"}:
        masker = model_bundle["masker"]
        autoencoder = model_bundle["autoencoder"].to(dev).eval()
        proj_head_image = model_bundle["proj_head_image"].to(dev).eval()
        flats: list[torch.Tensor] = []
        for rec in tqdm(networks_resampled, total=len(networks_resampled), desc="Encoding network maps"):
            x = torch.from_numpy(masker.transform(rec["image"])).float().to(dev)
            flats.append(x)
            if len(flats) >= batch_size:
                batch = torch.vstack(flats)
                embeddings.append(proj_head_image(autoencoder.encoder(batch)).detach().cpu())
                flats = []
        if flats:
            batch = torch.vstack(flats)
            embeddings.append(proj_head_image(autoencoder.encoder(batch)).detach().cpu())
    else:
        encode_fn = model_bundle.get("encode_fn")
        if encode_fn is None:
            raise ValueError(
                f"model_bundle for {model_name!r} must provide an encode_fn(image, model_bundle, preprocess_config)."
            )
        for rec in tqdm(networks_resampled, total=len(networks_resampled), desc=f"Encoding network maps with {model_name}"):
            emb = encode_fn(rec["image"], model_bundle, cfg)
            embeddings.append(torch.as_tensor(emb).detach().cpu().reshape(1, -1))

    return F.normalize(torch.vstack(embeddings).float(), dim=1, eps=1e-8)


def load_network_label_table(path: str | Path | None = None) -> pd.DataFrame:
    """Load the reusable network-label ground truth CSV."""

    module_repo_root = Path(__file__).resolve().parents[2]
    search_roots = [Path.cwd(), module_repo_root]
    search_roots.extend(Path.cwd().parents)

    candidates: list[Path] = []
    if path is not None:
        path_obj = Path(path).expanduser()
        if path_obj.is_absolute():
            candidates.append(path_obj)
        else:
            candidates.extend(root / path_obj for root in search_roots)

    relative_candidates = [
        Path("docs/03_evaluation/network_test_set_labels.csv"),
        Path("experiments/data/networks_labels/network_test_set_labels.csv"),
    ]
    candidates.extend(root / rel for root in search_roots for rel in relative_candidates)

    seen: set[Path] = set()
    for candidate in candidates:
        candidate = candidate.resolve()
        if candidate in seen:
            continue
        seen.add(candidate)
        if candidate.exists():
            return pd.read_csv(candidate)
    searched = "\n".join(str(p) for p in sorted(seen))
    raise FileNotFoundError(
        "Could not find network_test_set_labels.csv. Pass NETWORK_LABELS_CSV explicitly. "
        f"Searched:\n{searched}"
    )


def build_network_label_corpus(labels_df: pd.DataFrame) -> pd.DataFrame:
    """Build candidate label texts as ``term [SEP] definition``."""

    if {"network_key", "network_name"}.issubset(labels_df.columns):
        rows = []
        for key, grp in labels_df.dropna(subset=["network_key"]).groupby("network_key", sort=True):
            key = str(key)
            if key == "unknown":
                continue
            first = grp.iloc[0]
            name = str(first.get("network_name", key))
            definition = str(first.get("long_definition") or first.get("short_definition") or DEFAULT_NETWORK_LABEL_DEFINITIONS.get(key, (name, ""))[1])
            rows.append({"network_key": key, "network_name": name, "definition": definition, "text": f"{name} [SEP] {definition}"})
        return pd.DataFrame(rows)

    rows = []
    for key, (name, definition) in DEFAULT_NETWORK_LABEL_DEFINITIONS.items():
        rows.append({"network_key": key, "network_name": name, "definition": definition, "text": f"{name} [SEP] {definition}"})
    return pd.DataFrame(rows)


def find_network_term_corpus(path: str | Path | None = None) -> Path | None:
    """Find the deduplicated network-term definition corpus if it exists."""

    module_repo_root = Path(__file__).resolve().parents[2]
    search_roots = [Path.cwd(), module_repo_root]
    search_roots.extend(Path.cwd().parents)

    candidates: list[Path] = []
    if path is not None:
        path_obj = Path(path).expanduser()
        if path_obj.is_absolute():
            candidates.append(path_obj)
        else:
            candidates.extend(root / path_obj for root in search_roots)
    relative_candidates = [
        Path("experiments/evaluation_resources/networks_labels/network_terms_with_definitions.csv"),
        Path("networks_labels/network_terms_with_definitions.csv"),
    ]
    candidates.extend(root / rel for root in search_roots for rel in relative_candidates)

    seen: set[Path] = set()
    for candidate in candidates:
        candidate = candidate.resolve()
        if candidate in seen:
            continue
        seen.add(candidate)
        if candidate.exists():
            return candidate
    return None


def load_network_term_corpus(path: str | Path | None = None) -> pd.DataFrame | None:
    """Load the deduplicated network term corpus produced by the builder notebook."""

    found = find_network_term_corpus(path)
    if found is None:
        return None
    corpus = pd.read_csv(found)
    if "term" not in corpus.columns:
        raise KeyError(f"{found} must contain a 'term' column.")
    if "normalized_term" not in corpus.columns:
        corpus["normalized_term"] = corpus["term"].map(normalize_network_term)
    if "definition" not in corpus.columns:
        corpus["definition"] = ""
    corpus["definition"] = corpus["definition"].fillna("").astype(str)
    corpus["text"] = [
        f"{term} [SEP] {definition}" if definition else str(term)
        for term, definition in zip(corpus["term"].astype(str), corpus["definition"].astype(str))
    ]
    corpus = corpus.drop_duplicates("normalized_term", keep="first").reset_index(drop=True)
    return corpus


def build_network_term_corpus_from_label_table(
    labels_df: pd.DataFrame,
    definitions: Mapping[str, str] | None = None,
) -> pd.DataFrame:
    """Build a deduplicated term corpus from network_name/cognitive_terms/region_terms."""

    definitions = definitions or {}
    rows_by_norm: dict[str, dict[str, Any]] = {}
    for _, row in labels_df.iterrows():
        for column in NETWORK_TERM_COLUMNS:
            if column not in labels_df.columns:
                continue
            for term in split_network_term_cell(row.get(column)):
                norm = normalize_network_term(term)
                if not norm:
                    continue
                entry = rows_by_norm.setdefault(
                    norm,
                    {
                        "term": term,
                        "normalized_term": norm,
                        "source_columns": set(),
                        "definition": "",
                    },
                )
                entry["source_columns"].add(column)
    rows = []
    for norm, entry in sorted(rows_by_norm.items(), key=lambda item: item[1]["term"].lower()):
        definition = definitions.get(entry["term"], "") or definitions.get(norm, "")
        rows.append(
            {
                "term": entry["term"],
                "normalized_term": norm,
                "source_columns": "; ".join(sorted(entry["source_columns"])),
                "definition": definition,
                "text": f"{entry['term']} [SEP] {definition}" if definition else entry["term"],
            }
        )
    return pd.DataFrame(rows)


def align_network_ground_truth(
    network_records: Sequence[Mapping[str, Any]],
    labels_df: pd.DataFrame,
) -> pd.DataFrame:
    """Attach reusable ground-truth labels to network map records."""

    if "raw_network_label" not in labels_df.columns or "network_key" not in labels_df.columns:
        raise KeyError("network label CSV must contain raw_network_label and network_key columns.")
    lookup = labels_df.set_index("raw_network_label")
    rows = []
    for i, rec in enumerate(network_records):
        raw_label = str(rec["network_label"])
        row = {
            "row_index": i,
            "atlas": str(rec["atlas"]),
            "raw_network_label": raw_label,
            "network_key": "unknown",
            "network_name": "Unknown",
        }
        if raw_label in lookup.index:
            hit = lookup.loc[raw_label]
            if isinstance(hit, pd.DataFrame):
                hit = hit.iloc[0]
            row["network_key"] = str(hit.get("network_key", "unknown"))
            row["network_name"] = str(hit.get("network_name", "Unknown"))
        rows.append(row)
    return pd.DataFrame(rows)


def align_network_term_ground_truth(
    network_records: Sequence[Mapping[str, Any]],
    labels_df: pd.DataFrame,
    term_corpus: pd.DataFrame,
) -> pd.DataFrame:
    """Attach multi-positive term labels to each raw network map record."""

    if "raw_network_label" not in labels_df.columns:
        raise KeyError("network label CSV must contain a raw_network_label column.")
    if "normalized_term" not in term_corpus.columns:
        term_corpus = term_corpus.copy()
        term_corpus["normalized_term"] = term_corpus["term"].map(normalize_network_term)

    corpus_norms = set(term_corpus["normalized_term"].astype(str))
    lookup = labels_df.set_index("raw_network_label")
    rows = []
    for i, rec in enumerate(network_records):
        raw_label = str(rec["network_label"])
        base = {
            "row_index": i,
            "atlas": str(rec["atlas"]),
            "raw_network_label": raw_label,
            "network_key": "unknown",
            "network_name": "Unknown",
            "true_network_terms": [],
        }
        if raw_label not in lookup.index:
            rows.append(base)
            continue
        hit = lookup.loc[raw_label]
        if isinstance(hit, pd.DataFrame):
            hit = hit.iloc[0]
        base["network_key"] = str(hit.get("network_key", "unknown"))
        base["network_name"] = str(hit.get("network_name", "Unknown"))
        terms: list[str] = []
        seen: set[str] = set()
        for column in NETWORK_TERM_COLUMNS:
            for term in split_network_term_cell(hit.get(column)):
                norm = normalize_network_term(term)
                if norm and norm in corpus_norms and norm not in seen:
                    terms.append(term)
                    seen.add(norm)
        base["true_network_terms"] = terms
        rows.append(base)
    return pd.DataFrame(rows)


def evaluate_network_labeling(
    network_embeddings: torch.Tensor,
    label_embeddings: torch.Tensor,
    truth_df: pd.DataFrame,
    label_corpus: pd.DataFrame,
    *,
    out_dir: str | Path | None = None,
    top_k_for_examples: int = 3,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Evaluate network map embeddings against canonical network labels."""

    from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
    from sklearn.preprocessing import label_binarize

    out_path = Path(out_dir) if out_dir is not None else None
    if out_path is not None:
        out_path.mkdir(parents=True, exist_ok=True)

    label_keys = label_corpus["network_key"].astype(str).tolist()
    label_names = label_corpus["network_name"].astype(str).tolist()
    key_to_idx = {key: i for i, key in enumerate(label_keys)}
    y_true = truth_df["network_key"].astype(str).map(key_to_idx)
    eval_mask = y_true.notna().to_numpy()
    y_true_idx = y_true[eval_mask].astype(int).to_numpy()

    sim = F.normalize(network_embeddings.float(), dim=1, eps=1e-8) @ F.normalize(label_embeddings.float(), dim=1, eps=1e-8).T
    sim_np = sim.detach().cpu().numpy()
    eval_scores = sim_np[eval_mask]
    order = np.argsort(-eval_scores, axis=1)
    y_pred = order[:, 0]

    metrics: dict[str, Any] = {
        "n_network_maps": int(len(truth_df)),
        "n_evaluated": int(eval_mask.sum()),
        "n_skipped_unknown": int((~eval_mask).sum()),
        "accuracy": float((y_pred == y_true_idx).mean()) if len(y_true_idx) else math.nan,
        "top_2_accuracy": float(np.mean([yt in row[:2] for yt, row in zip(y_true_idx, order)])) if len(y_true_idx) else math.nan,
        "top_3_accuracy": float(np.mean([yt in row[:3] for yt, row in zip(y_true_idx, order)])) if len(y_true_idx) else math.nan,
    }

    y_bin = label_binarize(y_true_idx, classes=np.arange(len(label_keys)))
    per_class_auc: dict[str, float] = {}
    for idx, key in enumerate(label_keys):
        if y_bin[:, idx].sum() == 0 or y_bin[:, idx].sum() == len(y_bin):
            continue
        per_class_auc[key] = float(roc_auc_score(y_bin[:, idx], eval_scores[:, idx]))
    metrics["one_vs_rest_auc"] = per_class_auc
    metrics["macro_auc"] = float(np.mean(list(per_class_auc.values()))) if per_class_auc else math.nan
    metrics["classification_report"] = classification_report(
        y_true_idx,
        y_pred,
        labels=np.arange(len(label_keys)),
        target_names=label_names,
        output_dict=True,
        zero_division=0,
    )

    pred_rows = truth_df.loc[eval_mask].reset_index(drop=True).copy()
    pred_rows["predicted_network_key"] = [label_keys[i] for i in y_pred]
    pred_rows["predicted_network_name"] = [label_names[i] for i in y_pred]
    for rank in range(top_k_for_examples):
        pred_rows[f"top_{rank + 1}_network_key"] = [label_keys[row[rank]] for row in order]
        pred_rows[f"top_{rank + 1}_network_name"] = [label_names[row[rank]] for row in order]
        pred_rows[f"top_{rank + 1}_score"] = [float(eval_scores[i, row[rank]]) for i, row in enumerate(order)]

    if out_path is not None:
        with (out_path / "network_labeling_metrics.json").open("w") as f:
            json.dump(metrics, f, indent=2)
        pred_rows.to_csv(out_path / "network_labeling_predictions.csv", index=False)
        _plot_confusion_matrix(y_true_idx, y_pred, label_names, out_path / "network_confusion_matrix.png")
        _plot_one_vs_rest_auc(y_true_idx, eval_scores, label_names, out_path / "network_one_vs_rest_auc.png")

    return metrics, pred_rows


def evaluate_network_term_ranking(
    network_embeddings: torch.Tensor,
    term_embeddings: torch.Tensor,
    truth_df: pd.DataFrame,
    term_corpus: pd.DataFrame,
    *,
    out_dir: str | Path | None = None,
    top_k_examples: int = 20,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Evaluate each network map against a deduplicated term/definition corpus."""

    out_path = Path(out_dir) if out_dir is not None else None
    if out_path is not None:
        out_path.mkdir(parents=True, exist_ok=True)

    corpus = term_corpus.copy()
    if "normalized_term" not in corpus.columns:
        corpus["normalized_term"] = corpus["term"].map(normalize_network_term)
    candidate_terms = corpus["term"].astype(str).tolist()
    norm_to_idx = {norm: i for i, norm in enumerate(corpus["normalized_term"].astype(str))}

    true_indices: list[set[int]] = []
    true_terms: list[list[str]] = []
    for terms in truth_df["true_network_terms"]:
        terms_list = terms if isinstance(terms, list) else split_network_term_cell(terms)
        positives = []
        for term in terms_list:
            idx = norm_to_idx.get(normalize_network_term(term))
            if idx is not None:
                positives.append(int(idx))
        true_indices.append(set(positives))
        true_terms.append([candidate_terms[i] for i in sorted(set(positives))])

    sim = F.normalize(network_embeddings.float(), dim=1, eps=1e-8) @ F.normalize(term_embeddings.float(), dim=1, eps=1e-8).T
    scores = sim.detach().cpu().numpy()
    metrics = multi_positive_ranking_metrics(scores, true_indices, ks=(1, 5, 10, 20, 50), ndcg_k=10)
    metrics.update(
        {
            "n_network_maps": int(len(truth_df)),
            "n_candidate_terms": int(len(candidate_terms)),
            "n_maps_with_terms": int(sum(bool(x) for x in true_indices)),
        }
    )

    order = np.argsort(-scores, axis=1)
    rows = []
    for i, truth_row in truth_df.reset_index(drop=True).iterrows():
        positives = true_indices[i]
        if positives:
            ranks_by_candidate = np.empty(len(candidate_terms), dtype=np.int64)
            ranks_by_candidate[order[i]] = np.arange(1, len(candidate_terms) + 1)
            best_rank = min(int(ranks_by_candidate[j]) for j in positives)
            avg_rank = float(np.mean([int(ranks_by_candidate[j]) for j in positives]))
        else:
            best_rank = np.nan
            avg_rank = np.nan
        top = order[i, :top_k_examples]
        rows.append(
            {
                "atlas": truth_row.get("atlas"),
                "raw_network_label": truth_row.get("raw_network_label"),
                "network_key": truth_row.get("network_key"),
                "network_name": truth_row.get("network_name"),
                "true_network_terms": "|".join(true_terms[i]),
                f"top_{top_k_examples}_predicted_terms": "|".join(candidate_terms[j] for j in top),
                "scores": "|".join(f"{scores[i, j]:.6f}" for j in top),
                "hit@5": bool(any(j in positives for j in order[i, :5])),
                "hit@10": bool(any(j in positives for j in order[i, :10])),
                "best_true_term_rank": best_rank,
                "average_true_term_rank": avg_rank,
            }
        )
    pred_df = pd.DataFrame(rows)

    if out_path is not None:
        with (out_path / "network_term_ranking_metrics.json").open("w") as f:
            json.dump(metrics, f, indent=2)
        pred_df.to_csv(out_path / "network_term_predictions.csv", index=False)
        pred_df.sort_values("best_true_term_rank", na_position="last").head(50).to_csv(
            out_path / "network_term_topk_examples.csv",
            index=False,
        )
        corpus.to_json(out_path / "network_term_candidate_corpus.json", orient="records", indent=2)

    return metrics, pred_df


def _plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, labels: Sequence[str], path: Path) -> None:
    import matplotlib.pyplot as plt
    from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

    cm = confusion_matrix(y_true, y_pred, labels=np.arange(len(labels)), normalize="true")
    fig, ax = plt.subplots(figsize=(9, 8))
    ConfusionMatrixDisplay(cm, display_labels=list(labels)).plot(ax=ax, cmap="Blues", colorbar=False, values_format=".2f")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    ax.set_title("Network Labeling Confusion Matrix")
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _plot_one_vs_rest_auc(y_true: np.ndarray, scores: np.ndarray, labels: Sequence[str], path: Path) -> None:
    import matplotlib.pyplot as plt
    from sklearn.metrics import auc, roc_curve
    from sklearn.preprocessing import label_binarize

    y_bin = label_binarize(y_true, classes=np.arange(len(labels)))
    fig, ax = plt.subplots(figsize=(7, 7))
    for idx, label in enumerate(labels):
        if y_bin[:, idx].sum() == 0 or y_bin[:, idx].sum() == len(y_bin):
            continue
        fpr, tpr, _ = roc_curve(y_bin[:, idx], scores[:, idx])
        ax.plot(fpr, tpr, label=f"{label} AUC={auc(fpr, tpr):.3f}", linewidth=1.5)
    ax.plot([0, 1], [0, 1], "--", color="black", linewidth=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Network Labeling One-vs-Rest ROC")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def load_pmid_mesh_map(path: str | Path) -> dict[str, list[str]]:
    """Load a PMID -> MeSH terms JSON map."""

    with Path(path).open() as f:
        payload = json.load(f)
    if isinstance(payload, list):
        out = {}
        for row in payload:
            pmid = str(row.get("pmid") or row.get("PMID"))
            terms = row.get("mesh_terms") or row.get("mesh") or row.get("terms") or []
            out[pmid] = normalize_mesh_terms(terms)
        return out
    if not isinstance(payload, Mapping):
        raise TypeError("MeSH JSON must be a dict mapping PMID to terms or a list of row dicts.")
    out: dict[str, list[str]] = {}
    for pmid, value in payload.items():
        if isinstance(value, Mapping):
            terms = value.get("mesh_terms") or value.get("mesh") or value.get("terms") or []
        else:
            terms = value
        out[str(pmid)] = normalize_mesh_terms(terms)
    return out


def normalize_mesh_terms(terms: Any) -> list[str]:
    if terms is None:
        return []
    if isinstance(terms, str):
        terms = [terms]
    cleaned = []
    for term in terms:
        text = str(term).strip()
        if text:
            cleaned.append(text)
    return sorted(set(cleaned))


def build_mesh_candidate_corpus(
    pmid_mesh: Mapping[str, Sequence[str]],
    pmids: Sequence[str] | None = None,
    *,
    definition_lookup: Mapping[str, str] | None = None,
    strip_qualifiers_for_candidates: bool = True,
    node_type_lookup: Mapping[str, str] | None = None,
    allowed_node_types: Sequence[str] | None = DEFAULT_MESH_RANKING_NODE_TYPES,
) -> pd.DataFrame:
    """Create the MeSH candidate corpus from true labels."""

    selected_pmids = [str(p) for p in pmids] if pmids is not None else list(pmid_mesh.keys())
    terms: set[str] = set()
    for pmid in selected_pmids:
        for term in pmid_mesh.get(str(pmid), []):
            terms.add(mesh_descriptor_name(term) if strip_qualifiers_for_candidates else str(term))
    rows = []
    allowed = set(allowed_node_types) if allowed_node_types is not None else None
    for term in sorted(t for t in terms if t):
        node_type = ""
        if node_type_lookup is not None:
            node_type = node_type_lookup.get(term, "") or node_type_lookup.get(term.lower(), "")
            if allowed is not None and node_type not in allowed:
                continue
        definition = ""
        if definition_lookup is not None:
            definition = definition_lookup.get(term, "") or definition_lookup.get(term.lower(), "")
        rows.append(
            {
                "term": term,
                "definition": definition,
                "node_type": node_type,
                "text": f"{term} [SEP] {definition}" if definition else term,
            }
        )
    return pd.DataFrame(rows)


def mesh_descriptor_name(term: str) -> str:
    return str(term).split("/")[0].strip()


def definition_lookup_from_dataframe(df: pd.DataFrame) -> dict[str, str]:
    """Build a permissive term -> definition lookup from a MeSH dataframe."""

    name_cols = [c for c in ("term", "name", "descriptor_name", "mesh_term", "label") if c in df.columns]
    def_cols = [c for c in ("definition", "description", "scope_note", "scopeNote", "summary") if c in df.columns]
    if not name_cols or not def_cols:
        return {}
    name_col, def_col = name_cols[0], def_cols[0]
    out: dict[str, str] = {}
    for _, row in df[[name_col, def_col]].dropna(subset=[name_col]).iterrows():
        name = str(row[name_col]).strip()
        definition = "" if pd.isna(row[def_col]) else str(row[def_col]).strip()
        if name:
            out[name] = definition
            out[name.lower()] = definition
    return out


def mesh_node_type_lookup_from_dataframe(df: pd.DataFrame) -> dict[str, str]:
    """Build a permissive MeSH term -> node_type lookup."""

    name_cols = [c for c in ("name", "term", "descriptor_name", "mesh_term", "label") if c in df.columns]
    type_cols = [c for c in ("node_type", "semantic_type", "category", "type") if c in df.columns]
    if not name_cols or not type_cols:
        return {}
    name_col, type_col = name_cols[0], type_cols[0]
    out: dict[str, str] = {}
    for _, row in df[[name_col, type_col]].dropna(subset=[name_col, type_col]).iterrows():
        name = str(row[name_col]).strip()
        node_type = str(row[type_col]).strip()
        if name and node_type:
            out[name] = node_type
            out[name.lower()] = node_type
    return out


@torch.no_grad()
def encode_texts_with_specter(
    texts: Sequence[str],
    specter: Any,
    proj_head_text: torch.nn.Module,
    *,
    batch_size: int = 64,
    device: str | torch.device = "cpu",
) -> torch.Tensor:
    """Encode term/definition texts and project them into shared space."""

    dev = torch.device(device)
    proj_head_text = proj_head_text.to(dev).eval()
    chunks = []
    for start in tqdm(range(0, len(texts), batch_size), desc="Encoding label texts"):
        batch_texts = list(texts[start : start + batch_size])
        emb = specter(batch_texts).to(dev)
        chunks.append(proj_head_text(emb).detach().cpu())
    return F.normalize(torch.vstack(chunks).float(), dim=1, eps=1e-8)


def multi_positive_ranking_metrics(
    scores: np.ndarray,
    true_indices: Sequence[set[int]],
    *,
    ks: Sequence[int] = (1, 5, 10, 50),
    ndcg_k: int = 10,
) -> dict[str, float]:
    """Metrics for retrieval with multiple true positives per query."""

    n_queries, n_candidates = scores.shape
    order = np.argsort(-scores, axis=1)
    valid = [i for i, positives in enumerate(true_indices) if positives]
    if not valid:
        return {f"recall@{k}": math.nan for k in ks} | {
            "map": math.nan,
            "mrr": math.nan,
            f"ndcg@{ndcg_k}": math.nan,
            "median_best_true_term_rank": math.nan,
            "average_true_term_rank": math.nan,
            "n_queries": 0,
            "n_candidates": int(n_candidates),
        }

    hits_at = {k: [] for k in ks}
    aps = []
    reciprocal_ranks = []
    ndcgs = []
    best_ranks = []
    avg_true_ranks = []

    for i in valid:
        positives = true_indices[i]
        ranks_by_candidate = np.empty(n_candidates, dtype=np.int64)
        ranks_by_candidate[order[i]] = np.arange(1, n_candidates + 1)
        pos_ranks = sorted(int(ranks_by_candidate[j]) for j in positives)
        best_ranks.append(pos_ranks[0])
        avg_true_ranks.append(float(np.mean(pos_ranks)))
        reciprocal_ranks.append(1.0 / pos_ranks[0])
        for k in ks:
            hits_at[k].append(any(r <= k for r in pos_ranks))

        num_hits = 0
        precisions = []
        for rank, cand_idx in enumerate(order[i], start=1):
            if int(cand_idx) in positives:
                num_hits += 1
                precisions.append(num_hits / rank)
                if num_hits == len(positives):
                    break
        aps.append(float(np.mean(precisions)) if precisions else 0.0)

        dcg = 0.0
        for rank, cand_idx in enumerate(order[i, :ndcg_k], start=1):
            if int(cand_idx) in positives:
                dcg += 1.0 / math.log2(rank + 1)
        ideal_hits = min(len(positives), ndcg_k)
        idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
        ndcgs.append(dcg / idcg if idcg else 0.0)

    out = {f"recall@{k}": float(np.mean(hits_at[k])) for k in ks}
    out.update(
        {
            "map": float(np.mean(aps)),
            "mrr": float(np.mean(reciprocal_ranks)),
            f"ndcg@{ndcg_k}": float(np.mean(ndcgs)),
            "median_best_true_term_rank": float(np.median(best_ranks)),
            "average_true_term_rank": float(np.mean(avg_true_ranks)),
            "n_queries": int(len(valid)),
            "n_candidates": int(n_candidates),
        }
    )
    return out


def evaluate_mesh_term_ranking(
    brain_embeddings: torch.Tensor,
    pmids: Sequence[str],
    candidate_embeddings: torch.Tensor,
    candidate_corpus: pd.DataFrame,
    pmid_mesh: Mapping[str, Sequence[str]],
    *,
    out_dir: str | Path | None = None,
    top_k_examples: int = 20,
    strip_qualifiers_for_candidates: bool = True,
) -> tuple[dict[str, float], pd.DataFrame]:
    """Evaluate brain embeddings against MeSH term candidates."""

    out_path = Path(out_dir) if out_dir is not None else None
    if out_path is not None:
        out_path.mkdir(parents=True, exist_ok=True)

    candidate_terms = candidate_corpus["term"].astype(str).tolist()
    term_to_idx = {term: i for i, term in enumerate(candidate_terms)}
    true_indices = []
    normalized_true_terms = []
    for pmid in pmids:
        terms = []
        for term in pmid_mesh.get(str(pmid), []):
            candidate_term = mesh_descriptor_name(term) if strip_qualifiers_for_candidates else str(term)
            if candidate_term in term_to_idx:
                terms.append(candidate_term)
        normalized_true_terms.append(sorted(set(terms)))
        true_indices.append({term_to_idx[t] for t in set(terms)})

    sim = F.normalize(brain_embeddings.float(), dim=1, eps=1e-8) @ F.normalize(candidate_embeddings.float(), dim=1, eps=1e-8).T
    scores = sim.detach().cpu().numpy()
    metrics = multi_positive_ranking_metrics(scores, true_indices, ks=(1, 5, 10, 50), ndcg_k=10)
    if "node_type" in candidate_corpus.columns:
        metrics["candidate_allowed_node_types"] = list(DEFAULT_MESH_RANKING_NODE_TYPES)
        metrics["candidate_excluded_node_types"] = list(EXCLUDED_MESH_RANKING_NODE_TYPES)
        metrics["candidate_node_type_counts"] = {
            str(k): int(v)
            for k, v in candidate_corpus["node_type"].fillna("").astype(str).value_counts().to_dict().items()
        }

    order = np.argsort(-scores, axis=1)
    rows = []
    for i, pmid in enumerate(pmids):
        positives = true_indices[i]
        if positives:
            ranks_by_candidate = np.empty(len(candidate_terms), dtype=np.int64)
            ranks_by_candidate[order[i]] = np.arange(1, len(candidate_terms) + 1)
            best_rank = min(int(ranks_by_candidate[j]) for j in positives)
        else:
            best_rank = np.nan
        top = order[i, :top_k_examples]
        rows.append(
            {
                "pmid": str(pmid),
                "true_mesh_terms": "|".join(normalized_true_terms[i]),
                f"top_{top_k_examples}_predicted_terms": "|".join(candidate_terms[j] for j in top),
                "scores": "|".join(f"{scores[i, j]:.6f}" for j in top),
                "hit@5": bool(any(j in positives for j in order[i, :5])),
                "hit@10": bool(any(j in positives for j in order[i, :10])),
                "best_true_term_rank": best_rank,
            }
        )
    pred_df = pd.DataFrame(rows)

    if out_path is not None:
        with (out_path / "mesh_term_ranking_metrics.json").open("w") as f:
            json.dump(metrics, f, indent=2)
        pred_df.to_csv(out_path / "mesh_term_predictions.csv", index=False)
        pred_df.sort_values("best_true_term_rank", na_position="last").head(50).to_csv(
            out_path / "mesh_term_topk_examples.csv",
            index=False,
        )
        candidate_corpus.to_json(out_path / "mesh_term_candidate_corpus.json", orient="records", indent=2)

    return metrics, pred_df


def semantic_neighbor_positive_sets(
    text_embeddings: torch.Tensor,
    *,
    n_neighbors: int = 10,
) -> list[set[int]]:
    """Define each paper's exact match plus nearest text-neighbor positives."""

    text = F.normalize(text_embeddings.float(), dim=1, eps=1e-8)
    sim = (text @ text.T).detach().cpu()
    k = min(n_neighbors + 1, sim.shape[1])
    top = torch.topk(sim, k=k, dim=1).indices.cpu().numpy()
    positives = []
    for i, row in enumerate(top):
        s = {int(j) for j in row if int(j) != i}
        s.add(i)
        positives.append(s)
    return positives


def evaluate_semantic_neighbor_retrieval(
    brain_embeddings: torch.Tensor,
    text_embeddings: torch.Tensor,
    pmids: Sequence[str],
    *,
    neighbor_text_embeddings: torch.Tensor | None = None,
    n_neighbors: int = 10,
    out_dir: str | Path | None = None,
) -> tuple[dict[str, float], pd.DataFrame]:
    """Evaluate paper retrieval where nearest text neighbors count as positives."""

    out_path = Path(out_dir) if out_dir is not None else None
    if out_path is not None:
        out_path.mkdir(parents=True, exist_ok=True)

    neighbor_basis = text_embeddings if neighbor_text_embeddings is None else neighbor_text_embeddings
    positives = semantic_neighbor_positive_sets(neighbor_basis, n_neighbors=n_neighbors)
    sim = F.normalize(brain_embeddings.float(), dim=1, eps=1e-8) @ F.normalize(text_embeddings.float(), dim=1, eps=1e-8).T
    scores = sim.detach().cpu().numpy()
    metrics = multi_positive_ranking_metrics(scores, positives, ks=(10, 50), ndcg_k=10)
    semantic_auc = _multi_positive_recall_auc(scores, positives)
    metrics = {
        "semantic_recall@10": metrics["recall@10"],
        "semantic_recall@50": metrics["recall@50"],
        "semantic_mrr": metrics["mrr"],
        "semantic_paper_style_recall_curve_auc": semantic_auc,
        "semantic_normalized_k_recall_curve_auc": semantic_auc,
        "n_text_neighbors": int(n_neighbors),
        "n_queries": metrics["n_queries"],
    }

    order = np.argsort(-scores, axis=1)
    rows = []
    pmids = [str(p) for p in pmids]
    for i, pmid in enumerate(pmids):
        top10 = order[i, :10].tolist()
        rows.append(
            {
                "pmid": pmid,
                "semantic_positive_pmids": "|".join(pmids[j] for j in sorted(positives[i])),
                "top10_predicted_pmids": "|".join(pmids[j] for j in top10),
                "hit@10": bool(any(j in positives[i] for j in top10)),
            }
        )
    examples = pd.DataFrame(rows)

    if out_path is not None:
        with (out_path / "semantic_neighbor_retrieval_metrics.json").open("w") as f:
            json.dump(metrics, f, indent=2)
        examples.to_csv(out_path / "semantic_neighbor_examples.csv", index=False)
    return metrics, examples


def _multi_positive_recall_auc(scores: np.ndarray, positives: Sequence[set[int]]) -> float:
    order = np.argsort(-scores, axis=1)
    best_ranks = []
    for i, pos in enumerate(positives):
        ranks_by_candidate = np.empty(scores.shape[1], dtype=np.int64)
        ranks_by_candidate[order[i]] = np.arange(1, scores.shape[1] + 1)
        best_ranks.append(min(int(ranks_by_candidate[j]) for j in pos))
    ranks = torch.as_tensor(best_ranks, dtype=torch.long)
    return _recall_curve_auc_from_ranks(ranks.float(), scores.shape[1])


def find_default_mesh_json() -> Path | None:
    """Find a likely PMID -> MeSH JSON map in the local experiment tree."""

    candidates = [
        Path("experiments/data/mesh_kg/mesh_annotations.json"),
        Path("docs/03_evaluation/mesh_annotations.json"),
    ]
    candidates.extend(Path(".").glob("**/*mesh*annotations*.json"))
    for candidate in candidates:
        if candidate.exists() and candidate.is_file():
            return candidate
    return None


def set_random_seed(seed: int = 0) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
