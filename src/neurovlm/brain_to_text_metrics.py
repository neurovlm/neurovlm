"""Brain-to-text evaluation metrics and notebook workflow helpers."""

from __future__ import annotations

import re
import traceback
from typing import Any, Iterable

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm.notebook import tqdm

from neurovlm.data import load_dataset, load_latent
from neurovlm.metric_utils import as_latent_batch
from neurovlm.retrieval_metrics import normalized_k_values, normalized_recall_curve_auc


def bleu(references: list[str], hypothesis: str, n: int = 4) -> float:
    """Compute BLEU score for text generation evaluation.

    Useful for evaluating text produced from brain activations against a set
    of reference descriptions.

    Parameters
    ----------
    references : list of str
        One or more reference texts to compare against.
    hypothesis : str
        The generated/predicted text.
    n : int, optional
        Maximum n-gram order (1–4). Default is 4.

    Returns
    -------
    score : float
        Sentence-level BLEU score in [0, 1].

    Notes
    -----
    Requires ``nltk`` (included in the ``metrics`` optional dependency group).
    Uses ``SmoothingFunction.method1`` to avoid zero scores on short texts.
    """
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

    ref_tokens = [ref.lower().split() for ref in references]
    hyp_tokens = hypothesis.lower().split()
    weights = tuple(1.0 / n for _ in range(n))
    smoother = SmoothingFunction().method1
    return float(sentence_bleu(ref_tokens, hyp_tokens, weights=weights,
                               smoothing_function=smoother))


def rouge(reference: str, hypothesis: str) -> dict[str, dict[str, float]]:
    """Compute ROUGE-1, ROUGE-2, and ROUGE-L scores for text generation.

    Useful for evaluating text produced from brain activations against a
    reference description.

    Parameters
    ----------
    reference : str
        The ground-truth reference text.
    hypothesis : str
        The generated/predicted text.

    Returns
    -------
    scores : dict
        Keys are ``'rouge1'``, ``'rouge2'``, ``'rougeL'``.  Each value is a
        dict with ``'precision'``, ``'recall'``, and ``'fmeasure'`` floats.

    Notes
    -----
    Requires ``rouge-score`` (included in the ``metrics`` optional dependency
    group).  Stemming is enabled for robustness to inflectional variation.
    """
    from rouge_score import rouge_scorer

    scorer = rouge_scorer.RougeScorer(
        ['rouge1', 'rouge2', 'rougeL'], use_stemmer=True
    )
    raw = scorer.score(reference, hypothesis)
    return {
        key: {
            'precision': val.precision,
            'recall': val.recall,
            'fmeasure': val.fmeasure,
        }
        for key, val in raw.items()
    }


def bertscore_single(
    bert_score_fn,
    generated: str,
    reference: str,
    model_type: str,
) -> tuple[float, float, float]:
    """Compute BERTScore precision, recall, and F1 for one generated string."""

    p, r, f1 = bert_score_fn(
        cands=[generated],
        refs=[reference],
        lang="en",
        model_type=model_type,
        verbose=False,
    )
    return float(p[0]), float(r[0]), float(f1[0])


def semantic_similarity(st_model, st_util, generated: str, reference: str) -> float:
    """Sentence-level cosine similarity for generated/reference text."""

    emb1 = st_model.encode(generated, convert_to_tensor=True)
    emb2 = st_model.encode(reference, convert_to_tensor=True)
    return float(st_util.cos_sim(emb1, emb2))


def nvlm_latent_similarity(nvlm, brain_query_emb: torch.Tensor, generated: str) -> float:
    """Cosine similarity between a brain query and generated text in NeuroVLM space."""

    nvlm._ensure_projection_heads()
    with torch.no_grad():
        raw_emb = nvlm._encode_text(generated)
        raw_emb = F.normalize(raw_emb.to(nvlm.device), dim=1, eps=1e-8)
        z_text = nvlm._proj_head_text_infonce(raw_emb)
        z_text = F.normalize(z_text, dim=-1).cpu()
    z_brain = brain_query_emb.cpu()
    if z_brain.dim() == 1:
        z_brain = z_brain.unsqueeze(0)
    return float(F.cosine_similarity(z_brain, z_text))


def token_f1(reference: str, hypothesis: str) -> float:
    """Compute token-level F1 between a reference and hypothesis string.

    Standard SQuAD-style metric: multi-set token overlap over lowercased
    whitespace-split tokens.

    Parameters
    ----------
    reference : str
        Ground-truth text.
    hypothesis : str
        Generated/predicted text.

    Returns
    -------
    f1 : float
        Token F1 in [0, 1].  Returns 0.0 when either string is empty.
    """
    from collections import Counter

    ref_tokens = reference.lower().split()
    hyp_tokens = hypothesis.lower().split()
    if not ref_tokens or not hyp_tokens:
        return 0.0
    ref_counts = Counter(ref_tokens)
    hyp_counts = Counter(hyp_tokens)
    common = sum((ref_counts & hyp_counts).values())
    if common == 0:
        return 0.0
    precision = common / len(hyp_tokens)
    recall = common / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


def format_context_summary(table: pd.DataFrame) -> str:
    lines = []
    for _, row in table.iterrows():
        lines.append(f"[{row.get('dataset', '?')}] sim={row.get('cosine_similarity', float('nan')):.3f} | {row.get('title', '')}")
    return "\n".join(lines)


def run_b2t_sample(
    *,
    nvlm,
    st_model,
    st_util,
    bert_score_fn,
    bertscore_model: str,
    llm_backend: str,
    llm_model: str,
    b2t_datasets: list[str],
    b2t_top_k: int,
    b2t_sim_threshold: float,
    name,
    latent,
    short_gt,
    long_gt,
    short_prompt,
    long_prompt="",
    short_tokens=64,
    long_tokens=512,
    datasets=None,
) -> list[dict[str, Any]]:
    try:
        result = nvlm.brain(latent).to_text(datasets=datasets or b2t_datasets)
        all_table = result.top_k(b2t_top_k)
        table = all_table[all_table["cosine_similarity"] > b2t_sim_threshold]
        if table.empty:
            table = all_table
        if len(table) > b2t_top_k:
            table = table.nlargest(b2t_top_k, "cosine_similarity").reset_index(drop=True)

        records = []
        for mode, prompt, gt, tokens in [
            ("short", short_prompt, short_gt, short_tokens),
            ("long", long_prompt, long_gt, long_tokens),
        ]:
            generated = nvlm.generate_llm_response(
                backend=llm_backend,
                model_name=llm_model,
                table=table,
                user_prompt=prompt,
                max_new_tokens=tokens,
                verbose=False,
            )
            bert_p, bert_r, bert_f1 = bertscore_single(bert_score_fn, generated, gt, bertscore_model)
            records.append(
                {
                    "generated": generated,
                    "gt_text": gt,
                    "bert_p": bert_p,
                    "bert_r": bert_r,
                    "bert_f1": bert_f1,
                    "sem_sim": semantic_similarity(st_model, st_util, generated, gt),
                    "nvlm_sim": nvlm_latent_similarity(nvlm, result.query_embeddings, generated),
                    "name": name,
                    "mode": mode,
                    "context_summary": format_context_summary(table),
                }
            )
        return records
    except Exception as exc:
        print(f"[B2T error] {name}: {type(exc).__name__}: {exc}")
        traceback.print_exc()
        return []


def make_b2t_runner(**config):
    """Return a notebook-friendly positional B2T runner bound to local config."""

    def runner(name, latent, short_gt, long_gt, short_prompt, long_prompt="", short_tokens=64, long_tokens=512, datasets=None):
        return run_b2t_sample(
            **config,
            name=name,
            latent=latent,
            short_gt=short_gt,
            long_gt=long_gt,
            short_prompt=short_prompt,
            long_prompt=long_prompt,
            short_tokens=short_tokens,
            long_tokens=long_tokens,
            datasets=datasets,
        )

    return runner


def normalize_label_text(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(text).lower()).strip()


def network_aliases(row: dict[str, Any]) -> list[str]:
    aliases = []
    for value in [row["display"], row.get("mapped_terms", ""), row.get("raw_aliases", "")]:
        aliases.extend([normalize_label_text(x) for x in str(value).split(";")])
    aliases.append(normalize_label_text(row["network_key"].replace("_", " ")))
    return [x for x in dict.fromkeys(aliases) if x]


def predict_network_label(text: str, network_info: pd.DataFrame, st_model, st_util, min_semantic_margin: float = 0.02):
    network_rows = network_info.to_dict("records")
    alias_map = {row["network_key"]: network_aliases(row) for row in network_rows}
    text_norm = normalize_label_text(text)
    alias_hits = []
    for key, aliases in alias_map.items():
        for alias in aliases:
            if alias and re.search(rf"\b{re.escape(alias)}\b", text_norm):
                alias_hits.append((key, alias))
                break
    if len(alias_hits) == 1:
        return alias_hits[0][0], "alias", alias_hits[0][1], 1.0

    label_texts = [f"{row['display']}. {row['long_definition']}" for row in network_rows]
    generated_emb = st_model.encode(text, convert_to_tensor=True)
    label_emb = st_model.encode(label_texts, convert_to_tensor=True)
    sims = st_util.cos_sim(generated_emb, label_emb).cpu().numpy().ravel()
    order = sims.argsort()[::-1]
    keys = [row["network_key"] for row in network_rows]
    margin = float(sims[order[0]] - sims[order[1]]) if len(order) > 1 else float("nan")
    method = "semantic" if margin >= min_semantic_margin else "semantic_low_margin"
    best_row = network_rows[order[0]]
    return keys[order[0]], method, best_row["display"], float(sims[order[0]])


def add_network_label_accuracy(df: pd.DataFrame, *, networks_data: dict[str, dict[str, Any]], network_info: pd.DataFrame, st_model, st_util) -> pd.DataFrame:
    out = df.copy()
    preds = out["generated"].apply(lambda text: predict_network_label(text, network_info, st_model, st_util))
    out["pred_network_key"] = [p[0] for p in preds]
    out["label_match_method"] = [p[1] for p in preds]
    out["label_match_evidence"] = [p[2] for p in preds]
    out["label_match_score"] = [p[3] for p in preds]
    out["true_network_key"] = out["name"].map(lambda name: networks_data[name]["network_key"])
    out["network_label_correct"] = out["pred_network_key"] == out["true_network_key"]
    return out


def make_network_label_accuracy_adder(**config):
    """Return a dataframe annotator bound to network labels and embedding model."""

    def adder(df: pd.DataFrame) -> pd.DataFrame:
        return add_network_label_accuracy(df, **config)

    return adder


def normalize_term_text(text: str) -> str:
    text = str(text or "").lower()
    text = text.split("/")[0]
    return re.sub(r"[^a-z0-9]+", " ", text).strip()


def split_gold_terms(value) -> list[str]:
    if pd.isna(value):
        return []
    terms = []
    for chunk in re.split(r";|\n|\|", str(value)):
        term = chunk.strip()
        if term:
            terms.append(term)
    return terms


def terms_for_dataset(dataset_name: str) -> set[str]:
    df = load_dataset(dataset_name)
    if not isinstance(df, pd.DataFrame):
        return set()
    for col in ["term", "title", "name", "label"]:
        if col in df.columns:
            return {normalize_term_text(x) for x in df[col].dropna().astype(str)}
    return set()


def network_gold_terms(sample_name: str, networks_data: dict[str, dict[str, Any]], network_labels_df: pd.DataFrame) -> list[str]:
    d = networks_data[sample_name]
    label_rows = network_labels_df[network_labels_df["raw_network_label"].astype(str) == str(d["raw_network_label"])]
    if label_rows.empty:
        label_rows = network_labels_df[network_labels_df["network_key"] == d["network_key"]]
    terms = []
    for col in ["mapped_terms", "region_terms", "cognitive_terms"]:
        if col in label_rows.columns:
            for value in label_rows[col].dropna().tolist():
                terms.extend(split_gold_terms(value))
    return list(dict.fromkeys(terms))


def table_terms(table: pd.DataFrame) -> list[str]:
    if table is None or table.empty:
        return []
    return [str(x) for x in table["title"].fillna("").tolist() if str(x).strip()]


def retrieval_table_for_sample(
    *,
    nvlm,
    cache: dict[tuple[str, str, str], pd.DataFrame],
    term_datasets_by_eval_dataset: dict[str, list[str]],
    b2t_term_top_k: int,
    b2t_evidence_top_k: int,
    b2t_sim_threshold: float,
    latent,
    dataset_name: str,
    sample: str,
) -> pd.DataFrame:
    cache_key = (dataset_name, str(sample), "evidence")
    if cache_key in cache:
        return cache[cache_key]
    datasets = term_datasets_by_eval_dataset[dataset_name]
    result = nvlm.brain(latent).to_text(datasets=datasets)
    all_table = result.top_k(max(b2t_term_top_k, b2t_evidence_top_k))
    table = all_table[all_table["cosine_similarity"] > b2t_sim_threshold]
    if table.empty:
        table = all_table
    table = table.nlargest(max(b2t_term_top_k, b2t_evidence_top_k), "cosine_similarity").reset_index(drop=True)
    cache[cache_key] = table
    return table


def full_retrieval_table_for_sample(
    *,
    nvlm,
    cache: dict[tuple[str, str, str], pd.DataFrame],
    term_datasets_by_eval_dataset: dict[str, list[str]],
    latent,
    dataset_name: str,
    sample: str | None = None,
) -> pd.DataFrame:
    cache_key = (dataset_name, str(sample), "full") if sample is not None else None
    if cache_key is not None and cache_key in cache:
        return cache[cache_key]
    datasets = term_datasets_by_eval_dataset[dataset_name]
    result = nvlm.brain(latent).to_text(datasets=datasets)
    n_candidates = sum(int(scores.shape[0]) for scores in result.scores_by_dataset.values())
    table = result.top_k(n_candidates)
    if cache_key is not None:
        cache[cache_key] = table
    return table


def unique_ranked_terms_from_table(table: pd.DataFrame) -> list[str]:
    if table is None or table.empty:
        return []
    ranked = table.sort_values("cosine_similarity", ascending=False, kind="mergesort").copy()
    ranked["_normalized_term"] = ranked["title"].map(normalize_term_text)
    ranked = ranked[ranked["_normalized_term"] != ""]
    ranked = ranked.drop_duplicates("_normalized_term", keep="first")
    return ranked["title"].astype(str).tolist()


def pubmed_abstract_lookup() -> dict[str, str]:
    df_pubs = load_dataset("pubmed_text")
    pmid_col = "pmid" if "pmid" in df_pubs.columns else df_pubs.columns[0]
    abs_col = "description" if "description" in df_pubs.columns else ("abstract" if "abstract" in df_pubs.columns else None)
    if abs_col is None:
        return {}
    return (
        df_pubs.assign(_pmid_key=lambda df: df[pmid_col].astype(str))
        .drop_duplicates("_pmid_key")
        .set_index("_pmid_key")[abs_col]
        .astype(str)
        .to_dict()
    )


def dataset_records_for_retrieval_eval(
    *,
    run_networks: bool,
    run_pubmed: bool,
    run_neurovault: bool,
    networks_data: dict[str, dict[str, Any]],
    network_labels_df: pd.DataFrame,
    pubmed_eval: list[dict[str, Any]],
    neurovault_eval: list[dict[str, Any]],
    pubmed_abs_lookup: dict[str, str],
):
    if run_networks:
        for name, d in networks_data.items():
            yield "networks", name, d["latent"], network_gold_terms(name, networks_data, network_labels_df), {"long_description": d["long_gt"]}
    if run_pubmed:
        for d in pubmed_eval:
            pmid = str(d["pmid"])
            references = {"summary": d["long_gt"]}
            abstract = pubmed_abs_lookup.get(pmid, "")
            if abstract:
                references["abstract"] = abstract
            yield "pubmed", pmid, d["latent"], [], references
    if run_neurovault:
        for d in neurovault_eval:
            yield "neurovault", str(d["doi"]), d["latent"], [], {"abstract": d["long_gt"]}


def k_from_normalized_k(normalized_k: float, n_candidates: int) -> int:
    if n_candidates <= 0 or normalized_k <= 0:
        return 0
    return min(n_candidates, max(1, int(np.ceil(float(normalized_k) * n_candidates))))


def auc_trapezoid(x, y) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if hasattr(np, "trapezoid"):
        return float(np.trapezoid(y, x))
    return float(np.trapz(y, x))


def exact_term_ranking_outputs(
    *,
    dataset: str,
    sample: str,
    gold_terms: list[str],
    retrieved_terms: list[str],
    term_eval_normalized_ks: Iterable[float],
    term_recall_curve_normalized_ks: Iterable[float],
    reachable_terms: set[str] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any] | None]:
    gold_norm_all = {normalize_term_text(t) for t in gold_terms if normalize_term_text(t)}
    if reachable_terms is None:
        gold_norm = gold_norm_all
        excluded = set()
    else:
        gold_norm = gold_norm_all & reachable_terms
        excluded = gold_norm_all - reachable_terms

    retrieved_norm = []
    seen = set()
    for term in retrieved_terms:
        norm = normalize_term_text(term)
        if norm and norm not in seen:
            retrieved_norm.append(norm)
            seen.add(norm)

    if not gold_norm or not retrieved_norm:
        return [], [], None

    n_candidates = len(retrieved_norm)
    first_hit_rank = next((i + 1 for i, term in enumerate(retrieved_norm) if term in gold_norm), np.nan)
    normalized_first_hit_rank = float(first_hit_rank / n_candidates) if not pd.isna(first_hit_rank) else np.nan

    metric_rows = []
    for normalized_k_target in term_eval_normalized_ks:
        k = k_from_normalized_k(normalized_k_target, n_candidates)
        topk = retrieved_norm[:k]
        hits = set(topk) & gold_norm
        normalized_k = k / n_candidates if n_candidates else np.nan
        metric_rows.append(
            {
                "dataset": dataset,
                "sample": sample,
                "normalized_k_target": float(normalized_k_target),
                "normalized_k": float(normalized_k),
                "k": int(k),
                "n_candidate_terms": int(n_candidates),
                "n_gold_terms": len(gold_norm),
                "n_unreachable_gold_terms": len(excluded),
                "n_retrieved_terms": len(topk),
                "n_hits": len(hits),
                "precision_at_normalized_k": len(hits) / max(len(topk), 1),
                "recall_at_normalized_k": len(hits) / len(gold_norm),
                "hit_at_normalized_k": bool(hits),
                "mrr_at_normalized_k": 0.0 if pd.isna(first_hit_rank) or first_hit_rank > k else 1.0 / float(first_hit_rank),
                "normalized_first_hit_rank": normalized_first_hit_rank,
                "expected_random_recall_at_normalized_k": float(normalized_k_target),
                "matched_terms": "; ".join(sorted(hits)),
            }
        )

    curve_rows = []
    recall_values = []
    normalized_ks = list(term_recall_curve_normalized_ks)
    for normalized_k_target in normalized_ks:
        k = k_from_normalized_k(normalized_k_target, n_candidates)
        topk = retrieved_norm[:k]
        hits = set(topk) & gold_norm
        recall = len(hits) / len(gold_norm)
        recall_values.append(recall)
        curve_rows.append(
            {
                "dataset": dataset,
                "sample": sample,
                "normalized_k_target": float(normalized_k_target),
                "normalized_k": float(k / n_candidates) if n_candidates else np.nan,
                "k": int(k),
                "n_candidate_terms": int(n_candidates),
                "n_gold_terms": len(gold_norm),
                "recall_at_normalized_k": float(recall),
                "expected_random_recall_at_normalized_k": float(normalized_k_target),
            }
        )

    auc = auc_trapezoid(normalized_ks, recall_values)
    return metric_rows, curve_rows, {
        "dataset": dataset,
        "sample": sample,
        "n_candidate_terms": int(n_candidates),
        "n_gold_terms": len(gold_norm),
        "n_unreachable_gold_terms": len(excluded),
        "recall_auc": float(auc),
        "expected_random_recall_auc": 0.5,
        "recall_auc_minus_random": float(auc - 0.5),
        "normalized_first_hit_rank": normalized_first_hit_rank,
    }


def run_network_gold_term_ranking(
    *,
    nvlm,
    networks_data: dict[str, dict[str, Any]],
    network_labels_df: pd.DataFrame,
    pubmed_eval: list[dict[str, Any]],
    neurovault_eval: list[dict[str, Any]],
    pubmed_abs_lookup: dict[str, str],
    network_candidate_terms: set[str],
    term_datasets_by_eval_dataset: dict[str, list[str]],
    retrieval_table_cache: dict[tuple[str, str, str], pd.DataFrame],
    term_eval_normalized_ks: Iterable[float],
    term_recall_curve_normalized_ks: Iterable[float],
    b2t_term_example_top_k: int,
    output_dir,
    run_networks: bool = True,
    run_pubmed: bool = True,
    run_neurovault: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    term_metric_rows = []
    term_curve_rows = []
    term_auc_rows = []
    term_examples = []
    retrieval_records = list(
        dataset_records_for_retrieval_eval(
            run_networks=run_networks,
            run_pubmed=run_pubmed,
            run_neurovault=run_neurovault,
            networks_data=networks_data,
            network_labels_df=network_labels_df,
            pubmed_eval=pubmed_eval,
            neurovault_eval=neurovault_eval,
            pubmed_abs_lookup=pubmed_abs_lookup,
        )
    )

    for dataset, sample, latent, gold_terms, _references in tqdm(retrieval_records, desc="Network normalized gold-term ranking"):
        if dataset != "networks":
            continue
        table = full_retrieval_table_for_sample(
            nvlm=nvlm,
            cache=retrieval_table_cache,
            term_datasets_by_eval_dataset=term_datasets_by_eval_dataset,
            latent=latent,
            dataset_name=dataset,
            sample=sample,
        )
        retrieved_terms = unique_ranked_terms_from_table(table)
        ranked_reachable_terms = {normalize_term_text(term) for term in retrieved_terms if normalize_term_text(term)}
        metric_rows, curve_rows, auc_row = exact_term_ranking_outputs(
            dataset=dataset,
            sample=sample,
            gold_terms=gold_terms,
            retrieved_terms=retrieved_terms,
            term_eval_normalized_ks=term_eval_normalized_ks,
            term_recall_curve_normalized_ks=term_recall_curve_normalized_ks,
            reachable_terms=ranked_reachable_terms,
        )
        term_metric_rows.extend(metric_rows)
        term_curve_rows.extend(curve_rows)
        if auc_row is not None:
            term_auc_rows.append(auc_row)
        term_examples.append(
            {
                "dataset": dataset,
                "sample": sample,
                "gold_terms": "; ".join(gold_terms[:50]),
                "n_ranked_candidate_terms": len(retrieved_terms),
                "top_terms": "; ".join(retrieved_terms[:b2t_term_example_top_k]),
            }
        )

    metrics_df = pd.DataFrame(term_metric_rows)
    curve_df = pd.DataFrame(term_curve_rows)
    auc_df = pd.DataFrame(term_auc_rows)
    examples_df = pd.DataFrame(term_examples)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_df.to_csv(output_dir / "b2t_approach1_gold_term_ranking_metrics.csv", index=False)
    curve_df.to_csv(output_dir / "b2t_approach1_gold_term_recall_curve.csv", index=False)
    auc_df.to_csv(output_dir / "b2t_approach1_gold_term_recall_auc.csv", index=False)
    examples_df.to_csv(output_dir / "b2t_approach1_gold_term_examples.csv", index=False)
    return metrics_df, curve_df, auc_df, examples_df


def project_text_latents_to_shared(nvlm, text_latents, batch_size: int = 4096) -> torch.Tensor:
    nvlm._ensure_projection_heads()
    x = as_latent_batch(text_latents).float()
    chunks = []
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            text_batch = F.normalize(x[start : start + batch_size].to(nvlm.device), dim=1, eps=1e-8)
            z = nvlm._proj_head_text_infonce(text_batch)
            chunks.append(F.normalize(z.float(), dim=1, eps=1e-8).detach().cpu())
    return torch.cat(chunks, dim=0)


def project_brain_latents_to_shared(nvlm, brain_latents, batch_size: int = 4096) -> torch.Tensor:
    nvlm._ensure_projection_heads()
    batch = as_latent_batch(brain_latents).float()
    chunks = []
    image_head = nvlm._proj_head_image_infonce
    with torch.no_grad():
        for start in range(0, len(batch), batch_size):
            z = image_head(batch[start : start + batch_size].to(nvlm.device))
            chunks.append(F.normalize(z.float(), dim=1, eps=1e-8).detach().cpu())
    return torch.cat(chunks, dim=0)


def mesh_descriptor_name(term: str) -> str:
    return str(term).split("/")[0].strip()


def load_pubmed_mesh_gold_annotations_or_none():
    try:
        annotations = load_dataset("pubmed_mesh_annotations")
        print(f"Loaded PubMed MeSH annotations for {len(annotations):,} PMIDs")
        return annotations
    except Exception as exc:
        print("PubMed MeSH gold annotations are unavailable; skipping MeSH term-ranking diagnostics.")
        print(f"Loader error: {type(exc).__name__}: {exc}")
        return None


def all_target_recall_curve(scores: np.ndarray, true_indices: list[set[int]]) -> np.ndarray:
    """Average per-query fraction of target terms recovered by each rank."""

    n_queries, n_candidates = scores.shape
    order = np.argsort(-scores, axis=1)
    weighted_hits_by_rank = np.zeros(n_candidates, dtype=np.float64)
    for i, positives in enumerate(true_indices):
        ranks_by_candidate = np.empty(n_candidates, dtype=np.int64)
        ranks_by_candidate[order[i]] = np.arange(1, n_candidates + 1)
        weight = 1.0 / float(len(positives))
        for j in positives:
            weighted_hits_by_rank[int(ranks_by_candidate[j]) - 1] += weight
    return np.cumsum(weighted_hits_by_rank) / float(n_queries)


def all_target_ranking_metrics(
    scores: np.ndarray,
    true_indices: list[set[int]],
    *,
    ks: Iterable[int] = (1, 5, 10, 50),
) -> dict[str, float]:
    """Metrics where each query receives fractional credit for every target."""

    n_queries, n_candidates = scores.shape
    order = np.argsort(-scores, axis=1)
    target_ranks = []
    per_query_mean_ranks = []
    per_query_worst_ranks = []
    recall_at = {int(k): [] for k in ks}

    for i, positives in enumerate(true_indices):
        ranks_by_candidate = np.empty(n_candidates, dtype=np.int64)
        ranks_by_candidate[order[i]] = np.arange(1, n_candidates + 1)
        ranks = sorted(int(ranks_by_candidate[j]) for j in positives)
        target_ranks.extend(ranks)
        per_query_mean_ranks.append(float(np.mean(ranks)))
        per_query_worst_ranks.append(float(max(ranks)))
        for k in recall_at:
            recall_at[k].append(float(sum(r <= k for r in ranks)) / float(len(ranks)))

    curve = all_target_recall_curve(scores, true_indices)
    out = {
        "paper_recall_curve_auc": normalized_recall_curve_auc(torch.as_tensor(curve)),
        "normalized_k_recall_curve_auc": normalized_recall_curve_auc(torch.as_tensor(curve)),
        "median_target_term_rank": float(np.median(target_ranks)),
        "mean_target_term_rank": float(np.mean(target_ranks)),
        "median_query_mean_target_term_rank": float(np.median(per_query_mean_ranks)),
        "median_query_worst_target_term_rank": float(np.median(per_query_worst_ranks)),
        "n_queries": int(n_queries),
        "n_candidates": int(n_candidates),
        "n_target_terms": int(len(target_ranks)),
        "mean_target_terms_per_query": float(len(target_ranks) / max(n_queries, 1)),
    }
    for k, values in recall_at.items():
        out[f"recall@{k}"] = float(np.mean(values))
    return out


def run_pubmed_mesh_gold_ranking(
    *,
    nvlm,
    pubmed_eval: list[dict[str, Any]],
    pubmed_b2t_dataset: str,
    mesh_brain_rankable_node_types: Iterable[str],
    b2t_term_example_top_k: int,
    output_dir,
    run_pubmed: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    annotations = load_pubmed_mesh_gold_annotations_or_none()
    node_type_by_term: dict[str, str] = {}
    if annotations is not None:
        try:
            mesh_nodes_for_gold = load_dataset("pubmed_mesh_nodes")
            if "node_type" in mesh_nodes_for_gold.columns:
                name_col = "name" if "name" in mesh_nodes_for_gold.columns else "term"
                node_type_by_term = {
                    normalize_term_text(row[name_col]): row["node_type"]
                    for _, row in mesh_nodes_for_gold.iterrows()
                    if pd.notna(row.get(name_col)) and pd.notna(row.get("node_type"))
                }
        except Exception as exc:
            print(f"Could not load pubmed_mesh_nodes for node-type filtering: {type(exc).__name__}: {exc}")

    allowed_mesh_types = set(mesh_brain_rankable_node_types)
    mesh_candidate_df = load_dataset(pubmed_b2t_dataset).copy()
    mesh_candidate_latents, _mesh_candidate_terms = load_latent(pubmed_b2t_dataset)

    if len(mesh_candidate_df) != len(mesh_candidate_latents):
        raise ValueError(
            f"MeSH candidate metadata/latent length mismatch: metadata={len(mesh_candidate_df)} latents={len(mesh_candidate_latents)}"
        )

    mesh_term_col = next((col for col in ["term", "title", "name", "label"] if col in mesh_candidate_df.columns), None)
    if mesh_term_col is None:
        raise KeyError(f"{pubmed_b2t_dataset} must contain one of term/title/name/label columns.")

    mesh_candidate_df["term"] = mesh_candidate_df[mesh_term_col].astype(str).map(mesh_descriptor_name)
    mesh_candidate_df["normalized_term"] = mesh_candidate_df["term"].map(normalize_term_text)
    mesh_candidate_df["node_type"] = mesh_candidate_df["normalized_term"].map(node_type_by_term).fillna("")
    keep_mask = mesh_candidate_df["node_type"].isin(allowed_mesh_types).to_numpy() if node_type_by_term else np.ones(len(mesh_candidate_df), dtype=bool)
    mesh_candidate_df = mesh_candidate_df.loc[keep_mask].copy()
    mesh_candidate_latents = mesh_candidate_latents[keep_mask]
    unique_mask = ~mesh_candidate_df["normalized_term"].duplicated(keep="first").to_numpy()
    mesh_candidate_df = mesh_candidate_df.loc[unique_mask].reset_index(drop=True)
    mesh_candidate_latents = mesh_candidate_latents[unique_mask]

    mesh_candidate_embeddings = project_text_latents_to_shared(nvlm, mesh_candidate_latents)
    mesh_candidate_terms = mesh_candidate_df["term"].astype(str).tolist()
    mesh_norm_to_idx = {norm: i for i, norm in enumerate(mesh_candidate_df["normalized_term"].astype(str))}
    mesh_candidate_norms = set(mesh_norm_to_idx)

    def pubmed_mesh_gold_terms(pmid) -> list[str]:
        if annotations is None:
            return []
        out = []
        for term in annotations.get(str(pmid), []):
            base = mesh_descriptor_name(term)
            norm = normalize_term_text(base)
            if not norm or norm not in mesh_candidate_norms:
                continue
            if node_type_by_term and node_type_by_term.get(norm) not in allowed_mesh_types:
                continue
            out.append(base)
        return list(dict.fromkeys(out))

    mesh_records = []
    mesh_true_indices = []
    mesh_true_terms = []
    if run_pubmed and annotations is not None:
        for d in pubmed_eval:
            positives = []
            terms = []
            for term in pubmed_mesh_gold_terms(str(d["pmid"])):
                idx = mesh_norm_to_idx.get(normalize_term_text(term))
                if idx is not None:
                    positives.append(idx)
                    terms.append(mesh_candidate_terms[idx])
            if positives:
                mesh_records.append(d)
                mesh_true_indices.append(set(positives))
                mesh_true_terms.append(sorted(set(terms)))

    if mesh_records:
        mesh_brain_embeddings = project_brain_latents_to_shared(nvlm, [d["latent"] for d in mesh_records])
        mesh_scores = (mesh_brain_embeddings @ mesh_candidate_embeddings.T).detach().cpu().numpy()
        mesh_metrics = all_target_ranking_metrics(mesh_scores, mesh_true_indices, ks=(1, 5, 10, 50))
        mesh_order = np.argsort(-mesh_scores, axis=1)
        metrics_df = pd.DataFrame(
            [
                {
                    "dataset": "pubmed_mesh",
                    "mesh_recall_definition": "mean_fraction_of_all_target_terms_recovered",
                    "n_queries": mesh_metrics["n_queries"],
                    "n_candidates": mesh_metrics["n_candidates"],
                    "n_target_terms": mesh_metrics["n_target_terms"],
                    "mean_target_terms_per_query": mesh_metrics["mean_target_terms_per_query"],
                    "mesh_paper_recall_curve_auc": mesh_metrics["paper_recall_curve_auc"],
                    "mesh_normalized_k_recall_curve_auc": mesh_metrics["normalized_k_recall_curve_auc"],
                    "mesh_recall@1": mesh_metrics["recall@1"],
                    "mesh_recall@5": mesh_metrics["recall@5"],
                    "mesh_recall@10": mesh_metrics["recall@10"],
                    "mesh_recall@50": mesh_metrics["recall@50"],
                    "mesh_median_target_term_rank": mesh_metrics["median_target_term_rank"],
                    "mesh_mean_target_term_rank": mesh_metrics["mean_target_term_rank"],
                    "mesh_median_query_mean_target_term_rank": mesh_metrics["median_query_mean_target_term_rank"],
                    "mesh_median_query_worst_target_term_rank": mesh_metrics["median_query_worst_target_term_rank"],
                    "allowed_node_types": ";".join(mesh_brain_rankable_node_types),
                }
            ]
        )

        recall_curve = all_target_recall_curve(mesh_scores, mesh_true_indices)
        norm_k = normalized_k_values(len(mesh_candidate_terms)).cpu().numpy()
        curve_df = pd.DataFrame(
            {
                "dataset": "pubmed_mesh",
                "mesh_recall_definition": "mean_fraction_of_all_target_terms_recovered",
                "k": np.arange(1, len(mesh_candidate_terms) + 1),
                "normalized_k": norm_k,
                "recall_at_normalized_k": recall_curve,
                "expected_random_recall_at_normalized_k": norm_k,
            }
        )

        example_rows = []
        for i, d in enumerate(mesh_records):
            positives = mesh_true_indices[i]
            ranks_by_candidate = np.empty(len(mesh_candidate_terms), dtype=np.int64)
            ranks_by_candidate[mesh_order[i]] = np.arange(1, len(mesh_candidate_terms) + 1)
            target_ranks = sorted(int(ranks_by_candidate[j]) for j in positives)
            top = mesh_order[i, :b2t_term_example_top_k]
            example_rows.append(
                {
                    "dataset": "pubmed_mesh",
                    "sample": str(d["pmid"]),
                    "true_mesh_terms": "; ".join(mesh_true_terms[i]),
                    "target_term_ranks": "; ".join(str(rank) for rank in target_ranks),
                    "mean_target_term_rank": float(np.mean(target_ranks)),
                    "worst_target_term_rank": int(max(target_ranks)),
                    "top_terms": "; ".join(mesh_candidate_terms[j] for j in top),
                    "top_scores": "; ".join(f"{mesh_scores[i, j]:.6f}" for j in top),
                    "target_recall@5": sum(rank <= 5 for rank in target_ranks) / float(len(target_ranks)),
                    "target_recall@10": sum(rank <= 10 for rank in target_ranks) / float(len(target_ranks)),
                }
            )
        examples_df = pd.DataFrame(example_rows)
    else:
        metrics_df = pd.DataFrame()
        curve_df = pd.DataFrame()
        examples_df = pd.DataFrame()

    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_df.to_csv(output_dir / "b2t_pubmed_mesh_gold_term_ranking_metrics.csv", index=False)
    metrics_df.to_json(output_dir / "b2t_pubmed_mesh_gold_term_ranking_metrics.json", orient="records", indent=2)
    curve_df.to_csv(output_dir / "b2t_pubmed_mesh_gold_term_recall_curve.csv", index=False)
    curve_df.to_json(output_dir / "b2t_pubmed_mesh_gold_term_recall_curve.json", orient="records", indent=2)
    examples_df.to_csv(output_dir / "b2t_pubmed_mesh_gold_term_examples.csv", index=False)
    examples_df.to_json(output_dir / "b2t_pubmed_mesh_gold_term_examples.json", orient="records", indent=2)
    return metrics_df, curve_df, examples_df


def run_pubmed_mesh_node_type_rankings(
    *,
    nvlm,
    pubmed_eval: list[dict[str, Any]],
    pubmed_b2t_dataset: str,
    mesh_node_types: Iterable[str],
    b2t_term_example_top_k: int,
    output_dir,
    run_pubmed: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Evaluate PubMed MeSH target-term recall separately by MeSH node type."""

    annotations = load_pubmed_mesh_gold_annotations_or_none()
    node_type_by_term: dict[str, str] = {}
    if annotations is not None:
        try:
            mesh_nodes_for_gold = load_dataset("pubmed_mesh_nodes")
            if "node_type" in mesh_nodes_for_gold.columns:
                name_col = "name" if "name" in mesh_nodes_for_gold.columns else "term"
                node_type_by_term = {
                    normalize_term_text(row[name_col]): row["node_type"]
                    for _, row in mesh_nodes_for_gold.iterrows()
                    if pd.notna(row.get(name_col)) and pd.notna(row.get("node_type"))
                }
        except Exception as exc:
            print(f"Could not load pubmed_mesh_nodes for node-type filtering: {type(exc).__name__}: {exc}")

    if annotations is None or not node_type_by_term:
        print("PubMed MeSH node-type rankings require MeSH annotations and node types; skipping.")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    requested_node_types = list(dict.fromkeys(str(t) for t in mesh_node_types))
    requested_type_set = set(requested_node_types)
    mesh_candidate_df = load_dataset(pubmed_b2t_dataset).copy()
    mesh_candidate_latents, _mesh_candidate_terms = load_latent(pubmed_b2t_dataset)

    if len(mesh_candidate_df) != len(mesh_candidate_latents):
        raise ValueError(
            f"MeSH candidate metadata/latent length mismatch: metadata={len(mesh_candidate_df)} latents={len(mesh_candidate_latents)}"
        )

    mesh_term_col = next((col for col in ["term", "title", "name", "label"] if col in mesh_candidate_df.columns), None)
    if mesh_term_col is None:
        raise KeyError(f"{pubmed_b2t_dataset} must contain one of term/title/name/label columns.")

    mesh_candidate_df["term"] = mesh_candidate_df[mesh_term_col].astype(str).map(mesh_descriptor_name)
    mesh_candidate_df["normalized_term"] = mesh_candidate_df["term"].map(normalize_term_text)
    mesh_candidate_df["node_type"] = mesh_candidate_df["normalized_term"].map(node_type_by_term).fillna("")
    keep_mask = mesh_candidate_df["node_type"].isin(requested_type_set).to_numpy()
    combined_candidate_df = mesh_candidate_df.loc[keep_mask].copy()
    combined_candidate_latents = mesh_candidate_latents[keep_mask]
    unique_mask = ~combined_candidate_df["normalized_term"].duplicated(keep="first").to_numpy()
    combined_candidate_df = combined_candidate_df.loc[unique_mask].reset_index(drop=True)
    combined_candidate_latents = combined_candidate_latents[unique_mask]
    combined_candidate_terms = len(combined_candidate_df)

    metric_rows = []
    curve_frames = []
    example_frames = []
    for node_type in requested_node_types:
        type_mask = combined_candidate_df["node_type"].eq(node_type).to_numpy()
        type_candidate_df = combined_candidate_df.loc[type_mask].reset_index(drop=True)
        type_candidate_latents = combined_candidate_latents[type_mask]
        if len(type_candidate_df) == 0:
            metric_rows.append(
                {
                    "dataset": f"pubmed_mesh_{node_type}",
                    "node_type": node_type,
                    "n_queries": 0,
                    "n_candidates": 0,
                    "combined_candidate_terms": combined_candidate_terms,
                    "n_target_terms": 0,
                    "mean_target_terms_per_query": np.nan,
                    "mesh_normalized_k_recall_curve_auc": np.nan,
                }
            )
            continue

        candidate_embeddings = project_text_latents_to_shared(nvlm, type_candidate_latents)
        candidate_terms = type_candidate_df["term"].astype(str).tolist()
        norm_to_idx = {norm: i for i, norm in enumerate(type_candidate_df["normalized_term"].astype(str))}
        candidate_norms = set(norm_to_idx)

        mesh_records = []
        true_indices = []
        true_terms = []
        if run_pubmed:
            for d in pubmed_eval:
                positives = []
                terms = []
                for term in annotations.get(str(d["pmid"]), []):
                    base = mesh_descriptor_name(term)
                    norm = normalize_term_text(base)
                    if not norm or norm not in candidate_norms:
                        continue
                    if node_type_by_term.get(norm) != node_type:
                        continue
                    idx = norm_to_idx.get(norm)
                    if idx is not None:
                        positives.append(idx)
                        terms.append(candidate_terms[idx])
                if positives:
                    mesh_records.append(d)
                    true_indices.append(set(positives))
                    true_terms.append(sorted(set(terms)))

        if not mesh_records:
            metric_rows.append(
                {
                    "dataset": f"pubmed_mesh_{node_type}",
                    "node_type": node_type,
                    "n_queries": 0,
                    "n_candidates": len(candidate_terms),
                    "combined_candidate_terms": combined_candidate_terms,
                    "n_target_terms": 0,
                    "mean_target_terms_per_query": np.nan,
                    "mesh_normalized_k_recall_curve_auc": np.nan,
                }
            )
            continue

        brain_embeddings = project_brain_latents_to_shared(nvlm, [d["latent"] for d in mesh_records])
        scores = (brain_embeddings @ candidate_embeddings.T).detach().cpu().numpy()
        metrics = all_target_ranking_metrics(scores, true_indices, ks=(1, 5, 10, 50))
        order = np.argsort(-scores, axis=1)
        metric_rows.append(
            {
                "dataset": f"pubmed_mesh_{node_type}",
                "node_type": node_type,
                "mesh_recall_definition": "mean_fraction_of_all_target_terms_recovered",
                "n_queries": metrics["n_queries"],
                "n_candidates": metrics["n_candidates"],
                "combined_candidate_terms": combined_candidate_terms,
                "n_target_terms": metrics["n_target_terms"],
                "mean_target_terms_per_query": metrics["mean_target_terms_per_query"],
                "mesh_paper_recall_curve_auc": metrics["paper_recall_curve_auc"],
                "mesh_normalized_k_recall_curve_auc": metrics["normalized_k_recall_curve_auc"],
                "mesh_recall@1": metrics["recall@1"],
                "mesh_recall@5": metrics["recall@5"],
                "mesh_recall@10": metrics["recall@10"],
                "mesh_recall@50": metrics["recall@50"],
                "mesh_median_target_term_rank": metrics["median_target_term_rank"],
                "mesh_mean_target_term_rank": metrics["mean_target_term_rank"],
                "mesh_median_query_mean_target_term_rank": metrics["median_query_mean_target_term_rank"],
                "mesh_median_query_worst_target_term_rank": metrics["median_query_worst_target_term_rank"],
            }
        )

        recall_curve = all_target_recall_curve(scores, true_indices)
        norm_k = normalized_k_values(len(candidate_terms)).cpu().numpy()
        curve_frames.append(
            pd.DataFrame(
                {
                    "dataset": f"pubmed_mesh_{node_type}",
                    "node_type": node_type,
                    "mesh_recall_definition": "mean_fraction_of_all_target_terms_recovered",
                    "k": np.arange(1, len(candidate_terms) + 1),
                    "normalized_k": norm_k,
                    "recall_at_normalized_k": recall_curve,
                    "expected_random_recall_at_normalized_k": norm_k,
                }
            )
        )

        example_rows = []
        for i, d in enumerate(mesh_records):
            positives = true_indices[i]
            ranks_by_candidate = np.empty(len(candidate_terms), dtype=np.int64)
            ranks_by_candidate[order[i]] = np.arange(1, len(candidate_terms) + 1)
            target_ranks = sorted(int(ranks_by_candidate[j]) for j in positives)
            top = order[i, :b2t_term_example_top_k]
            example_rows.append(
                {
                    "dataset": f"pubmed_mesh_{node_type}",
                    "node_type": node_type,
                    "sample": str(d["pmid"]),
                    "true_mesh_terms": "; ".join(true_terms[i]),
                    "target_term_ranks": "; ".join(str(rank) for rank in target_ranks),
                    "mean_target_term_rank": float(np.mean(target_ranks)),
                    "worst_target_term_rank": int(max(target_ranks)),
                    "top_terms": "; ".join(candidate_terms[j] for j in top),
                    "top_scores": "; ".join(f"{scores[i, j]:.6f}" for j in top),
                    "target_recall@5": sum(rank <= 5 for rank in target_ranks) / float(len(target_ranks)),
                    "target_recall@10": sum(rank <= 10 for rank in target_ranks) / float(len(target_ranks)),
                }
            )
        example_frames.append(pd.DataFrame(example_rows))

    metrics_df = pd.DataFrame(metric_rows)
    curves_df = pd.concat(curve_frames, ignore_index=True) if curve_frames else pd.DataFrame()
    examples_df = pd.concat(example_frames, ignore_index=True) if example_frames else pd.DataFrame()
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_df.to_csv(output_dir / "b2t_pubmed_mesh_node_type_ranking_metrics.csv", index=False)
    metrics_df.to_json(output_dir / "b2t_pubmed_mesh_node_type_ranking_metrics.json", orient="records", indent=2)
    curves_df.to_csv(output_dir / "b2t_pubmed_mesh_node_type_recall_curves.csv", index=False)
    curves_df.to_json(output_dir / "b2t_pubmed_mesh_node_type_recall_curves.json", orient="records", indent=2)
    examples_df.to_csv(output_dir / "b2t_pubmed_mesh_node_type_examples.csv", index=False)
    examples_df.to_json(output_dir / "b2t_pubmed_mesh_node_type_examples.json", orient="records", indent=2)
    return metrics_df, curves_df, examples_df


def brain_latents_for_generated_group(
    df: pd.DataFrame,
    *,
    networks_data: dict[str, dict[str, Any]],
    pubmed_eval: list[dict[str, Any]],
    neurovault_eval: list[dict[str, Any]],
) -> list[Any]:
    pubmed_by_pmid = {str(d["pmid"]): d["latent"] for d in pubmed_eval}
    neurovault_by_doi = {str(d["doi"]): d["latent"] for d in neurovault_eval}
    brain_embs = []
    for _, row in df.iterrows():
        source = row["dataset"]
        name = str(row["name"])
        if source == "networks":
            brain_embs.append(networks_data[name]["latent"])
        elif source == "pubmed":
            brain_embs.append(pubmed_by_pmid[name])
        elif source == "neurovault":
            brain_embs.append(neurovault_by_doi[name])
    return brain_embs


def generated_text_retrieval_curve(
    nvlm,
    df: pd.DataFrame,
    *,
    networks_data: dict[str, dict[str, Any]],
    pubmed_eval: list[dict[str, Any]],
    neurovault_eval: list[dict[str, Any]],
) -> tuple[float, pd.DataFrame]:
    if len(df) < 2:
        return np.nan, pd.DataFrame()
    generated = df["generated"].astype(str).tolist()
    z_text = project_text_latents_to_shared(nvlm, nvlm._encode_text(generated))
    z_brain = project_brain_latents_to_shared(
        nvlm,
        brain_latents_for_generated_group(df, networks_data=networks_data, pubmed_eval=pubmed_eval, neurovault_eval=neurovault_eval),
    )
    scores = z_text @ z_brain.T
    order = scores.argsort(dim=1, descending=True)
    target = torch.arange(len(df)).view(-1, 1)
    first_hits = order.eq(target).int().argmax(dim=1)
    hit_counts = torch.bincount(first_hits.cpu(), minlength=len(df)).float()
    recall_curve = torch.cumsum(hit_counts, dim=0) / float(len(df))
    normalized_k = normalized_k_values(len(df)).cpu().numpy()
    auc = normalized_recall_curve_auc(recall_curve)
    curve_df = pd.DataFrame(
        {
            "k": np.arange(1, len(df) + 1),
            "normalized_k": normalized_k,
            "recall_at_normalized_k": recall_curve.cpu().numpy(),
            "expected_random_recall_at_normalized_k": normalized_k,
        }
    )
    return auc, curve_df


def generated_text_metric_summary(
    *,
    nvlm,
    b2t_all: pd.DataFrame,
    networks_data: dict[str, dict[str, Any]],
    pubmed_eval: list[dict[str, Any]],
    neurovault_eval: list[dict[str, Any]],
    output_dir,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    generated_text_recall_rows = []
    generated_text_curve_rows = []
    for (dataset, mode), sub in b2t_all.groupby(["dataset", "mode"]):
        sub = sub.reset_index(drop=True)
        auc, curve_df = generated_text_retrieval_curve(
            nvlm,
            sub,
            networks_data=networks_data,
            pubmed_eval=pubmed_eval,
            neurovault_eval=neurovault_eval,
        )
        generated_text_recall_rows.append(
            {
                "dataset": dataset,
                "mode": mode,
                "generated_text_normalized_k_recall_curve_auc": auc,
                "n": len(sub),
            }
        )
        if len(curve_df):
            curve_df.insert(0, "mode", mode)
            curve_df.insert(0, "dataset", dataset)
            generated_text_curve_rows.append(curve_df)

    recall_auc_df = pd.DataFrame(generated_text_recall_rows)
    recall_curve_df = pd.concat(generated_text_curve_rows, ignore_index=True) if generated_text_curve_rows else pd.DataFrame()
    summary = b2t_all.groupby(["dataset", "mode"])[["nvlm_sim", "bert_f1", "sem_sim"]].agg(["mean", "std", "count"]).round(3)
    label_summary = pd.DataFrame()
    output_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_dir / "b2t_generated_text_metric_summary.csv")
    summary.reset_index().to_json(output_dir / "b2t_generated_text_metric_summary.json", orient="records", indent=2)
    recall_auc_df.round(3).to_csv(output_dir / "b2t_generated_text_recall_auc.csv", index=False)
    recall_auc_df.round(3).to_json(output_dir / "b2t_generated_text_recall_auc.json", orient="records", indent=2)
    if len(recall_curve_df):
        recall_curve_df.to_csv(output_dir / "b2t_generated_text_recall_curve.csv", index=False)
        recall_curve_df.to_json(output_dir / "b2t_generated_text_recall_curve.json", orient="records", indent=2)
    if "network_label_correct" in b2t_all.columns:
        label_summary = b2t_all[b2t_all["dataset"] == "networks"].groupby("mode")["network_label_correct"].agg(["mean", "sum", "count"]).round(3)
        label_summary.to_csv(output_dir / "b2t_generated_text_network_label_accuracy_summary.csv")
        label_summary.reset_index().to_json(output_dir / "b2t_generated_text_network_label_accuracy_summary.json", orient="records", indent=2)
    return summary, recall_auc_df, recall_curve_df, label_summary


def generated_text_pair_baseline(
    *,
    nvlm,
    b2t_all: pd.DataFrame,
    networks_data: dict[str, dict[str, Any]],
    pubmed_eval: list[dict[str, Any]],
    neurovault_eval: list[dict[str, Any]],
) -> pd.DataFrame:
    rows = []
    for (dataset, mode), sub in b2t_all.groupby(["dataset", "mode"]):
        if len(sub) < 2:
            continue
        sub = sub.reset_index(drop=True)
        z_text = project_text_latents_to_shared(nvlm, nvlm._encode_text(sub["generated"].astype(str).tolist()))
        z_brain = project_brain_latents_to_shared(
            nvlm,
            brain_latents_for_generated_group(sub, networks_data=networks_data, pubmed_eval=pubmed_eval, neurovault_eval=neurovault_eval),
        )
        scores = z_text @ z_brain.T
        eye = torch.eye(len(sub), dtype=torch.bool)
        for val in scores[eye].numpy():
            rows.append({"dataset": dataset, "mode": mode, "pair": "matched", "score": float(val)})
        for val in scores[~eye].numpy():
            rows.append({"dataset": dataset, "mode": mode, "pair": "random/off-diagonal", "score": float(val)})
    return pd.DataFrame(rows)


def paper_record_text(record: dict) -> str:
    title = str(record.get("short_gt", "")).strip()
    body = str(record.get("long_gt", "")).strip()
    if title and body:
        return f"{title}. {body}"
    return title or body


def paper_records_for_dataset(dataset_name: str, pubmed_eval: list[dict[str, Any]], neurovault_eval: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if dataset_name == "pubmed":
        return [
            {
                "dataset": "pubmed",
                "sample": str(d["pmid"]),
                "latent": d["latent"],
                "text_latent": d["text_latent"],
                "text": paper_record_text(d),
            }
            for d in pubmed_eval
            if "text_latent" in d
        ]
    if dataset_name == "neurovault":
        return [
            {
                "dataset": "neurovault",
                "sample": str(d["doi"]),
                "latent": d["latent"],
                "text_latent": d["text_latent"],
                "text": paper_record_text(d),
            }
            for d in neurovault_eval
            if "text_latent" in d
        ]
    raise ValueError(f"Unknown paper retrieval dataset: {dataset_name}")


def semantic_recall_curve(scores: np.ndarray, positives: list[set[int]]) -> np.ndarray:
    order = np.argsort(-scores, axis=1)
    n_queries, n_candidates = scores.shape
    best_ranks = []
    for i, pos in enumerate(positives):
        ranks_by_candidate = np.empty(n_candidates, dtype=np.int64)
        ranks_by_candidate[order[i]] = np.arange(1, n_candidates + 1)
        best_ranks.append(min(int(ranks_by_candidate[j]) for j in pos))
    counts = torch.bincount(torch.as_tensor(best_ranks, dtype=torch.long) - 1, minlength=n_candidates)
    return (counts.cumsum(0).float() / float(n_queries)).numpy()


def run_paper_retrieval_eval(
    *,
    nvlm,
    dataset_name: str,
    pubmed_eval: list[dict[str, Any]],
    neurovault_eval: list[dict[str, Any]],
    semantic_neighbors: int = 10,
    brain_batch_size: int = 512,
    text_batch_size: int = 512,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    from neurovlm.semantic_evaluation import exact_pmid_retrieval_metrics, exact_recall_curve, semantic_neighbor_positive_sets

    records = paper_records_for_dataset(dataset_name, pubmed_eval, neurovault_eval)
    if len(records) < 2:
        return {}, pd.DataFrame(), pd.DataFrame()

    ids = [r["sample"] for r in records]
    brain_embeddings = project_brain_latents_to_shared(nvlm, [r["latent"] for r in records], batch_size=brain_batch_size)
    text_embeddings = project_text_latents_to_shared(nvlm, [r["text_latent"] for r in records], batch_size=text_batch_size)
    sim = brain_embeddings @ text_embeddings.T

    brain_to_paper = exact_pmid_retrieval_metrics(sim)
    paper_to_brain = exact_pmid_retrieval_metrics(sim.T)
    semantic_positives = semantic_neighbor_positive_sets(
        text_embeddings,
        n_neighbors=min(semantic_neighbors, max(len(records) - 1, 1)),
    )
    semantic_curve = semantic_recall_curve(sim.detach().cpu().numpy(), semantic_positives)
    semantic_auc = normalized_recall_curve_auc(torch.as_tensor(semantic_curve))

    metrics = {
        "dataset": dataset_name,
        "n_papers": len(records),
        "brain_to_paper_normalized_k_recall_curve_auc": brain_to_paper["normalized_k_recall_curve_auc"],
        "paper_to_brain_normalized_k_recall_curve_auc": paper_to_brain["normalized_k_recall_curve_auc"],
        "semantic_normalized_k_recall_curve_auc": semantic_auc,
    }

    normalized_k = normalized_k_values(len(records)).cpu().numpy()
    curve_df = pd.DataFrame(
        {
            "dataset": dataset_name,
            "k": np.arange(1, len(records) + 1),
            "normalized_k": normalized_k,
            "brain_to_paper_recall_curve": exact_recall_curve(sim).cpu().numpy(),
            "paper_to_brain_recall_curve": exact_recall_curve(sim.T).cpu().numpy(),
            "semantic_recall_curve": semantic_curve,
            "random_recall_curve": normalized_k,
        }
    )

    order = torch.argsort(sim, dim=1, descending=True).cpu().numpy()
    example_rows = []
    for i, sample_id in enumerate(ids):
        top = order[i, : min(10, len(ids))].tolist()
        correct_rank = int(np.where(order[i] == i)[0][0] + 1)
        example_rows.append(
            {
                "dataset": dataset_name,
                "sample": sample_id,
                "correct_rank": correct_rank,
                "top10_samples": "|".join(ids[j] for j in top),
                "top10_scores": "|".join(f"{float(sim[i, j]):.6f}" for j in top),
                "reference_text": records[i]["text"][:500],
            }
        )
    return metrics, curve_df, pd.DataFrame(example_rows)


def run_paper_retrieval_evaluations(
    *,
    nvlm,
    pubmed_eval: list[dict[str, Any]],
    neurovault_eval: list[dict[str, Any]],
    output_dir,
    run_pubmed: bool = True,
    run_neurovault: bool = True,
    semantic_neighbors: int = 10,
    brain_batch_size: int = 512,
    text_batch_size: int = 512,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    paper_metric_rows = []
    paper_curve_frames = []
    paper_example_frames = []
    for dataset_name, should_run in [("pubmed", run_pubmed), ("neurovault", run_neurovault)]:
        if not should_run:
            continue
        metrics, curves, examples = run_paper_retrieval_eval(
            nvlm=nvlm,
            dataset_name=dataset_name,
            pubmed_eval=pubmed_eval,
            neurovault_eval=neurovault_eval,
            semantic_neighbors=semantic_neighbors,
            brain_batch_size=brain_batch_size,
            text_batch_size=text_batch_size,
        )
        if metrics:
            paper_metric_rows.append(metrics)
            paper_curve_frames.append(curves)
            paper_example_frames.append(examples)

    metrics_df = pd.DataFrame(paper_metric_rows)
    curves_df = pd.concat(paper_curve_frames, ignore_index=True) if paper_curve_frames else pd.DataFrame()
    examples_df = pd.concat(paper_example_frames, ignore_index=True) if paper_example_frames else pd.DataFrame()
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_df.to_csv(output_dir / "b2t_paper_retrieval_metrics.csv", index=False)
    metrics_df.to_json(output_dir / "b2t_paper_retrieval_metrics.json", orient="records", indent=2)
    curves_df.to_csv(output_dir / "b2t_paper_retrieval_curves.csv", index=False)
    curves_df.to_json(output_dir / "b2t_paper_retrieval_curves.json", orient="records", indent=2)
    examples_df.to_csv(output_dir / "b2t_paper_retrieval_examples.csv", index=False)
    examples_df.to_json(output_dir / "b2t_paper_retrieval_examples.json", orient="records", indent=2)
    return metrics_df, curves_df, examples_df


def predownload_hf_model(model_name: str, tokenizer_cls, model_cls) -> None:
    print(f"Pre-downloading Hugging Face model for BERTScore: {model_name}")
    tokenizer_cls.from_pretrained(model_name, use_fast=False)
    model_cls.from_pretrained(model_name)
    print("BERTScore model is cached.")

__all__ = [
    "token_f1",
    "rouge",
    "bleu",
    "add_network_label_accuracy",
    "all_target_ranking_metrics",
    "all_target_recall_curve",
    "auc_trapezoid",
    "bertscore_single",
    "brain_latents_for_generated_group",
    "dataset_records_for_retrieval_eval",
    "exact_term_ranking_outputs",
    "generated_text_metric_summary",
    "generated_text_pair_baseline",
    "generated_text_retrieval_curve",
    "full_retrieval_table_for_sample",
    "k_from_normalized_k",
    "make_b2t_runner",
    "make_network_label_accuracy_adder",
    "network_gold_terms",
    "nvlm_latent_similarity",
    "predownload_hf_model",
    "project_brain_latents_to_shared",
    "project_text_latents_to_shared",
    "pubmed_abstract_lookup",
    "retrieval_table_for_sample",
    "run_b2t_sample",
    "run_network_gold_term_ranking",
    "run_paper_retrieval_eval",
    "run_paper_retrieval_evaluations",
    "run_pubmed_mesh_gold_ranking",
    "run_pubmed_mesh_node_type_rankings",
    "semantic_similarity",
    "terms_for_dataset",
    "unique_ranked_terms_from_table",
]
