#!/usr/bin/env python
"""Re-create the PubMed summaries: Llama-3.3-70B-Instruct summaries of title[SEP]abstract through the HF router.

The summary prompt of docs/02_data/07_transform_text.ipynb (cell 3, verbatim) is the system message and the paper's
title[SEP]abstract (pinned neurovlm/neuro_image_papers publications.parquet) the user message, decoded as in cell 4
(temperature 0.3, top-p 0.9, top-k 50, 256 new tokens). The released summaries were made with Llama-3.3-70B-Instruct
(dataset card of neurovlm/pubmed_summary_qa). Decoding samples, so a new run gives a re-wording, not the same text.
The proof runs on test-set papers and puts each new summary next to the released ones with their MiniLM cosine.

    .venv/bin/python scripts/data/09_summaries.py                 # proof: 6 test-set papers, examples .md
    .venv/bin/python scripts/data/09_summaries.py --limit 0       # every paper (30,826 requests)

Needs a Hugging Face token (HF_TOKEN or `huggingface-cli login`). The router bills the account.
"""
from __future__ import annotations

import datetime
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # scripts/
from common import base_parser, finalize_args, get_logger, hf_pinned, timed  # noqa: E402

STAGE = "data_09_summaries"
HERE = Path(__file__).resolve().parent
ROUTER = "https://router.huggingface.co/v1"
MODEL_ID = "meta-llama/Llama-3.3-70B-Instruct"
MINILM = "sentence-transformers/all-MiniLM-L6-v2"
log = get_logger(STAGE)

# docs/02_data/07_transform_text.ipynb cell 3, verbatim (leading and trailing newline included)
SYSTEM_PROMPT = """
You are writing a short, topic-focused neuroscience mini-review.

INPUT:
You will receive:
    title + " [SEP] " + abstract from an fMRI publication.

GOAL:
Write a single coherent paragraph (4–6 sentences) that:
- Describes the scientific topic implied by the paper (e.g., meditation and the default mode
  network, exercise and working memory), not the paper’s methods.
- Explains the key cognitive processes (e.g., working memory, self-referential thought,
  emotion regulation) and the main brain systems or networks involved.
- May mention typical effects or relationships in general terms (e.g., “has been associated
  with changes in connectivity” or “can influence executive control”), but should stay high-level.

ALLOWED:
- General statements about effects or relationships (e.g., “exercise has been associated
  with improved cognition”, “meditation can alter DMN connectivity”).
- Generic references to “studies” or “research” in the field.
- Qualitative descriptions of brain activity or connectivity (e.g., “increased activation
  in prefrontal regions”, “changes in functional connectivity”).

AVOID:
- Exact numerical details (sample sizes, p-values, coordinates, time points).
- Detailed task or protocol descriptions from the specific study.
- Directly quoting or closely paraphrasing long sentences from the abstract.
- Overly narrow focus on this single experiment; write as if summarizing the topic area.

STYLE:
- Neutral, scientific tone, like a short review or encyclopedia entry.
- One paragraph only. No bullet points or headings.
- Focus on giving a clear sense of “what this topic is about” in the context of the field.
"""


def load_papers() -> pd.DataFrame:
    """Papers with both released summaries: pubmed_summary_qa (stage 13's training text) and neuro_summaries.parquet."""
    pubs = pd.read_parquet(hf_pinned("neurovlm/neuro_image_papers", "publications.parquet"))
    qa = pd.read_parquet(hf_pinned("neurovlm/pubmed_summary_qa", "pubmed_summary_qa.parquet"), columns=["pmid", "summary"])
    ns = pd.read_parquet(hf_pinned("neurovlm/neuro_image_papers", "neuro_summaries.parquet"), columns=["pmid", "summary"])
    ns["pmid"] = pd.to_numeric(ns["pmid"], errors="coerce").astype("Int64")
    df = pubs.merge(qa.rename(columns={"summary": "summary_qa"}), on="pmid", how="left")
    df = df.merge(ns.dropna().rename(columns={"summary": "summary_ns"}), on="pmid", how="left")
    return df


def generate(client, user: str, args) -> dict:
    """One chat completion, retried 4 times with backoff."""
    messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}]
    extra = {"top_k": args.top_k} if args.top_k else None
    err = None
    for attempt in range(4):
        try:
            out = client.chat.completions.create(model=args.model, messages=messages, max_tokens=args.max_new_tokens,
                                                 temperature=args.temperature, top_p=args.top_p, extra_body=extra)
            text = (out.choices[0].message.content or "").strip()
            if text:
                return {"summary": text, "finish_reason": out.choices[0].finish_reason}
            err = "empty reply"
        except Exception as e:
            err = f"{type(e).__name__}: {e}"
        time.sleep(min(2 ** attempt, 20))
    return {"summary": None, "finish_reason": None, "error": err}


def minilm(texts: list[str]) -> np.ndarray:
    from sentence_transformers import SentenceTransformer

    st = SentenceTransformer(MINILM, revision=hf_pinned(MINILM), device="cpu")
    return st.encode(texts, batch_size=32, normalize_embeddings=True, show_progress_bar=False)


def reading(cos: float, chance: float) -> str:
    if cos >= 0.8:
        return "approximately similar: same topic, different wording"
    if cos >= 0.6:
        return "similar topic, partly different content"
    return "different content" if cos < chance + 0.1 else "related topic, different content"


def write_examples(df: pd.DataFrame, settings: dict, path: Path) -> None:
    ok = df[df["summary"].notna()].reset_index(drop=True)
    n = len(ok)
    emb = minilm(ok["summary"].tolist() + ok["summary_ns"].tolist() + ok["summary_qa"].tolist())
    gen, ns, qa = emb[:n], emb[n:2 * n], emb[2 * n:]
    cos_ns, cos_qa = (gen * ns).sum(1), (gen * qa).sum(1)
    off = ~np.eye(n, dtype=bool)
    chance = float((gen @ ns.T)[off].mean())
    words = lambda s: len(str(s).split())  # noqa: E731
    lines = [
        "# PubMed summaries: re-creation examples", "",
        "Written by `scripts/data/09_summaries.py`. The summary prompt (`docs/02_data/07_transform_text.ipynb`, cell 3) is the",
        "system message and the paper's title and abstract, joined by `[SEP]`, the user message. Decoding samples, so each",
        "run gives a new wording. Each new summary is shown next to the two released summaries of the same test-set paper,",
        "`neuro_summaries.parquet` (neurovlm/neuro_image_papers) and `pubmed_summary_qa.parquet` (neurovlm/pubmed_summary_qa,",
        "whose card names Llama-3.3-70B-Instruct), with their MiniLM cosine.", "",
        f"Mean cosine to neuro_summaries: {cos_ns.mean():.2f} (same paper) against {chance:.2f} (other papers of this set).",
        f"Mean cosine to pubmed_summary_qa: {cos_qa.mean():.2f}. Mean length: new {ok['summary'].map(words).mean():.0f} words,",
        f"neuro_summaries {ok['summary_ns'].map(words).mean():.0f}, pubmed_summary_qa {ok['summary_qa'].map(words).mean():.0f}.", "",
        f"Settings: `{json.dumps(settings)}`", "",
    ]
    for i, r in ok.iterrows():
        lines += [f"## PMID {r['pmid']}: {r['name']}", "",
                  f"Reading: {reading(cos_ns[i], chance)} (cosine {cos_ns[i]:.2f} to neuro_summaries, {cos_qa[i]:.2f} to "
                  f"pubmed_summary_qa).", "",
                  "**New summary**", "", r["summary"], "",
                  "**Released, neuro_summaries.parquet**", "", f"> {r['summary_ns']}", "",
                  "**Released, pubmed_summary_qa.parquet**", "", f"> {r['summary_qa']}", ""]
    for r in df[df["summary"].isna()].itertuples():
        lines += [f"Failed: PMID {r.pmid}: {r.error}", ""]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n")
    log.info("wrote %s", path)


def main(argv=None):
    p = base_parser("Re-create the PubMed summaries with the summary prompt through the HF router.", stage=STAGE)
    p.add_argument("--provider", default="hf-router", choices=["hf-router"], help="Hugging Face Inference Providers router.")
    p.add_argument("--model", default=MODEL_ID, help="Router model id (a ':provider' or ':cheapest' suffix is allowed).")
    p.add_argument("--n-per-seed", type=int, default=1, help="Summaries per paper.")
    p.add_argument("--temperature", type=float, default=0.3)
    p.add_argument("--top-p", type=float, default=0.9)
    p.add_argument("--top-k", type=int, default=50, help="Sent as an extra field (0 = not sent).")
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--limit", type=int, default=6, help="Random test-set papers with released summaries (0 = every paper).")
    p.add_argument("--workers", type=int, default=8, help="Parallel requests.")
    p.add_argument("--examples", type=Path, default=HERE / "examples" / "summaries_examples.md",
                   help="Markdown with new and released summaries side by side (written when --limit > 0).")
    args = finalize_args(p.parse_args(argv))
    from huggingface_hub import InferenceClient, get_token

    with timed("load", log, out=args.out, stage=STAGE):
        df = load_papers()
    if args.limit:
        pool = df[df["test"].fillna(False).astype(bool) & df["summary_qa"].notna() & df["summary_ns"].notna()]
        df = pool.sample(n=args.limit, random_state=args.seed).sort_values("pmid")
    df = df.loc[df.index.repeat(args.n_per_seed)].reset_index(drop=True)
    df["sample"] = df.groupby("pmid").cumcount()
    users = (df["name"] + "[SEP]" + df["description"]).tolist()
    log.info("%d requests", len(users))

    settings = {"provider": args.provider, "router": ROUTER, "temperature": args.temperature, "top_p": args.top_p,
                "top_k": args.top_k or None, "max_tokens": args.max_new_tokens}
    client = InferenceClient(base_url=ROUTER, api_key=get_token())
    with timed("generate", log, out=args.out, stage=STAGE, n=len(users)):
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            results = list(ex.map(lambda u: generate(client, u, args), users))
    res = pd.DataFrame(results)
    df = pd.concat([df.reset_index(drop=True), res], axis=1)
    df["model"], df["settings"] = args.model, json.dumps(settings)
    df["timestamp"] = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")
    out = args.out / "summaries.parquet"
    df.drop(columns=["train", "val", "test"]).to_parquet(out, index=False)
    log.info("wrote %s (%d rows, %d failed)", out, len(df), int(df["summary"].isna().sum()))
    if args.limit:
        write_examples(df[df["sample"] == 0], {"model": args.model, **settings}, args.examples)
    return 0 if df["summary"].notna().all() else 1


if __name__ == "__main__":
    sys.exit(main())
