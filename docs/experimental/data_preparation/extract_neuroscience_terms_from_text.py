# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.5
#   kernelspec:
#     display_name: neurovlm
#     language: python
#     name: neurovlm
# ---

# %% [markdown]
# # LLM Neuroscience Term Extraction From PubMed and NeuroVault
#
# This notebook expands the lookup-table corpus beyond MeSH and CogAtlas by asking an LLM to read each paper title and abstract and extract relevant neuroscience terms. It is designed to be resumable: every document-level extraction is cached, so full-test-set runs can be stopped and restarted without losing completed papers.
#
# The final table writes lowercase terms, categories, definitions, evidence snippets, document counts, and whether the term already appears in the existing NeuroVLM lookup sources.

# %%
import json
import os
import re
import time
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
from IPython.display import display

from neurovlm.data import load_dataset
from neurovlm.retrieval.summarization import _generate_with_ollama, _generate_with_huggingface

# %%
# LLM settings
LLM_BACKEND = "ollama"               # "ollama" | "huggingface"
LLM_MODEL = "qwen2.5:14b-instruct"   # best quality for corpus expansion; use qwen2.5:7b-instruct for faster drafts
MAX_NEW_TOKENS = 1200                # HuggingFace only
REQUEST_SLEEP_SECONDS = 0.0          # raise if your backend needs throttling
FORCE_RETRY_ERRORS = True          # retry documents cached as errors from earlier runs
OVERWRITE_OK_CACHE = False         # set True to regenerate successful cached docs with the current model
PROMPT_VERSION = "llm_terms_v2"    # bump this if you substantially change the extraction prompt

# Scale settings
MAX_DOCS = 2 if os.environ.get("NEUROVLM_SMOKE") else None  # set to None for all PubMed + NeuroVault docs
DOC_OFFSET = 0                       # useful for chunked full runs
BATCH_SIZE = 1                       # one paper per prompt is more accurate and easier to cache

# Output paths
OUTPUT_DIR = Path("docs/experimental/data_preparation/outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CACHE_PATH = OUTPUT_DIR / "llm_neuroscience_terms_doc_cache.json"
RAW_OUTPUT_PATH = OUTPUT_DIR / "llm_extracted_neuroscience_terms_by_doc.csv"
AGG_OUTPUT_PATH = OUTPUT_DIR / "llm_extracted_neuroscience_terms_lookup.csv"

ALLOWED_CATEGORIES = {
    "anatomical_region",
    "brain_network",
    "cognitive_function",
    "cognitive_concept",
    "method",
    "imaging_modality",
    "cognitive_task",
    "experimental_task",
    "disease_or_disorder",
    "population_or_phenotype",
    "measurement_or_statistic",
    "other_neuroscience_term",
}


# %%
def normalize_text(text):
    return re.sub(r"\s+", " ", str(text or "")).strip()


def normalize_term(term):
    term = normalize_text(term).lower()
    term = re.sub(r"^[^a-z0-9]+|[^a-z0-9]+$", "", term)
    term = re.sub(r"\s+", " ", term)
    return term


def normalize_category(category):
    category = normalize_term(category).replace(" ", "_").replace("-", "_")
    aliases = {
        "anatomy": "anatomical_region",
        "brain_region": "anatomical_region",
        "region": "anatomical_region",
        "network": "brain_network",
        "cognition": "cognitive_function",
        "cognitive_process": "cognitive_function",
        "task": "cognitive_task",
        "experimental_paradigm": "experimental_task",
        "disease": "disease_or_disorder",
        "disorder": "disease_or_disorder",
        "modality": "imaging_modality",
        "statistic": "measurement_or_statistic",
        "measure": "measurement_or_statistic",
    }
    category = aliases.get(category, category)
    return category if category in ALLOWED_CATEGORIES else "other_neuroscience_term"


def compact_definition(definition):
    definition = normalize_text(definition).lower()
    return definition[:600]


def extract_json_array(text):
    text = str(text or "").strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.IGNORECASE | re.DOTALL).strip()
    try:
        obj = json.loads(text)
        if isinstance(obj, dict) and "terms" in obj:
            return obj["terms"]
        if isinstance(obj, list):
            return obj
    except json.JSONDecodeError:
        pass
    match = re.search(r"\[.*\]", text, flags=re.DOTALL)
    if not match:
        return []
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return []


def clean_term_record(record):
    if not isinstance(record, dict):
        return None
    term = normalize_term(record.get("term", ""))
    if len(term) < 3 or term in {"brain", "human", "humans", "study", "result", "results"}:
        return None
    category = normalize_category(record.get("category", "other_neuroscience_term"))
    definition = compact_definition(record.get("definition", ""))
    evidence = normalize_text(record.get("evidence", ""))[:400]
    if not definition:
        definition = f"{term} is a neuroscience term identified from the source paper."
    return {
        "term": term,
        "category": category,
        "definition": definition,
        "evidence": evidence,
    }


# %%
def load_documents():
    docs = []

    pubmed = load_dataset("pubmed_text")
    title_col = "name" if "name" in pubmed.columns else ("title" if "title" in pubmed.columns else pubmed.columns[0])
    abs_col = "description" if "description" in pubmed.columns else ("abstract" if "abstract" in pubmed.columns else None)
    id_col = "pmid" if "pmid" in pubmed.columns else pubmed.columns[0]
    for _, row in pubmed.iterrows():
        title = normalize_text(row[title_col])
        abstract = normalize_text(row[abs_col] if abs_col else "")
        if title or abstract:
            docs.append({
                "dataset": "pubmed",
                "doc_id": str(row[id_col]),
                "title": title,
                "abstract": abstract,
            })

    nv = load_dataset("neurovault_text")
    title_col = "title" if "title" in nv.columns else nv.columns[1]
    abs_col = "abstract" if "abstract" in nv.columns else nv.columns[2]
    id_col = "doi" if "doi" in nv.columns else nv.columns[0]
    for _, row in nv.iterrows():
        title = normalize_text(row[title_col])
        abstract = normalize_text(row[abs_col])
        if title or abstract:
            docs.append({
                "dataset": "neurovault",
                "doc_id": str(row[id_col]),
                "title": title,
                "abstract": abstract,
            })

    df = pd.DataFrame(docs).drop_duplicates(["dataset", "doc_id"]).reset_index(drop=True)
    return df


docs_df = load_documents()
if MAX_DOCS is None:
    docs_eval = docs_df.iloc[DOC_OFFSET:].reset_index(drop=True)
else:
    docs_eval = docs_df.iloc[DOC_OFFSET:DOC_OFFSET + MAX_DOCS].reset_index(drop=True)

print(f"Loaded {len(docs_df):,} documents. This run will process {len(docs_eval):,}.")
display(docs_eval.head())

# %%
SYSTEM_PROMPT = """You are a careful neuroscience terminology curator.
Extract terms from neuroimaging paper titles and abstracts to expand a neuroscience lookup corpus beyond existing MeSH and CogAtlas vocabularies.
Return only valid JSON. Do not include markdown fences or explanatory text."""


def build_user_prompt(doc):
    title = normalize_text(doc["title"])
    abstract = normalize_text(doc["abstract"])
    # Keep prompts bounded for local models while preserving the most informative part.
    if len(abstract) > 4500:
        abstract = abstract[:4500]

    json_schema = """[
  {"term": "...", "category": "...", "definition": "...", "evidence": "..."}
]"""

    return f"""Read this paper title and abstract. Extract important neuroscience terms that would be useful in a lookup table for brain-to-text and text-to-brain retrieval.

Include terms in these categories when present:
- anatomical_region
- brain_network
- cognitive_function
- cognitive_concept
- method
- imaging_modality
- cognitive_task
- experimental_task
- disease_or_disorder
- population_or_phenotype
- measurement_or_statistic
- other_neuroscience_term

Rules:
- Return terms in lowercase.
- Prefer canonical terms over one-off phrases.
- Include abbreviations only when they are common or defined in the text, e.g. "default mode network" and "dmn" can both be included.
- Do not extract generic words like "brain", "human", "study", "result", "analysis", or "participant" unless part of a specific term.
- Definitions should be short, self-contained, and based on neuroscience knowledge plus the text context.
- Evidence should be a short quote or phrase from the title/abstract showing why the term was extracted.
- Return at most 25 terms.

Return JSON in exactly this shape:
{json_schema}

Paper id: {doc['dataset']}:{doc['doc_id']}
Title: {title}
Abstract: {abstract}
"""


def build_messages(doc):
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_user_prompt(doc)},
    ]


def call_llm(messages):
    if LLM_BACKEND == "ollama":
        return _generate_with_ollama(messages, model_name=LLM_MODEL, verbose=False)
    if LLM_BACKEND == "huggingface":
        return _generate_with_huggingface(messages, model_name=LLM_MODEL, max_new_tokens=MAX_NEW_TOKENS, verbose=False)
    raise ValueError(f"Unknown LLM_BACKEND: {LLM_BACKEND}")


# %% [markdown]
# ## Run LLM Extraction
#
# Run the next cell to actually call the LLM over `docs_eval`. Results are saved after every document to `CACHE_PATH`, so you can interrupt and resume safely. Documents cached as `error` are retried by default via `FORCE_RETRY_ERRORS = True`; successful cached docs are skipped unless `OVERWRITE_OK_CACHE = True`.
#

# %%
def load_cache():
    if CACHE_PATH.exists():
        return json.loads(CACHE_PATH.read_text())
    return {}


def save_cache(cache):
    tmp = CACHE_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(cache, indent=2, sort_keys=True))
    tmp.replace(CACHE_PATH)


def print_cached_errors(cache, n=5):
    errors = [item for item in cache.values() if item.get("status") == "error"]
    if not errors:
        return
    print(f"\nShowing {min(n, len(errors))} cached errors:")
    for item in errors[:n]:
        print(f"- {item.get('dataset')}:{item.get('doc_id')} | {item.get('error', 'unknown error')}")


def should_skip_cached(cached, force_retry_errors=False, overwrite_ok_cache=False):
    if not cached:
        return False
    if cached.get("status") == "ok":
        return not overwrite_ok_cache
    if cached.get("status") == "error":
        return not force_retry_errors
    return False


def run_llm_extraction(docs, force_retry_errors=FORCE_RETRY_ERRORS, overwrite_ok_cache=OVERWRITE_OK_CACHE):
    """Call the configured LLM for each document and cache document-level terms."""
    cache = load_cache()
    print(f"Loaded {len(cache):,} cached document extractions from {CACHE_PATH}")
    print(f"LLM backend/model: {LLM_BACKEND} / {LLM_MODEL}")
    print(f"Documents queued in this run: {len(docs):,}")
    print(f"Retry cached errors: {force_retry_errors}; overwrite successful cache: {overwrite_ok_cache}")
    print_cached_errors(cache, n=5)

    n_called = 0
    n_skipped = 0
    n_retried_errors = 0
    for doc in tqdm(docs.to_dict("records"), desc="LLM extracting terms"):
        cache_key = f"{doc['dataset']}:{doc['doc_id']}"
        cached = cache.get(cache_key)
        if should_skip_cached(cached, force_retry_errors=force_retry_errors, overwrite_ok_cache=overwrite_ok_cache):
            n_skipped += 1
            continue
        if cached and cached.get("status") == "error":
            n_retried_errors += 1

        try:
            raw = call_llm(build_messages(doc))
            parsed = extract_json_array(raw)
            terms = [clean_term_record(item) for item in parsed]
            terms = [item for item in terms if item is not None]
            cache[cache_key] = {
                "status": "ok",
                "dataset": doc["dataset"],
                "doc_id": doc["doc_id"],
                "title": doc["title"],
                "n_terms": len(terms),
                "terms": terms,
                "raw_response": raw,
                "llm_backend": LLM_BACKEND,
                "llm_model": LLM_MODEL,
                "prompt_version": PROMPT_VERSION,
            }
        except Exception as e:
            cache[cache_key] = {
                "status": "error",
                "dataset": doc["dataset"],
                "doc_id": doc["doc_id"],
                "title": doc["title"],
                "error": f"{type(e).__name__}: {e}",
                "llm_backend": LLM_BACKEND,
                "llm_model": LLM_MODEL,
                "prompt_version": PROMPT_VERSION,
            }

        n_called += 1
        save_cache(cache)
        if REQUEST_SLEEP_SECONDS:
            time.sleep(REQUEST_SLEEP_SECONDS)

    print(f"LLM calls made: {n_called:,}")
    print(f"Cached/skipped docs: {n_skipped:,}")
    print(f"Cached errors retried: {n_retried_errors:,}")
    print("Cache status counts:")
    print(pd.Series([v.get("status") for v in cache.values()]).value_counts().to_string())
    print_cached_errors(cache, n=5)
    return cache


# This is the cell that actually calls the LLM.
# For a smoke run, keep MAX_DOCS small in the config cell. For the full corpus, set MAX_DOCS = None.
cache = run_llm_extraction(docs_eval)

# %%
raw_rows = []
for cache_key, item in cache.items():
    if item.get("status") != "ok":
        continue
    for term in item.get("terms", []):
        raw_rows.append({
            "dataset": item["dataset"],
            "doc_id": item["doc_id"],
            "title": item.get("title", ""),
            "term": normalize_term(term.get("term", "")),
            "category": normalize_category(term.get("category", "")),
            "definition": compact_definition(term.get("definition", "")),
            "evidence": normalize_text(term.get("evidence", "")),
        })

raw_terms_df = pd.DataFrame(raw_rows).drop_duplicates(["dataset", "doc_id", "term", "category"]).reset_index(drop=True)
raw_terms_df.to_csv(RAW_OUTPUT_PATH, index=False)
print(f"Wrote document-level extracted terms: {RAW_OUTPUT_PATH} ({len(raw_terms_df):,} rows)")
display(raw_terms_df.head(20))


# %%
def load_existing_terms():
    existing = set()
    for dataset_name in ["cogatlas", "cogatlas_task", "cogatlas_disorder", "pubmed_mesh", "wiki", "ngrams"]:
        try:
            df = load_dataset(dataset_name)
        except Exception:
            continue
        for col in ["term", "title", "name", "description"]:
            if col in df.columns:
                existing.update(normalize_term(x) for x in df[col].dropna().astype(str).tolist())
                break
    return {x for x in existing if x}

existing_terms = load_existing_terms()
print(f"Loaded {len(existing_terms):,} existing lookup/source terms for novelty flagging.")

if raw_terms_df.empty:
    extracted_lookup = pd.DataFrame(columns=[
        "term", "category", "definition", "document_count", "datasets", "example_docs", "evidence_examples", "already_in_existing_sources"
    ])
else:
    grouped = []
    for (term, category), sub in raw_terms_df.groupby(["term", "category"]):
        definition_counts = Counter(sub["definition"].dropna().astype(str))
        best_definition = definition_counts.most_common(1)[0][0] if definition_counts else ""
        grouped.append({
            "term": term,
            "category": category,
            "definition": best_definition,
            "document_count": int(sub[["dataset", "doc_id"]].drop_duplicates().shape[0]),
            "datasets": "; ".join(sorted(sub["dataset"].unique())),
            "example_docs": "; ".join((sub["dataset"] + ":" + sub["doc_id"]).drop_duplicates().head(5)),
            "evidence_examples": " | ".join(sub["evidence"].dropna().astype(str).drop_duplicates().head(3)),
            "already_in_existing_sources": term in existing_terms,
        })
    extracted_lookup = pd.DataFrame(grouped).sort_values(["document_count", "term"], ascending=[False, True]).reset_index(drop=True)

extracted_lookup.to_csv(AGG_OUTPUT_PATH, index=False)
print(f"Wrote aggregated lookup candidates: {AGG_OUTPUT_PATH} ({len(extracted_lookup):,} rows)")
display(extracted_lookup.head(30))

# %%
if len(extracted_lookup):
    summary = extracted_lookup.groupby("category").agg(
        n_terms=("term", "count"),
        n_new_terms=("already_in_existing_sources", lambda x: int((~x).sum())),
        median_docs=("document_count", "median"),
        max_docs=("document_count", "max"),
    ).sort_values("n_terms", ascending=False)
    display(summary)

    ax = summary[["n_terms", "n_new_terms"]].plot(kind="bar", figsize=(11, 4), color=["steelblue", "darkorange"])
    ax.set_title("LLM-extracted lookup terms by category")
    ax.set_ylabel("Unique lowercase terms")
    ax.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "llm_extracted_terms_by_category.png", dpi=150, bbox_inches="tight")
    plt.show()
else:
    print("No terms extracted yet. Run the LLM extraction cell first.")
