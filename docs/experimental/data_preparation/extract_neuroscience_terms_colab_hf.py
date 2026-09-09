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
#     display_name: Python 3
#     name: python3
# ---

# %% [markdown]
# # LLM Neuroscience Term Extraction — Colab Hugging Face GPU
#
# Colab version of `24_extract_neuroscience_terms_from_text.ipynb`. This uses a Hugging Face Qwen instruct model on the Colab GPU instead of local Ollama, with Drive-backed caching and resumable outputs.
#
# Recommended runtime: A100 GPU for `Qwen/Qwen2.5-14B-Instruct`. For T4/L4 or quick prompt tests, use `Qwen/Qwen2.5-7B-Instruct` with 4-bit loading.

# %% [markdown]
# ## 0 — Runtime Check

# %%
import subprocess, sys

gpu_info = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
if gpu_info.returncode != 0:
    raise RuntimeError('No GPU detected. In Colab, go to Runtime > Change runtime type > GPU.')
print(gpu_info.stdout)

# %% [markdown]
# ## 1 — Install Dependencies

# %%
import subprocess, sys

subprocess.run([
    sys.executable, '-m', 'pip', 'install', '-q',
    'transformers>=4.45', 'accelerate>=0.34', 'bitsandbytes>=0.43',
    'sentencepiece', 'protobuf', 'pandas', 'pyarrow', 'matplotlib', 'tqdm'
], check=True)
print('Dependencies installed.')

# %% [markdown]
# ## 2 — Clone NeuroVLM

# %%
import os, sys, subprocess
from pathlib import Path

REPO_URL = 'https://github.com/neurovlm/neurovlm.git'
REPO_BRANCH = 'main'  # change if needed
REPO_DIR = Path('/content/neurovlm')

if not REPO_DIR.exists():
    # !git clone --branch {REPO_BRANCH} {REPO_URL} {REPO_DIR}
else:
    # %cd {REPO_DIR}
    # !git fetch origin
    # !git checkout {REPO_BRANCH}
    # !git pull --ff-only || true

# %cd {REPO_DIR}
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '-e', '.'], check=True)
sys.path.insert(0, str(REPO_DIR / 'src'))
print(f'Repo ready at {REPO_DIR}')

# %% [markdown]
# ## 3 — Mount Drive and Set Paths

# %%
from pathlib import Path
from google.colab import drive

drive.mount('/content/drive')
DRIVE_ROOT = Path('/content/drive/MyDrive/neurovlm')
OUTPUT_DIR = DRIVE_ROOT / 'llm_term_extraction'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CACHE_PATH = OUTPUT_DIR / 'llm_neuroscience_terms_doc_cache_hf.json'
RAW_OUTPUT_PATH = OUTPUT_DIR / 'llm_extracted_neuroscience_terms_by_doc_hf.csv'
AGG_OUTPUT_PATH = OUTPUT_DIR / 'llm_extracted_neuroscience_terms_lookup_hf.csv'
print(f'Outputs will be written to {OUTPUT_DIR}')

# %% [markdown]
# ## 4 — Configure Model and Run Size
#
# Use `Qwen/Qwen2.5-14B-Instruct` on A100 for quality. If memory is tight, switch to `Qwen/Qwen2.5-7B-Instruct` and set `LOAD_IN_4BIT = True`.

# %%
MODEL_NAME = 'Qwen/Qwen2.5-14B-Instruct'
LOAD_IN_4BIT = False
MAX_NEW_TOKENS = 900
BATCH_SIZE = 4
DO_SAMPLE = False
TEMPERATURE = None

MAX_DOCS = 50          # smoke run; set to None for all PubMed + NeuroVault docs
DOC_OFFSET = 0
FORCE_RETRY_ERRORS = True
OVERWRITE_OK_CACHE = False
REQUEST_SLEEP_SECONDS = 0.0
PROMPT_VERSION = 'llm_terms_colab_hf_v1'

ALLOWED_CATEGORIES = {
    'anatomical_region', 'brain_network', 'cognitive_function', 'cognitive_concept',
    'method', 'imaging_modality', 'cognitive_task', 'experimental_task',
    'disease_or_disorder', 'population_or_phenotype', 'measurement_or_statistic',
    'other_neuroscience_term',
}

# %% [markdown]
# ## 5 — Load Model

# %%
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

print(f'Loading {MODEL_NAME}')
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = 'left'

quant_config = None
if LOAD_IN_4BIT:
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type='nf4',
        bnb_4bit_use_double_quant=True,
    )

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    device_map='auto',
    quantization_config=quant_config,
    trust_remote_code=True,
)
model.eval()
print('Model loaded.')

# %% [markdown]
# ## 6 — Load PubMed and NeuroVault Documents

# %%
import json, re, time
from collections import Counter
import pandas as pd
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from neurovlm.data import load_dataset


def normalize_text(text):
    return re.sub(r'\s+', ' ', str(text or '')).strip()


def normalize_term(term):
    term = normalize_text(term).lower()
    term = re.sub(r'^[^a-z0-9]+|[^a-z0-9]+$', '', term)
    term = re.sub(r'\s+', ' ', term)
    return term


def normalize_category(category):
    category = normalize_term(category).replace(' ', '_').replace('-', '_')
    aliases = {'anatomy': 'anatomical_region', 'brain_region': 'anatomical_region', 'region': 'anatomical_region', 'network': 'brain_network', 'cognition': 'cognitive_function', 'cognitive_process': 'cognitive_function', 'task': 'cognitive_task', 'experimental_paradigm': 'experimental_task', 'disease': 'disease_or_disorder', 'disorder': 'disease_or_disorder', 'modality': 'imaging_modality', 'statistic': 'measurement_or_statistic', 'measure': 'measurement_or_statistic'}
    category = aliases.get(category, category)
    return category if category in ALLOWED_CATEGORIES else 'other_neuroscience_term'


def compact_definition(definition):
    return normalize_text(definition).lower()[:600]


def load_documents():
    docs = []
    pubmed = load_dataset('pubmed_text')
    title_col = 'name' if 'name' in pubmed.columns else ('title' if 'title' in pubmed.columns else pubmed.columns[0])
    abs_col = 'description' if 'description' in pubmed.columns else ('abstract' if 'abstract' in pubmed.columns else None)
    id_col = 'pmid' if 'pmid' in pubmed.columns else pubmed.columns[0]
    for _, row in pubmed.iterrows():
        title = normalize_text(row[title_col])
        abstract = normalize_text(row[abs_col] if abs_col else '')
        if title or abstract:
            docs.append({'dataset': 'pubmed', 'doc_id': str(row[id_col]), 'title': title, 'abstract': abstract})

    nv = load_dataset('neurovault_text')
    title_col = 'title' if 'title' in nv.columns else nv.columns[1]
    abs_col = 'abstract' if 'abstract' in nv.columns else nv.columns[2]
    id_col = 'doi' if 'doi' in nv.columns else nv.columns[0]
    for _, row in nv.iterrows():
        title = normalize_text(row[title_col])
        abstract = normalize_text(row[abs_col])
        if title or abstract:
            docs.append({'dataset': 'neurovault', 'doc_id': str(row[id_col]), 'title': title, 'abstract': abstract})
    return pd.DataFrame(docs).drop_duplicates(['dataset', 'doc_id']).reset_index(drop=True)


docs_df = load_documents()
docs_eval = docs_df.iloc[DOC_OFFSET:].reset_index(drop=True) if MAX_DOCS is None else docs_df.iloc[DOC_OFFSET:DOC_OFFSET + MAX_DOCS].reset_index(drop=True)
print(f'Loaded {len(docs_df):,} documents. This run will process {len(docs_eval):,}.')
display(docs_eval.head())

# %% [markdown]
# ## 7 — Prompt and JSON Parsing

# %%
SYSTEM_PROMPT = """You are a careful neuroscience terminology curator.
Extract terms from neuroimaging paper titles and abstracts to expand a neuroscience lookup corpus beyond existing MeSH and CogAtlas vocabularies.
Return only valid JSON. Do not include markdown fences or explanatory text."""


def build_user_prompt(doc):
    title = normalize_text(doc['title'])
    abstract = normalize_text(doc['abstract'])
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
    return [{'role': 'system', 'content': SYSTEM_PROMPT}, {'role': 'user', 'content': build_user_prompt(doc)}]


def extract_json_array(text):
    text = str(text or '').strip()
    text = re.sub(r'^```(?:json)?\s*|\s*```$', '', text, flags=re.IGNORECASE | re.DOTALL).strip()
    try:
        obj = json.loads(text)
        if isinstance(obj, dict) and 'terms' in obj:
            return obj['terms']
        if isinstance(obj, list):
            return obj
    except json.JSONDecodeError:
        pass
    match = re.search(r'\[.*\]', text, flags=re.DOTALL)
    if not match:
        return []
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return []


def clean_term_record(record):
    if not isinstance(record, dict):
        return None
    term = normalize_term(record.get('term', ''))
    if len(term) < 3 or term in {'brain', 'human', 'humans', 'study', 'result', 'results'}:
        return None
    definition = compact_definition(record.get('definition', ''))
    if not definition:
        definition = f'{term} is a neuroscience term identified from the source paper.'
    return {'term': term, 'category': normalize_category(record.get('category', 'other_neuroscience_term')), 'definition': definition, 'evidence': normalize_text(record.get('evidence', ''))[:400]}


# %% [markdown]
# ## 8 — Batched Hugging Face Generation

# %%
def generate_batch(docs):
    prompts = [tokenizer.apply_chat_template(build_messages(doc), tokenize=False, add_generation_prompt=True) for doc in docs]
    inputs = tokenizer(prompts, return_tensors='pt', padding=True, truncation=True, max_length=8192).to(model.device)
    gen_kwargs = {'max_new_tokens': MAX_NEW_TOKENS, 'do_sample': DO_SAMPLE, 'pad_token_id': tokenizer.eos_token_id, 'eos_token_id': tokenizer.eos_token_id}
    if TEMPERATURE is not None:
        gen_kwargs['temperature'] = TEMPERATURE
    with torch.no_grad():
        generated = model.generate(**inputs, **gen_kwargs)
    full_input_width = inputs['input_ids'].shape[1]
    return [tokenizer.decode(row[full_input_width:], skip_special_tokens=True).strip() for row in generated]


def load_cache():
    return json.loads(CACHE_PATH.read_text()) if CACHE_PATH.exists() else {}


def save_cache(cache):
    tmp = CACHE_PATH.with_suffix('.tmp')
    tmp.write_text(json.dumps(cache, indent=2, sort_keys=True))
    tmp.replace(CACHE_PATH)


def should_skip_cached(cached):
    if not cached:
        return False
    if cached.get('status') == 'ok':
        return not OVERWRITE_OK_CACHE
    if cached.get('status') == 'error':
        return not FORCE_RETRY_ERRORS
    return False


def run_llm_extraction(docs):
    cache = load_cache()
    queued, skipped, retried_errors = [], 0, 0
    for doc in docs.to_dict('records'):
        key = f"{doc['dataset']}:{doc['doc_id']}"
        cached = cache.get(key)
        if should_skip_cached(cached):
            skipped += 1
            continue
        if cached and cached.get('status') == 'error':
            retried_errors += 1
        queued.append(doc)
    print(f'Loaded {len(cache):,} cached docs from {CACHE_PATH}')
    print(f'Documents queued: {len(queued):,}; skipped: {skipped:,}; cached errors retried: {retried_errors:,}')

    calls = 0
    for start in tqdm(range(0, len(queued), BATCH_SIZE), desc='HF extracting terms'):
        batch = queued[start:start + BATCH_SIZE]
        try:
            raw_outputs = generate_batch(batch)
            batch_error = None
        except Exception as e:
            raw_outputs = [None] * len(batch)
            batch_error = f'{type(e).__name__}: {e}'
        for doc, raw in zip(batch, raw_outputs):
            key = f"{doc['dataset']}:{doc['doc_id']}"
            if raw is None:
                cache[key] = {'status': 'error', 'dataset': doc['dataset'], 'doc_id': doc['doc_id'], 'title': doc['title'], 'error': batch_error, 'model_name': MODEL_NAME, 'prompt_version': PROMPT_VERSION}
                continue
            parsed = extract_json_array(raw)
            terms = [clean_term_record(item) for item in parsed]
            terms = [item for item in terms if item is not None]
            cache[key] = {'status': 'ok', 'dataset': doc['dataset'], 'doc_id': doc['doc_id'], 'title': doc['title'], 'n_terms': len(terms), 'terms': terms, 'raw_response': raw, 'model_name': MODEL_NAME, 'prompt_version': PROMPT_VERSION}
        calls += len(batch)
        save_cache(cache)
        if REQUEST_SLEEP_SECONDS:
            time.sleep(REQUEST_SLEEP_SECONDS)
    print(f'Documents generated: {calls:,}')
    print(pd.Series([v.get('status') for v in cache.values()]).value_counts().to_string())
    return cache


cache = run_llm_extraction(docs_eval)

# %% [markdown]
# ## 9 — Write Document-Level Terms

# %%
raw_rows = []
for item in cache.values():
    if item.get('status') != 'ok':
        continue
    for term in item.get('terms', []):
        raw_rows.append({'dataset': item['dataset'], 'doc_id': item['doc_id'], 'title': item.get('title', ''), 'term': normalize_term(term.get('term', '')), 'category': normalize_category(term.get('category', '')), 'definition': compact_definition(term.get('definition', '')), 'evidence': normalize_text(term.get('evidence', ''))})
raw_terms_df = pd.DataFrame(raw_rows).drop_duplicates(['dataset', 'doc_id', 'term', 'category']).reset_index(drop=True)
raw_terms_df.to_csv(RAW_OUTPUT_PATH, index=False)
print(f'Wrote document-level extracted terms: {RAW_OUTPUT_PATH} ({len(raw_terms_df):,} rows)')
display(raw_terms_df.head(20))


# %% [markdown]
# ## 10 — Aggregate Lookup Candidates

# %%
def load_existing_terms():
    existing = set()
    for dataset_name in ['cogatlas', 'cogatlas_task', 'cogatlas_disorder', 'pubmed_mesh', 'wiki', 'ngrams']:
        try:
            df = load_dataset(dataset_name)
        except Exception:
            continue
        for col in ['term', 'title', 'name', 'description']:
            if col in df.columns:
                existing.update(normalize_term(x) for x in df[col].dropna().astype(str).tolist())
                break
    return {x for x in existing if x}

existing_terms = load_existing_terms()
if raw_terms_df.empty:
    extracted_lookup = pd.DataFrame(columns=['term', 'category', 'definition', 'document_count', 'datasets', 'example_docs', 'evidence_examples', 'already_in_existing_sources'])
else:
    rows = []
    for (term, category), sub in raw_terms_df.groupby(['term', 'category']):
        definition_counts = Counter(sub['definition'].dropna().astype(str))
        rows.append({'term': term, 'category': category, 'definition': definition_counts.most_common(1)[0][0] if definition_counts else '', 'document_count': int(sub[['dataset', 'doc_id']].drop_duplicates().shape[0]), 'datasets': '; '.join(sorted(sub['dataset'].unique())), 'example_docs': '; '.join((sub['dataset'] + ':' + sub['doc_id']).drop_duplicates().head(5)), 'evidence_examples': ' | '.join(sub['evidence'].dropna().astype(str).drop_duplicates().head(3)), 'already_in_existing_sources': term in existing_terms})
    extracted_lookup = pd.DataFrame(rows).sort_values(['document_count', 'term'], ascending=[False, True]).reset_index(drop=True)
extracted_lookup.to_csv(AGG_OUTPUT_PATH, index=False)
print(f'Wrote aggregated lookup candidates: {AGG_OUTPUT_PATH} ({len(extracted_lookup):,} rows)')
display(extracted_lookup.head(30))

# %% [markdown]
# ## 11 — Summary Plot

# %%
if len(extracted_lookup):
    summary = extracted_lookup.groupby('category').agg(n_terms=('term', 'count'), n_new_terms=('already_in_existing_sources', lambda x: int((~x).sum())), median_docs=('document_count', 'median'), max_docs=('document_count', 'max')).sort_values('n_terms', ascending=False)
    display(summary)
    ax = summary[['n_terms', 'n_new_terms']].plot(kind='bar', figsize=(11, 4), color=['steelblue', 'darkorange'])
    ax.set_title('HF LLM-extracted lookup terms by category')
    ax.set_ylabel('Unique lowercase terms')
    ax.grid(axis='y', alpha=0.25)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'llm_extracted_terms_by_category_hf.png', dpi=150, bbox_inches='tight')
    plt.show()
else:
    print('No terms extracted yet. Run the extraction cell first.')
