#!/usr/bin/env python
"""Re-create the synthetic texts: short LLM entries for brain regions, functions and networks.

The original generation code was not kept. This script repeats the process: the synthetic-text prompt of
docs/01_data/07_transform_text_gen.ipynb (cell 8, verbatim) sent to Qwen3.5-27B through the Hugging Face Inference
router (cell 9 settings), with the concept name alone as the user message. Seeds are region names of the nilearn atlases
(AAL SPM12 as in stage 11, and Harvard-Oxford), Cognitive Atlas concept names (neurovlm/cognitive_atlas) and the network
names of the `networks` dataset (neurovlm/embedded_text). Sampling a seed several times gives re-wordings of one concept.
The 1,338 curated [NETWORK]/[REGION]/[FUNCTION] texts and the rows of synth.parquet were made this way.

    .venv/bin/python scripts/data/08_synthetic_texts.py                      # proof: 2 seeds per level, examples .md
    .venv/bin/python scripts/data/08_synthetic_texts.py --limit 0 --n-per-seed 4 --temperature 0.7   # every seed

Needs a Hugging Face token (HF_TOKEN or `huggingface-cli login`). The router bills the account.
"""
from __future__ import annotations

import datetime
import gzip
import json
import pickle
import re
import string
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # scripts/
from common import base_parser, finalize_args, get_logger, hf_pinned, timed  # noqa: E402

STAGE = "data_08_synthetic_texts"
HERE = Path(__file__).resolve().parent
ROUTER = "https://router.huggingface.co/v1"
MODEL_ID = "Qwen/Qwen3.5-27B:cheapest"  # the notebook's id (cell 7). "Qwen/Qwen3.5-27B" lets the router pick
MINILM = "sentence-transformers/all-MiniLM-L6-v2"
log = get_logger(STAGE)

# docs/01_data/07_transform_text_gen.ipynb cell 8, verbatim
SYSTEM_PROMPT = """
Write a short educational neuroscience entry for the provided fMRI concept.

Requirements:
- Explain the concept in general neuroscience terms.
- Always include relevant anatomical regions.
- Connect the anatomy to the cognitive or neural function.
- Use cautious, broad language; avoid overclaiming.
- Do not mention studies, patients, diseases, disorders, treatments, age groups, tasks, results, methods, or statistics.
- Do not use: critical for, responsible for, causal, proven, this study, activation, altered, reduced, increased.
- Do not invent highly specific subregions unless they are commonly associated with the concept.
- If the concept is broad, name the main commonly associated regions or networks.
- If the concept is anatomical, describe its broader network/function associations.
- Write 3-5 complete sentences.

Return valid JSON only:
{
  "title": "1-5 words",
  "description": "3-5 sentences"
}
""".strip()

# AAL labels (stage 11 atlas) written out as region names, hemisphere dropped (Tzourio-Mazoyer et al., 2002)
AAL_NAMES = {
    "Precentral": "Precentral Gyrus", "Frontal_Sup": "Superior Frontal Gyrus",
    "Frontal_Sup_Orb": "Superior Frontal Gyrus, Orbital Part", "Frontal_Mid": "Middle Frontal Gyrus",
    "Frontal_Mid_Orb": "Middle Frontal Gyrus, Orbital Part", "Frontal_Inf_Oper": "Inferior Frontal Gyrus, Opercular Part",
    "Frontal_Inf_Tri": "Inferior Frontal Gyrus, Triangular Part", "Frontal_Inf_Orb": "Inferior Frontal Gyrus, Orbital Part",
    "Rolandic_Oper": "Rolandic Operculum", "Supp_Motor_Area": "Supplementary Motor Area", "Olfactory": "Olfactory Cortex",
    "Frontal_Sup_Medial": "Medial Superior Frontal Gyrus", "Frontal_Med_Orb": "Medial Orbitofrontal Cortex",
    "Rectus": "Gyrus Rectus", "Insula": "Insula", "Cingulum_Ant": "Anterior Cingulate Gyrus",
    "Cingulum_Mid": "Middle Cingulate Gyrus", "Cingulum_Post": "Posterior Cingulate Gyrus", "Hippocampus": "Hippocampus",
    "ParaHippocampal": "Parahippocampal Gyrus", "Amygdala": "Amygdala", "Calcarine": "Calcarine Cortex", "Cuneus": "Cuneus",
    "Lingual": "Lingual Gyrus", "Occipital_Sup": "Superior Occipital Gyrus", "Occipital_Mid": "Middle Occipital Gyrus",
    "Occipital_Inf": "Inferior Occipital Gyrus", "Fusiform": "Fusiform Gyrus", "Postcentral": "Postcentral Gyrus",
    "Parietal_Sup": "Superior Parietal Gyrus", "Parietal_Inf": "Inferior Parietal Lobule", "SupraMarginal": "Supramarginal Gyrus",
    "Angular": "Angular Gyrus", "Precuneus": "Precuneus", "Paracentral_Lobule": "Paracentral Lobule", "Caudate": "Caudate Nucleus",
    "Putamen": "Putamen", "Pallidum": "Globus Pallidus", "Thalamus": "Thalamus", "Heschl": "Heschl's Gyrus",
    "Temporal_Sup": "Superior Temporal Gyrus", "Temporal_Pole_Sup": "Superior Temporal Pole", "Temporal_Mid": "Middle Temporal Gyrus",
    "Temporal_Pole_Mid": "Middle Temporal Pole", "Temporal_Inf": "Inferior Temporal Gyrus",
}
ROMAN = {"1": "I", "2": "II", "3": "III", "4": "IV", "5": "V", "6": "VI", "7": "VII", "7b": "VIIB", "8": "VIII", "9": "IX", "10": "X"}
HO_SKIP = {"Cerebral White Matter", "Cerebral Cortex", "Lateral Ventricle"}  # tissue classes, not regions
COGATLAS_DROP = ["active cognitive inhibition", "transfer data"]  # as stage 05


def aal_name(label: str) -> str:
    base = re.sub(r"_[LR]$", "", label)
    if base in AAL_NAMES:
        return AAL_NAMES[base]
    if base.startswith("Cerebelum_Crus"):
        return f"Cerebellar Crus {ROMAN[base[-1]]}"
    if base.startswith(("Cerebelum_", "Vermis_")):
        parts = base.split("_")
        lobules = "-".join(ROMAN[p] for p in parts[1:])
        return f"Cerebellar Lobule {lobules}" if parts[0] == "Cerebelum" else f"Vermis {lobules}"
    raise KeyError(label)


def region_seeds() -> list[str]:
    from nilearn import datasets

    names = [aal_name(lab) for lab in datasets.fetch_atlas_aal(version="SPM12", verbose=0).labels if lab != "Background"]
    names += list(datasets.fetch_atlas_harvard_oxford("cort-maxprob-thr25-2mm", verbose=0).labels[1:])
    sub = [re.sub(r"^(Left|Right) ", "", lab) for lab in datasets.fetch_atlas_harvard_oxford("sub-maxprob-thr25-2mm", verbose=0).labels[1:]]
    names += [s for s in sub if s not in HO_SKIP]
    return names


def function_seeds() -> list[str]:
    terms = pd.read_parquet(hf_pinned("neurovlm/cognitive_atlas", "cogatlas.parquet"))["term"].str.strip()
    return [string.capwords(t) for t in terms if t.lower() not in COGATLAS_DROP]


def network_seeds() -> list[str]:
    """The `networks` dataset: its standard network names, then the atlas network names (ICA components dropped)."""
    canonical = pd.read_parquet(hf_pinned("neurovlm/embedded_text", "network_text.parquet"))["title"]
    names = [f"{t} Network" for t in canonical]
    with gzip.open(hf_pinned("neurovlm/embedded_text", "networks_arrays.pkl.gz"), "rb") as f:
        atlases = pickle.load(f)
    for nets in atlases.values():
        names += [n for n in nets if not re.fullmatch(r"ICA\d+|NONE", n)]
    return names


def norm(s: str) -> str:
    s = re.sub(r"\[(NETWORK|REGION|FUNCTION)\]", "", str(s)).lower()
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", " ", s.replace(" network", ""))).strip()


def dedupe(names: list[str]) -> list[str]:
    seen, out = set(), []
    for n in names:
        if norm(n) not in seen:
            seen.add(norm(n))
            out.append(n)
    return out


def published_rows() -> pd.DataFrame:
    """The released synthetic and curated texts: synth.parquet, text_synth_less.parquet, and stage 14's own tables."""
    sys.path.insert(0, str(HERE.parent / "models"))
    import _qformer_anat_data as qa

    synth = pd.read_parquet(hf_pinned("neurovlm/synthetic_pairs", "synth.parquet"))
    synth["source"] = np.where(synth["model"].notna(), "synth.parquet, Qwen row", "synth.parquet, curated or template row")
    less = pd.read_parquet(hf_pinned("neurovlm/synthetic_pairs", "text_synth_less.parquet")).assign(source="text_synth_less.parquet")
    tagged = pd.concat([qa.df_func, qa.df_more])["text"].str.split("\n", n=1, expand=True)
    tagged = pd.DataFrame({"title": tagged[0].str.replace(r"^\[\w+\]", "", regex=True), "description": tagged[1],
                           "source": "stage 14 tables"})
    df = pd.concat([synth[["title", "description", "source"]], less, tagged], ignore_index=True)
    return df.dropna(subset=["description"]).drop_duplicates(["title", "description"]).reset_index(drop=True)


def pick(seeds: list[str], k: int, titles: set[str], rng) -> list[str]:
    """k seeds, those with a released text of the same name first (so the output can be compared), in random order."""
    order = rng.permutation(len(seeds))
    ranked = sorted(order, key=lambda i: norm(seeds[i]) not in titles)
    return [seeds[i] for i in ranked[:k]]


def parse_json(content: str) -> dict:
    """Cell 9: the reply must be a JSON object with a title and a description."""
    data = json.loads(content.strip())
    title = " ".join(str(data["title"]).split())
    description = " ".join(str(data["description"]).split())
    if not title or not description:
        raise ValueError("empty title or description")
    return {"model_title": title, "description": description, "raw_text": content}


def generate(client, seed: str, args) -> dict:
    """One chat completion, retried 4 times with backoff as in cell 9."""
    messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": seed}]
    err = None
    for attempt in range(4):
        try:
            out = client.chat.completions.create(
                model=args.model, messages=messages, max_tokens=args.max_new_tokens, temperature=args.temperature,
                top_p=args.top_p, extra_body={"chat_template_kwargs": {"enable_thinking": False}})
            return parse_json(out.choices[0].message.content or "")
        except Exception as e:  # router errors and malformed JSON are retried
            err = f"{type(e).__name__}: {e}"
            time.sleep(min(2 ** attempt, 20))
    return {"model_title": None, "description": None, "raw_text": None, "error": err}


def minilm(texts: list[str]) -> np.ndarray:
    from sentence_transformers import SentenceTransformer

    st = SentenceTransformer(MINILM, revision=hf_pinned(MINILM), device="cpu")
    return st.encode(texts, batch_size=32, normalize_embeddings=True, show_progress_bar=False)


def write_examples(df: pd.DataFrame, pub: pd.DataFrame, settings: dict, path: Path) -> None:
    ok = df[df["description"].notna()].reset_index(drop=True)
    emb = minilm(ok["description"].tolist() + pub["description"].tolist())
    gen, ref = emb[: len(ok)], emb[len(ok):]
    sim = gen @ ref.T
    pub_norm = pub["title"].map(norm).to_numpy()
    same = [np.flatnonzero(pub_norm == norm(seed)) for seed in ok["seed"]]
    same_cos = np.array([sim[i, js].max() for i, js in enumerate(same) if len(js)] or [np.nan])
    lines = [
        "# Synthetic texts: re-creation examples", "",
        "Written by `scripts/data/08_synthetic_texts.py`. The original generation code was not kept, so this run repeats",
        "the process: the synthetic-text prompt (`docs/01_data/07_transform_text_gen.ipynb`, cell 8) with the concept name as",
        "the user message, sent through the Hugging Face Inference router. Each generated text is compared with the released",
        "text of the same name (synth.parquet, text_synth_less.parquet, stage 14 tables) by MiniLM cosine. *Closest* is the",
        "most similar released text of any name, shown when it is not the same-name text.", "",
        f"Cosine to the released text of the same name: {same_cos.min():.2f} to {same_cos.max():.2f} over {len(same_cos)} texts.",
        f"Median cosine of a generated text to all {len(pub):,} released texts: {np.median(sim):.2f}.", "",
        f"Settings: `{json.dumps(settings)}`", "",
    ]
    for (level, seed), g in ok.groupby(["level", "seed"], sort=False):
        lines += [f"## {level}: {seed}", ""]
        for i, r in g.iterrows():
            best, j = int(sim[i].argmax()), None
            lines += [f"**Sample {r['sample'] + 1}** (model title \"{r['model_title']}\")", "", r["description"], ""]
            if len(same[i]):
                j = same[i][sim[i, same[i]].argmax()]
                lines += [f"Released text of the same name ({pub.at[j, 'source']}), cosine {sim[i, j]:.2f}:", "",
                          f"> {pub.at[j, 'description']}", ""]
            else:
                lines += ["No released text has this name.", ""]
            if best != j:
                lines += [f"Closest released text: \"{pub.at[best, 'title']}\" ({pub.at[best, 'source']}), cosine {sim[i, best]:.2f}.", ""]
        if len(g) > 1:
            e = gen[g.index.to_numpy()]
            pair = e @ e.T
            lines += [f"Cosine between the re-wordings: {pair[np.triu_indices(len(g), 1)].mean():.2f}.", ""]
    for r in df[df["description"].isna()].itertuples():
        lines += [f"Failed: {r.level} {r.seed} sample {r.sample + 1}: {r.error}", ""]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n")
    log.info("wrote %s", path)


def main(argv=None):
    p = base_parser("Re-create the synthetic texts with the synthetic-text prompt through the HF router.", stage=STAGE)
    p.add_argument("--provider", default="hf-router", choices=["hf-router"], help="Hugging Face Inference Providers router.")
    p.add_argument("--model", default=MODEL_ID, help="Router model id (a ':provider' or ':cheapest' suffix is allowed).")
    p.add_argument("--n-per-seed", type=int, default=1, help="Samples per seed (re-wordings of one concept).")
    p.add_argument("--temperature", type=float, default=0.1)
    p.add_argument("--top-p", type=float, default=0.9)
    p.add_argument("--max-new-tokens", type=int, default=512)
    p.add_argument("--limit", type=int, default=6, help="Seeds in total, split evenly over the three levels (0 = every seed).")
    p.add_argument("--workers", type=int, default=8, help="Parallel requests (cell 9).")
    p.add_argument("--examples", type=Path, default=HERE / "examples" / "synthetic_texts_examples.md",
                   help="Markdown with the outputs and their closest released texts (written when --limit > 0).")
    args = finalize_args(p.parse_args(argv))
    from huggingface_hub import InferenceClient, get_token

    rng = np.random.default_rng(args.seed)
    with timed("seeds", log, out=args.out, stage=STAGE):
        levels = {"network": dedupe(network_seeds()), "region": dedupe(region_seeds()), "function": dedupe(function_seeds())}
        pub = published_rows()
    log.info("seeds %s, released texts %d", {k: len(v) for k, v in levels.items()}, len(pub))
    if args.limit:
        titles = set(pub["title"].map(norm))
        levels = {k: pick(v, args.limit // 3, titles, rng) for k, v in levels.items()}
    jobs = [(level, seed, s) for level, seeds in levels.items() for seed in seeds for s in range(args.n_per_seed)]

    settings = {"provider": args.provider, "router": ROUTER, "temperature": args.temperature, "top_p": args.top_p,
                "max_tokens": args.max_new_tokens, "enable_thinking": False, "n_per_seed": args.n_per_seed}
    client = InferenceClient(base_url=ROUTER, api_key=get_token())
    with timed("generate", log, out=args.out, stage=STAGE, n=len(jobs)):
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            results = list(ex.map(lambda j: generate(client, j[1], args), jobs))
    now = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")
    df = pd.DataFrame([{"level": lv, "seed": sd, "sample": s, **r} for (lv, sd, s), r in zip(jobs, results)])
    df["title"] = df["seed"]  # synth.parquet keeps the seed as the title
    df["text"] = (df["seed"] + " [sep] " + df["description"]).str.lower()
    df["model"], df["settings"], df["timestamp"] = args.model, json.dumps(settings), now
    cols = ["level", "seed", "sample", "title", "description", "raw_text", "model", "text", "model_title", "settings", "timestamp"]
    df = df[cols + (["error"] if "error" in df else [])]
    out = args.out / "synthetic_texts.parquet"
    df.to_parquet(out, index=False)
    log.info("wrote %s (%d rows, %d failed)", out, len(df), int(df["description"].isna().sum()))
    if args.limit:
        write_examples(df, pub, {"model": args.model, **settings}, args.examples)
    return 0 if df["description"].notna().all() else 1


if __name__ == "__main__":
    sys.exit(main())
