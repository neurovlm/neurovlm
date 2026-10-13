#!/usr/bin/env python
"""Topic-level LLM summaries of PubMed title[SEP]abstract (port of docs/02_data/07_transform_text.ipynb).
Default: write the released summaries (load_dataset("pubmed_summaries")) to the notebook's three files;
--regenerate runs Llama-3.1-8B-Instruct instead (gated, needs HF_TOKEN; hours on a GPU).

    .venv/bin/python scripts/data/07_transform_text.py [--smoke] [--force] [--regenerate --batch-size 32]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # scripts/
from common import artifacts_exist, base_parser, finalize_args, get_logger, hf_pinned, select_device, setup_torch, timed  # noqa: E402
from data_utils import add_data_args, hf_cache_pinned, resolve_artifact_dir  # noqa: E402

STAGE = "data_07_transform_text"
MODEL_ID = "meta-llama/Meta-Llama-3.1-8B-Instruct"
SMOKE_FALLBACK_MODEL = "Qwen/Qwen3-0.6B-Base"
BATCH_SIZE = 16
MAX_NEW_TOKENS = 256  # ~5–8 sentence paragraph
GEN_KWARGS = dict(do_sample=True, temperature=0.3, top_p=0.9, top_k=50)
log = get_logger(STAGE)

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


def _json_safe(v):
    return None if v is None or (not isinstance(v, str) and pd.isna(v)) else v


def model_is_cached(model_id: str) -> bool:
    """config.json of ``model_id`` at its pinned revision is in the HF cache."""
    from huggingface_hub import try_to_load_from_cache

    return isinstance(try_to_load_from_cache(model_id, "config.json", revision=hf_pinned(model_id)), str)


def read_jsonl(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame(columns=["doi", "summary", "pmid"])
    return pd.read_json(path, lines=True, dtype={"pmid": "Int64"})


def write_outputs(df_pubs: pd.DataFrame, jsonl_path: Path, parquet_path: Path, csv_path: Path):
    """Notebook cells 5-6: pubmed_text + summary -> parquet; jsonl joined with (doi, name, description) -> csv.

    Summaries are matched on pmid, not on DOI as in the notebook: 1,493 papers have no DOI and 1,498 DOIs repeat.
    """
    df_summaries = read_jsonl(jsonl_path)
    by_pmid = dict(zip(df_summaries["pmid"].astype("int64"), df_summaries["summary"]))
    out = df_pubs.copy()
    out["summary"] = out["pmid"].astype("int64").map(by_pmid)
    missing = int(out["summary"].isna().sum())
    if missing:
        log.warning("%d publications have no summary; they are dropped from %s", missing, parquet_path.name)
        out = out[out["summary"].notna()]
    out.to_parquet(parquet_path)
    sample = df_summaries.merge(df_pubs[["pmid", "doi", "name", "description"]].assign(pmid=lambda d: d["pmid"].astype("int64")),
                                on="pmid", how="left", suffixes=("", "_pub"))
    sample = sample[["doi", "summary", "name", "description", "pmid"]]
    sample.to_csv(csv_path, index=False)
    log.info("saved %s (%d rows), %s (%d rows)", parquet_path, len(out), csv_path, len(sample))
    return out


def materialise_from_release(df_pubs: pd.DataFrame, jsonl_path: Path):
    from neurovlm.data import load_dataset

    rel = load_dataset("pubmed_summaries")
    rel = rel.assign(pmid=rel["pmid"].astype("int64")).drop_duplicates("pmid")
    by_pmid = dict(zip(rel["pmid"], rel["summary"]))
    n = 0
    with jsonl_path.open("w", encoding="utf-8") as f_out:
        for pmid, doi in zip(df_pubs["pmid"].astype("int64"), df_pubs["doi"]):
            summary = by_pmid.get(int(pmid))
            if summary is None:
                continue
            f_out.write(json.dumps({"doi": _json_safe(doi), "summary": summary, "pmid": int(pmid)}, ensure_ascii=False) + "\n")
            n += 1
    log.info("materialised %d/%d released summaries into %s", n, len(df_pubs), jsonl_path)


def build_prompts(tokenizer, names, descs, prompt_format: str = "auto"):
    """System prompt + name[SEP]desc through the chat template (the notebook), or as plain text for base models
    (wrapped in a chat template they only echo the system prompt). auto = chat if the tokenizer has a template."""
    has_template = getattr(tokenizer, "chat_template", None) is not None
    use_chat = prompt_format == "chat" or (prompt_format == "auto" and has_template)
    if use_chat and not has_template:
        raise ValueError("--prompt-format chat requested but the tokenizer has no chat template")
    prompts = []
    for name, desc in zip(names, descs):
        if use_chat:
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"{name}[SEP]{desc}"},
            ]
            prompts.append(tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False))
        else:
            prompts.append(f"{SYSTEM_PROMPT}\n\n{name}[SEP]{desc}\n\nSummary:")
    return prompts


def generate_summaries(df_pubs: pd.DataFrame, jsonl_path: Path, *, model_id: str, device, batch_size: int,
                       max_new_tokens: int, max_input_tokens: int | None, prompt_format: str, out: Path, meta: dict):
    """Run the LLM on every pmid not yet in the jsonl (appends after each batch, so a rerun resumes)."""
    import torch
    from tqdm.auto import tqdm
    from transformers import AutoModelForCausalLM, AutoTokenizer

    done = read_jsonl(jsonl_path)
    done_pmids = set(done["pmid"].dropna().astype("int64").tolist())
    todo = df_pubs[~df_pubs["pmid"].astype("int64").isin(done_pmids)]
    if len(todo) == 0:
        log.info("%s already covers all %d publications", jsonl_path, len(df_pubs))
        return
    log.info("%d/%d publications still need a summary (resuming %d)", len(todo), len(df_pubs), len(done_pmids))

    with timed("load_llm", log, out=out, model=model_id, **meta):
        rev = hf_pinned(model_id)  # scripts/hf_manifest.json (KeyError for an unpinned --model)
        tokenizer = AutoTokenizer.from_pretrained(model_id, revision=rev, padding_side="left")
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        dtype = torch.bfloat16 if device.type in ("cuda", "cpu") else torch.float32
        model = AutoModelForCausalLM.from_pretrained(model_id, revision=rev, dtype=dtype).to(device)
        model.config.pad_token_id = tokenizer.pad_token_id
        model.eval()
        for prm in model.parameters():
            prm.requires_grad_(False)

    names, descs = todo["name"].tolist(), todo["description"].tolist()
    pmids, dois = todo["pmid"].astype("int64").tolist(), todo["doi"].tolist()
    prompts = build_prompts(tokenizer, names, descs, prompt_format)
    log.info("prompt format: %s", "chat template" if prompts[0] != f"{SYSTEM_PROMPT}\n\n{names[0]}[SEP]{descs[0]}\n\nSummary:" else "plain")
    if max_input_tokens is None:
        max_input_tokens = int(min(getattr(tokenizer, "model_max_length", 8192) or 8192, 8192))
    lengths = np.array([len(ids) for ids in tokenizer(prompts, add_special_tokens=False)["input_ids"]])
    order = np.argsort(-lengths, kind="stable")  # longest first: least padding, OOM shows up at once
    batches = [order[i:i + batch_size] for i in range(0, len(order), batch_size)]
    log.info("prompt tokens: median %d, max %d; %d batches of %d", int(np.median(lengths)), int(lengths.max()), len(batches), batch_size)

    n_new_tokens = 0
    with timed("generate", log, out=out, n_docs=len(todo), batch_size=batch_size, max_new_tokens=max_new_tokens, model=model_id, device=str(device), **meta) as t, \
            jsonl_path.open("a", encoding="utf-8") as f_out, torch.inference_mode():
        for idx in tqdm(batches, desc="summaries", total=len(batches)):
            enc = tokenizer([prompts[i] for i in idx], return_tensors="pt", padding=True, truncation=True, max_length=max_input_tokens).to(device)
            gen = model.generate(**enc, max_new_tokens=max_new_tokens, pad_token_id=tokenizer.pad_token_id, **GEN_KWARGS)
            new_tokens = gen[:, enc["input_ids"].shape[1]:]
            n_new_tokens += int((new_tokens != tokenizer.pad_token_id).sum())
            texts = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)
            for i, summary in zip(idx, texts):
                f_out.write(json.dumps({"doi": _json_safe(dois[i]), "summary": summary.strip(), "pmid": int(pmids[i])}, ensure_ascii=False) + "\n")
            f_out.flush()
            if device.type == "cuda":
                torch.cuda.empty_cache()
    log.info("generated %d summaries, %d new tokens (%.1f tok/s)", len(todo), n_new_tokens, n_new_tokens / max(t.seconds, 1e-9))
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()


def main(argv=None):
    p = base_parser("Topic-level LLM summaries of PubMed abstracts (released ones by default, --regenerate runs the LLM).", stage=STAGE)
    add_data_args(p, amp=False)
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--from-release", action="store_true", help="Write the released pubmed_summaries (default).")
    mode.add_argument("--regenerate", action="store_true", help="Run the LLM instead of using the released summaries.")
    p.add_argument("--summaries-dir", type=Path, default=None, help="Where neuro_summaries.* go (default: --artifact-dir).")
    p.add_argument("--model", default=MODEL_ID, help="Causal LM for --regenerate (pinned in scripts/hf_manifest.json).")
    p.add_argument("--max-new-tokens", type=int, default=None, help=f"Default {MAX_NEW_TOKENS} (64 in --smoke).")
    p.add_argument("--max-input-tokens", type=int, default=None, help="Prompt truncation length (default: min(model max, 8192)).")
    p.add_argument("--prompt-format", choices=["auto", "chat", "plain"], default="auto",
                   help="chat = notebook (chat template), plain = for base models, auto = chat if the tokenizer has a template.")
    args = finalize_args(p.parse_args(argv))
    art = resolve_artifact_dir(args)
    out_dir = Path(args.summaries_dir) if args.summaries_dir else art
    out_dir.mkdir(parents=True, exist_ok=True)
    setup_torch(args.seed)

    device = select_device(args.device)
    meta = dict(stage=STAGE, smoke=args.smoke)
    jsonl_path, parquet_path, csv_path = out_dir / "neuro_summaries.jsonl", out_dir / "neuro_summaries.parquet", out_dir / "neuro_summaries_sample.csv"
    if artifacts_exist(jsonl_path, parquet_path, csv_path, force=args.force):
        log.info("artifacts exist (%s); use --force to recompute", parquet_path)
        return 0
    if args.force and jsonl_path.exists():
        jsonl_path.unlink()

    from neurovlm.data import load_dataset

    hf_cache_pinned("neurovlm/neuro_image_papers")  # load_dataset takes no revision (pubmed_text, pubmed_summaries)
    with timed("load_dataset", log, out=args.out, **meta):
        df_pubs = load_dataset("pubmed_text")
        if args.max_samples:
            df_pubs = df_pubs.head(args.max_samples).reset_index(drop=True)

    if args.regenerate:
        model_id, prompt_format = args.model, args.prompt_format
        # --smoke: use a small cached base model when the Llama weights are not in the HF cache (no multi-GB download)
        if args.smoke and not model_is_cached(model_id):
            if not model_is_cached(SMOKE_FALLBACK_MODEL):
                raise RuntimeError(f"--smoke --regenerate: neither {model_id} nor {SMOKE_FALLBACK_MODEL} is in the HF cache")
            log.warning("--smoke: %s is not in the local HF cache; using %s (not the notebook's model)", model_id, SMOKE_FALLBACK_MODEL)
            model_id = SMOKE_FALLBACK_MODEL
            if prompt_format == "auto":
                prompt_format = "plain"
        max_new_tokens = args.max_new_tokens if args.max_new_tokens is not None else (64 if args.smoke else MAX_NEW_TOKENS)
        generate_summaries(df_pubs, jsonl_path, model_id=model_id, device=device, batch_size=args.batch_size or BATCH_SIZE,
                           max_new_tokens=max_new_tokens, max_input_tokens=args.max_input_tokens, prompt_format=prompt_format,
                           out=args.out, meta=meta)
    else:
        with timed("materialise_release", log, out=args.out, n_docs=len(df_pubs), **meta):
            materialise_from_release(df_pubs, jsonl_path)

    with timed("write_outputs", log, out=args.out, **meta):
        write_outputs(df_pubs, jsonl_path, parquet_path, csv_path)
    log.info("done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
