#!/usr/bin/env python
"""Comment 2.m6: the [NETWORK] / [REGION] titles of the six worst test-set captions (Fig. R11, a7 pubmed_b2t_worst)
re-decoded free and with the title constraint (--title-constraint of run_correctness.py), in the recipe of the stage-2
test captions (scripts/pipeline/validate_models.py: greedy, 128 new tokens, projection temperature 0.035, repetition penalty
1.25, batches of 128 test maps in its order). The unconstrained run must reproduce the released test captions.
    .venv/bin/python scripts/figures/review/core/a6_steering_tokens/worst_case_titles.py   # GPU, ~1 min
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import _a6_common as C  # noqa: E402
import _final_model as M  # noqa: E402
import _hf_pins as T  # noqa: E402  (title constraint, as scripts/models/_common_models.py)
import run_correctness as R  # noqa: E402
from run_correctness import split_caption, title_tries  # noqa: E402

A7_PANELS = HERE.parent / "a7_failure_cases_percentiles" / "outputs" / "panels.json"
TEST_CAPTIONS = M.ROOT / "scripts" / "outputs" / "validation" / "final" / "df_gen_canonical.parquet"
RECIPE = dict(do_sample=False, num_beams=1, max_new_tokens=128, projection_temp=0.035, repetition_penalty=1.25)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model-dir", type=Path, default=M.DEFAULT_MODEL_DIR, help="Training run with the final model.")
    p.add_argument("--batch", type=int, default=128, help="Generation batch of the test captions.")
    p.add_argument("--device", default="cuda")
    p.add_argument("--out", type=Path, default=HERE / "outputs" / "worst_case_titles.csv")
    args = p.parse_args()
    import torch

    C.set_offline_env()
    M.set_model_dir(args.model_dir)
    R.VOCAB_FILE = args.model_dir / R.VOCAB_FILE.name
    worst = [x["id"].replace("PMID ", "") for x in json.loads(A7_PANELS.read_text())["pubmed_b2t_worst"]]
    ref = pd.read_parquet(TEST_CAPTIONS)
    order = ref["pmid"].astype(str).tolist()
    X = C.load_pubmed_maps(np.array([int(x) for x in order]))
    nvlm = M.make_nvlm(args.device, M.component("autoencoder", args.device), M.load_projector(args.device))
    lm, tok = nvlm._neuro_qwen_model, nvlm._neuro_qwen_tokenizer
    rows = []
    for start in sorted({order.index(w) // args.batch * args.batch for w in worst}):
        sl = slice(start, start + args.batch)
        d = {"ids": order[sl], "X": X[sl].numpy(), "binary": False}
        for basis in ("network", "region"):
            for mode in ("none", "vocab", "shortlist"):
                a = SimpleNamespace(title_constraint=mode, shortlist_k=50, cache_dir=HERE / "outputs" / "cache")
                with T.constrained_titles(lm, T.TitleConstraint(tok, title_tries(a, d, basis, tok), 1)):
                    out = M.nvlm_generate(nvlm, X[sl], basis=basis, seed=123 + start, **RECIPE)
                for pm, text in zip(d["ids"], out):
                    if pm in worst:
                        rows.append({"rank": worst.index(pm) + 1, "pmid": pm, "token": basis, "constraint": mode,
                                     "title": split_caption(text)[0], "caption": text})
            torch.cuda.empty_cache()
    df = pd.DataFrame(rows).sort_values(["rank", "token", "constraint"])
    vocab = T.read_title_vocabulary(args.model_dir / "title_vocabulary.json")["titles"]
    key = lambda t: " ".join(str(t).lower().split())
    df["canonical"] = [key(t) in {key(n) for n in vocab[b]} for t, b in zip(df["title"], df["token"])]
    released = {(str(pm), b): split_caption(ref.set_index(ref["pmid"].astype(str)).loc[str(pm), b])[0] for pm in worst for b in ("network", "region")}
    df["released_title"] = [released[(pm, b)] for pm, b in zip(df["pmid"], df["token"])]
    df.to_csv(args.out, index=False)
    none = df[df["constraint"] == "none"]
    print(f"unconstrained titles equal to the released test captions: {int((none['title'] == none['released_title']).sum())} / {len(none)}")
    print(df.pivot_table(index=["rank", "pmid", "token"], columns="constraint", values="title", aggfunc="first").to_string())
    print(df.groupby("constraint")["canonical"].agg(["sum", "size"]).to_string())


if __name__ == "__main__":
    main()
