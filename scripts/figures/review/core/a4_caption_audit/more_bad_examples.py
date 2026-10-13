#!/usr/bin/env python
"""Twenty more uninformative table captions for outputs/caption_examples.{md,csv} (comment 3.7).

Draws 5 captions of each kind from caption_flags.parquet (seed 0; the 40 existing examples skipped) and appends them to
both files; idempotent.  CPU, ~10 s:
    .venv/bin/python scripts/figures/review/core/a4_caption_audit/more_bad_examples.py
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from caption_usability import GENERIC, OUT, SEED, STAT_EXPR, clean_caption, content_words, load_papers, md_table, strip_label  # noqa: E402

N_PER_KIND = 5
SECTION = "## 20 more uninformative captions"
CODE = r"(?:[A-Z][A-Z0-9]{1,5}|[A-Za-z]{1,3}\d{1,2}[A-Za-z]?)"  # short upper-case or letter+digit code
CODE_RE = re.compile(r"(?<![\w-])(" + CODE + r")\s*(>|<|–|−|—|-|vs\.?|minus)\s*(" + CODE + r")(?![\w-])")
# standard method / statistics / region abbreviations: not condition codes
STANDARD = set("MNI BOLD FWE FDR ROI SPM MRI PET CT ICA IC BA SVC TFCE ALE EEG MEG VBM DTI FA MD ALFF FC RSFC GM WM CSF GMV "
               "DMN SMA IFG ACC PCC DLPFC OFC STG MTG IPL SPL IPS TPJ LC NE PPI".split())


def letters(t: str) -> str:
    return re.sub(r"\d", "", t).upper()


def is_code_pair(m: re.Match) -> bool:
    a, joiner, b = m.group(1), m.group(2), m.group(3)
    if {letters(a), letters(b)} & STANDARD:
        return False
    return not (joiner in "–−—-" and letters(a) == letters(b))  # "IC1–IC8" is a range, not a contrast


def n_specific(text: str) -> int:
    return sum(w not in GENERIC and w != "simple" for w in content_words(clean_caption(text)))  # "simple effects" (ANOVA)


def kind(caption: str, category: str) -> str | None:
    """First match of: condition code (only an opaque code such as "JUDGE –BODY"), statistics only (a threshold or
    correction and no specific word), region list only, boilerplate."""
    raw = strip_label(caption or "")
    s = STAT_EXPR.sub(" ", raw)
    if len(raw.split()) < 2:  # empty or one-word captions: nothing to show in the table
        return None
    codes = [m for m in CODE_RE.finditer(s) if is_code_pair(m)]
    if codes and n_specific(CODE_RE.sub(" ", s)) == 0:
        return "condition code"
    if STAT_EXPR.search(raw) and n_specific(raw) == 0:
        return "statistics only"
    if category == "region list":
        return "region list only"
    if category == "boilerplate":
        return "boilerplate"
    return None


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path, default=OUT, help="Directory with caption_usability.py's outputs; the examples are added there.")
    out = p.parse_args().out
    ex = pd.read_csv(out / "caption_examples.csv", dtype={"table_label": str})
    ex = ex[~ex["stratum"].str.startswith("uninformative")]  # the 40 original rows
    assert len(ex) == 40, len(ex)
    f = pd.read_parquet(out / "caption_flags.parquet")
    f["kind"] = [kind(c, k) for c, k in zip(f["caption_text"].fillna(""), f["category"])]
    seen = set(zip(ex["pmid"].astype(int), ex["caption_full"].fillna("").str.strip()))
    f = f[[(int(p), str(c).strip()) not in seen for p, c in zip(f["pmid"], f["caption_text"].fillna(""))]]
    rng = np.random.default_rng(SEED)
    picks = []
    for k in ("condition code", "statistics only", "region list only", "boilerplate"):
        pool = f.index[f["kind"] == k].to_numpy()
        print(f"{k}: {len(pool)} tables")
        picks += [(k, i) for i in rng.choice(pool, size=N_PER_KIND, replace=False)]
    new = f.loc[[i for _, i in picks]].copy()
    new["stratum"] = [f"uninformative: {k}" for k, _ in picks]
    new["caption_full"] = new["caption_text"]
    titles = load_papers().set_index("pmid")["title"]
    new["title"] = new["pmid"].astype("int64").map(titles)
    new = new[[c for c in ex.columns if c in new.columns]]
    pd.concat([ex, new], ignore_index=True).to_csv(out / "caption_examples.csv", index=False)

    md = (out / "caption_examples.md").read_text().split("\n" + SECTION)[0].rstrip("\n")
    md = md.replace("40 seeded examples", "60 seeded examples", 1)
    table = md_table(pd.DataFrame({
        "Kind": [k for k, _ in picks], "PMID": new["pmid"].astype(str), "Table": new["table_label"].fillna(""),
        "Caption (full)": new["caption_full"].fillna("").str.strip(),
        "Coords (labelled / n)": [f"{a}/{b}" for a, b in zip(new["n_labelled"], new["n_coords"])],
        "HO top-3 (count)": new["top3_str"], "Paper title": new["title"].fillna("")}))
    (out / "caption_examples.md").write_text(
        md + f"\n\n{SECTION} (n = {len(new)})\n\nRound 3 addition: 5 per kind, `numpy.random.default_rng(0)` over "
        "`caption_flags.parquet`, the 40 rows above excluded (`more_bad_examples.py`). Kinds: condition code (only an opaque "
        "code such as JUDGE –BODY), statistics only, region list only, boilerplate.\n\n" + table)
    print(table)


if __name__ == "__main__":
    main()
