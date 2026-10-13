#!/usr/bin/env python
"""Brain-to-text table (generation vs retrieval) from the run_analysis.py results.

    .venv/bin/python scripts/figures/review/core/a1_brain_to_text_baselines/make_figure.py
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
ROWS = {  # key in results_table.csv -> label in the table (top to bottom)
    "neurovlm_generated": "NeuroVLM generation",
    "nn_contrastive_xmodal": "Retrieval: nearest training text to the map",
    "nn_contrastive_image": "Retrieval: text of the nearest training map",
    "random_train_text": "Random training text",
    "null_shuffled_neurovlm": "Null: generation vs another paper's reference",
}
METRICS = {"minilm_cos": "MiniLM cosine", "bertscore_f1": "BERTScore F1"}


def markdown(df: pd.DataFrame) -> str:
    lines = ["| " + " | ".join(df.columns) + " |", "|" + "|".join("---" for _ in df.columns) + "|"]
    lines += ["| " + " | ".join(str(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join(lines) + "\n"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path, default=HERE / "outputs")
    out = p.parse_args().out

    res = pd.read_csv(out / "results_table.csv").set_index(["metric", "method"])
    table = pd.DataFrame({
        "Text returned for a test map": list(ROWS.values()),
        **{label: [f"{res.loc[(m, k), 'mean']:.3f}" for k in ROWS] for m, label in METRICS.items()},
    })
    n = int(res.loc[("minilm_cos", "neurovlm_generated"), "n"])

    near = pd.read_csv(out / "nearest_train_diagnostic.csv").set_index(["space", "kind"])
    gen, ref = near.loc[("minilm", "generated")], near.loc[("minilm", "true_reference")]
    footnote = (f"PubMed test set, n = {n:,} maps; reference = title + LLM summary of the paper. Every difference from "
                "NeuroVLM is p < 0.001 (paired t-test, Holm). Nearest training text (MiniLM cosine, median): generated "
                f"{gen['50%']:.3f} vs real held-out text {ref['50%']:.3f}; above 0.90: {gen['frac_gt_0.90']:.0%} vs "
                f"{ref['frac_gt_0.90']:.0%}.")

    table.to_csv(out / "brain_to_text_baselines_table.csv", index=False)
    (out / "brain_to_text_baselines_table.md").write_text(markdown(table) + "\n" + footnote + "\n")
    print(markdown(table) + "\n" + footnote)


if __name__ == "__main__":
    main()
