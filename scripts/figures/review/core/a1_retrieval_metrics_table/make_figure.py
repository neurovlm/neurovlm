#!/usr/bin/env python
"""Retrieval metrics table (Table R2; and a reference plot of the recall@k curves) from the run_analysis.py results.

    .venv/bin/python scripts/figures/review/core/a1_retrieval_metrics_table/make_figure.py
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
FIG_LETTER_W = 6.5  # response letter text width (scripts/pipeline/common.py)
DATASETS = {"pubmed": "PubMed", "neurovault": "NeuroVault, selected image*", "neurovault_first": "NeuroVault, first image"}
DIRECTIONS = {"text_to_brain": "Text → brain", "brain_to_text": "Brain → text"}
# Table R2: PubMed only, one block of rows per candidate set ("hard_negatives" = only the test papers of the query's
# own cognitive domain are candidates)
BLOCKS = {
    "all": "All candidates ({n:,.0f} test papers)",
    "hard_negatives": "Hard negatives (papers of the query's own domain, mean {n:,.0f})",
    "random_matched": "Random negatives (random papers, as many as the hard negatives)",
}
METRICS = {"recall@1": "Recall@1", "recall@5": "Recall@5", "recall@10": "Recall@10", "mrr": "MRR", "median_rank": "Median rank"}


def fmt(metric: str, v: float) -> str:
    if metric == "median_rank":
        return f"{v:,.0f}"
    return "<0.001" if v < 0.0005 else f"{v:.3f}"


def table_rows(m: pd.DataFrame) -> list[dict]:
    """Block header rows (metrics empty) followed by text -> brain, brain -> text and chance."""
    rows = []
    for cset, header in BLOCKS.items():
        n = m.loc[("pubmed", "text_to_brain", cset), "mean_candidates"]
        rows.append({"Candidates": header.format(n=n), "Direction": ""} | {lab: "" for lab in METRICS.values()})
        for direction, dlabel in DIRECTIONS.items():
            r = m.loc[("pubmed", direction, cset)]
            rows.append({"Candidates": "", "Direction": dlabel} | {lab: fmt(k, r[k]) for k, lab in METRICS.items()})
        r = m.loc[("pubmed", "text_to_brain", cset)]  # chance depends only on the pool sizes
        rows.append({"Candidates": "", "Direction": "Chance"} | {lab: fmt(k, r[f"chance_{k}"]) for k, lab in METRICS.items()})
    return rows


def markdown(rows: list[dict]) -> str:
    cols = ["", *METRICS.values()]
    lines = ["| " + " | ".join(cols) + " |", "|---|" + "|".join("---:" for _ in METRICS) + "|"]
    for r in rows:
        if r["Candidates"]:
            name, rest = r["Candidates"].split(" (", 1)
            lines.append(f"| **{name}** ({rest} |" + " |" * len(METRICS))
        else:
            lines.append(f"| {r['Direction']} | " + " | ".join(r[c] for c in METRICS.values()) + " |")
    return "\n".join(lines) + "\n"


def latex(rows: list[dict], caption: str) -> str:
    def tex(s: str) -> str:
        return s.replace("→", "$\\rightarrow$").replace("<0.001", "$<$0.001")

    out = ["% written by a1_retrieval_metrics_table/make_figure.py", "\\begin{table}[ht]", "\\centering", "\\small",
           f"\\caption{{{caption}}}", "\\label{tab:retrieval_metrics}", "\\begin{tabular}{lccccc}", "\\toprule",
           " & " + " & ".join(METRICS.values()) + " \\\\"]
    for r in rows:
        if r["Candidates"]:
            name, rest = r["Candidates"].split(" (", 1)  # the block's name in bold, its details plain
            out += ["\\midrule", f"\\multicolumn{{6}}{{l}}{{\\textbf{{{name}}} ({tex(rest)}}} \\\\"]
        else:
            label = f"\\textit{{{r['Direction']}}}" if r["Direction"] == "Chance" else tex(r["Direction"])
            out.append(f"\\quad {label} & " + " & ".join(tex(r[c]) for c in METRICS.values()) + " \\\\")
    out += ["\\bottomrule", "\\end{tabular}", "\\end{table}", ""]
    return "\n".join(out)


def recall_figure(out: Path) -> None:
    """Recall@k vs k (log x); rows = direction, columns = dataset."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8, "xtick.labelsize": 8, "ytick.labelsize": 8,
                         "legend.fontsize": 8, "legend.frameon": False, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": True, "grid.color": "#e1e0d9", "grid.linewidth": 0.6, "savefig.dpi": 300})
    c = pd.read_csv(out / "recall_curves.csv")
    styles = {"all": ("All candidates", "#2a78d6"), "hard_negatives": ("Hard negatives (same domain)", "#eb6834"),
              "random_matched": ("Random negatives, same number", "#1baf7a")}
    fig, axes = plt.subplots(2, 3, figsize=(FIG_LETTER_W, 5.2), layout="constrained")
    for i, (direction, dlabel) in enumerate(DIRECTIONS.items()):
        for j, (dataset, title) in enumerate(DATASETS.items()):
            ax = axes[i, j]
            sub = c[(c["dataset"] == dataset) & (c["direction"] == direction)]
            for cset, (lab, col) in styles.items():
                g = sub[sub["candidate_set"] == cset].sort_values("k")
                ax.plot(g["k"], g["recall"], color=col, ls="-", lw=1.4, label=lab)
            g = sub[sub["candidate_set"] == "hard_negatives"].sort_values("k")
            ax.plot(g["k"], g["chance"], color="#898781", ls=(0, (3, 2)), lw=1.0, label="Chance, hard negatives")
            ax.set_xscale("log")
            ax.margins(x=0.05, y=0.05)
            ax.set_box_aspect(1)
            if i == 0:
                ax.set_title(title)
            ax.set_xlabel("k (candidates ranked)")  # absolute k: the pools differ in size
            if j == 0:
                ax.set_ylabel(f"recall@k, {dlabel.lower()}")
    h, lab = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, lab, loc="outside lower center", ncol=2)
    for ext in ("png", "svg"):
        fig.savefig(out / f"retrieval_recall_curves.{ext}")
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path, default=HERE / "outputs")
    out = p.parse_args().out

    m = pd.read_csv(out / "retrieval_metrics.csv").set_index(["dataset", "direction", "candidate_set"])
    rows = table_rows(m)
    caption = ("Contrastive retrieval on the PubMed test set (2,971 papers) with the final model. Hard negatives are "
               "test papers from the query's own cognitive domain. Random negatives match their number.")
    tidy = pd.DataFrame(rows).replace("", None)
    tidy["Candidates"] = tidy["Candidates"].ffill()  # one row per direction, the block in the first column
    tidy.dropna(subset=["Direction"]).to_csv(out / "retrieval_metrics_table.csv", index=False)
    (out / "retrieval_metrics_table.md").write_text(markdown(rows) + "\n" + caption + "\n")
    (out / "retrieval_metrics_table.tex").write_text(latex(rows, caption))
    recall_figure(out)
    print(markdown(rows) + "\n" + caption)


if __name__ == "__main__":
    main()
