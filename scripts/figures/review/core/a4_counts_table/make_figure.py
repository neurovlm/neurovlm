#!/usr/bin/env python
"""Dataset-count table (comments 1.2, 1.m1) and per-stage split table of the final model (2.2) from outputs/counts.json
and stage_splits.csv.

    .venv/bin/python scripts/figures/review/core/a4_counts_table/make_figure.py   # outputs/counts_table.{md,csv,tex}, stage_splits.{md,tex}
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent


def markdown(df: pd.DataFrame) -> str:
    lines = ["| " + " | ".join(df.columns) + " |", "|" + "|".join("---" for _ in df.columns) + "|"]
    lines += ["| " + " | ".join(str(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join(lines) + "\n"


def esc(v) -> str:
    return str(v).replace("_", "\\_").replace("[", "{[}").replace("]", "{]}").replace("->", "$\\rightarrow$").replace(" x ", " $\\times$ ")


def latex_counts(counts: pd.DataFrame) -> str:
    """Dataset-count table as LaTeX (booktabs)."""
    body = [" & ".join(esc(v).replace("> ", "$>$ ") for v in row) + " \\\\" for row in counts.itertuples(index=False)]
    return "\n".join(["% written by a4_counts_table/make_figure.py", "\\begin{table}[t]", "\\centering", "\\small",
                      "\\caption{Dataset counts recomputed from the released data.}", "\\label{tab:a4_counts}",
                      "\\begin{tabular}{p{5.2cm}p{9cm}}", "\\toprule", "Question & Released data \\\\", "\\midrule", *body,
                      "\\bottomrule", "\\end{tabular}", "\\end{table}", ""])


def latex_stages(stages: pd.DataFrame, model_dir: str) -> str:
    """Per-stage table as LaTeX (booktabs)."""
    body = [" & ".join(esc(v) for v in row) + " \\\\" for row in stages.itertuples(index=False)]
    return "\n".join(["% written by a4_counts_table/make_figure.py", "\\begin{table}[t]", "\\centering", "\\small",
                      f"\\caption{{Samples used by each training stage of the final model ({esc(model_dir)}), by split.}}",
                      "\\label{tab:a4_stages}", "\\begin{tabular}{lp{5cm}rrrl}", "\\toprule",
                      "Stage & Data & Train & Val & Test & Unit \\\\", "\\midrule", *body, "\\bottomrule", "\\end{tabular}",
                      "\\end{table}", ""])


def counts_rows(c: dict) -> list[tuple[str, str]]:
    pm, sp, nv, net = c["pubmed"], c["splits"], c["neurovault"], c["networks"]
    q = pm["coords_per_publication"]
    excluded = pm["flag_none_(excluded_from_splits)"]
    return [
        ("Coordinates combined across experiments / contrasts?",
         "Yes. One map per paper: every peak of the paper, ALE kernel (9 mm FWHM), voxel-wise maximum."),
        ("Positive vs negative effects?", "Not separable: the released coordinates carry no sign, so both are pooled."),
        ("Several coordinate tables per DOI?", "Pooled into the paper's single map (no table identifiers are released)."),
        ("Publications / peaks", f"{pm['publications']:,} papers; {pm['coordinate_rows']:,} peaks "
                                 f"(median {q['p50']:.0f} per paper, IQR {q['p25']:.0f}-{q['p75']:.0f})"),
        ("Experiments / contrasts / tables", "Not identifiable in the released data (columns: pmid, doi, x, y, z)."),
        ("Train / val / test papers", f"{sp['split_train']:,} / {sp['split_val']:,} / {sp['split_test']:,} by PMID; "
                                      f"{excluded:,} papers in no split (> 100 peaks or an all-zero map)"),
        ("NeuroVault maps (paper: 3,000 and 4,553)",
         f"{nv['released_images_(images_neurovault.parquet)']:,} maps / "
         f"{nv['released_publications_(publications_neurovault.parquet)']} papers released (= '3,000'); '4,553' is not "
         f"reproducible ({nv['scrape_after_map_type_filter_(input_to_03_neurovault)']:,} statistic maps before image QC)"),
        ("Network maps (paper: 300 and 100)",
         f"Neither reproducible: {net['network_maps_total_(networks_labels.parquet)']} maps = "
         f"{net['labelled_network_maps_(non-ICA,_used_for_Fig5_confusion)']} labelled + "
         f"{net['unlabelled_ICA_components_(HCP+UKB)']} ICA components"),
    ]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path, default=HERE / "outputs", help="Output directory.")
    out = p.parse_args().out
    c = json.loads((out / "counts.json").read_text())

    counts = pd.DataFrame(counts_rows(c), columns=["Question", "Released data"])
    counts.to_csv(out / "counts_table.csv", index=False)
    (out / "counts_table.md").write_text(markdown(counts))
    (out / "counts_table.tex").write_text(latex_counts(counts))

    st = pd.read_csv(out / "stage_splits.csv")
    stages = pd.DataFrame({"Training stage": st["stage"], "Data": st["data"],
                           "Train": st["train"].map("{:,}".format), "Val": st["val"].map("{:,}".format),
                           "Test": st["test"].map("{:,}".format), "Unit": st["unit"]})
    (out / "stage_splits.md").write_text(markdown(stages))
    (out / "stage_splits.tex").write_text(latex_stages(stages, c["stage_notes"]["model_dir"]))
    print(markdown(counts) + "\n" + markdown(stages))


if __name__ == "__main__":
    main()
