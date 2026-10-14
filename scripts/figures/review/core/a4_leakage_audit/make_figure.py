#!/usr/bin/env python
"""Leakage-audit table (comment 2.4): held-out papers flagged by each check and removed from every evaluation, from
outputs/removal_list.csv and outputs/neurovault_exclusions.csv (run_analysis.py) -> outputs/leakage_table.{md,csv}."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
CHECKS = {  # reason prefix in removal_list.csv (vs training, other held-out set or NeuroVault papers) -> table row
    "b_title_fuzzy": "near-identical title",
    "c_minilm_abstract": "near-identical title and abstract",
    "d_identical_coordinate_set": "identical coordinate table",
    "d_coordinate_jaccard": "near-identical coordinate table",
}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, default=HERE / "outputs", help="Directory with the audit outputs; the table is written there.")
    out = p.parse_args().out
    removal = pd.read_csv(out / "removal_list.csv")
    n_nv = len(pd.read_csv(out / "neurovault_exclusions.csv"))

    def flagged(split: str, reason: str = "") -> int:
        return int(((removal["split"] == split) & removal["reasons"].str.contains(reason)).sum())

    rows = [(text, flagged("test", code), flagged("val", code)) for code, text in CHECKS.items()]
    rows.append(("removed from every evaluation", flagged("test"), f"{flagged('val')}, plus {n_nv} NeuroVault papers"))
    df = pd.DataFrame(rows, columns=["Check against other papers", "Test set", "Validation set"])
    df.to_csv(out / "leakage_table.csv", index=False)
    md = "\n".join(["| " + " | ".join(df.columns) + " |", "|---|---|---|"] + ["| " + " | ".join(map(str, r)) + " |" for r in df.itertuples(index=False)])
    (out / "leakage_table.md").write_text(md + "\n")
    print(md)


if __name__ == "__main__":
    main()
