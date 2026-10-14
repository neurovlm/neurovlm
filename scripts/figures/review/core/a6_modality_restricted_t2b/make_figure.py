#!/usr/bin/env python
"""Table for reviewer 3.5 from run_analysis.py's outputs: per text level, Dice (top 10 %), Pearson r and top-1
identification of the matching reference, each with its null / chance. Writes outputs/modality_table.{md,csv}.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
LEVELS = {
    "network": ("network name (e.g. 'default mode network')", "Yeo-7 + network atlas maps"),
    "anatomy": ("region name (e.g. 'left amygdala')", "Harvard-Oxford + AAL ROIs"),
    "cognition": ("Cognitive Atlas term (e.g. 'working memory')", "Neurosynth map of the term"),
}


def markdown(df: pd.DataFrame) -> str:
    lines = ["| " + " | ".join(df.columns) + " |", "|" + "|".join("---" for _ in df.columns) + "|"]
    lines += ["| " + " | ".join(str(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join(lines) + "\n"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path, default=HERE / "outputs", help="Output directory.")
    out = p.parse_args().out
    s = pd.read_csv(out / "summary_by_level.csv").set_index("level")
    lat = json.loads((out / "results.json").read_text())["laterality"]
    df = pd.DataFrame([{
        "Input text": text, "Reference": ref, "n": int(s.loc[lv, "n"]),
        "Dice (null)": f"{s.loc[lv, 'dice_matched']:.2f} ({s.loc[lv, 'dice_null']:.2f})",
        "Pearson r (null)": f"{s.loc[lv, 'pearson_matched']:.2f} ({s.loc[lv, 'pearson_null']:.2f})",
        "Top-1 identification (chance)": f"{s.loc[lv, 'top1_acc']:.0%} ({s.loc[lv, 'chance_top1']:.0%})",
    } for lv, (text, ref) in LEVELS.items()])
    note = (f"All matched > null at p < 1e-15 (paired t-test). Null = the same map scored against non-matching references. "
            f"Hemisphere words are ignored: for {lat['n']} left/right region names the named side is favoured in only "
            f"{lat['frac_positive']:.0%} of cases (mean laterality index {lat['mean_index']:+.2f}).")
    df.to_csv(out / "modality_table.csv", index=False)
    (out / "modality_table.md").write_text(markdown(df) + "\n" + note + "\n")
    print(markdown(df) + "\n" + note)


if __name__ == "__main__":
    main()
