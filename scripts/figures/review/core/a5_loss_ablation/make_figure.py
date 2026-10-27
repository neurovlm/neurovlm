#!/usr/bin/env python
"""Loss-ablation tables (reviewer 3.4) from outputs/table_{contrastive,text_to_brain}.csv -> outputs/a5_loss_ablation_table.{md,csv}
(one row per variant: main metric, mean +- sd over 3 seeds, and the change from the final model's loss) and
outputs/a5_loss_ablation_{contrastive,text_to_brain}.{csv,tex} (all metrics).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
CONTRASTIVE = {  # metric: mean of the two retrieval AUCs
    "symmetric_t0.07": ("symmetric InfoNCE, τ = 0.07 (final)", "symmetric InfoNCE, $\\tau$ 0.07 (final)"),
    "t2i_only": ("text → brain cross-entropy only", "text$\\rightarrow$brain CE only"),
    "i2t_only": ("brain → text cross-entropy only", "brain$\\rightarrow$text CE only"),
    "symmetric_t0.03": ("τ = 0.03", "symmetric, $\\tau$ 0.03"),
    "symmetric_t0.20": ("τ = 0.20", "symmetric, $\\tau$ 0.20"),
    "learnable_temp": ("learnable τ", "symmetric, learnable $\\tau$"),
    "cosine_only": ("no cross-entropy (cosine of matched pairs only)", "no contrastive CE (cosine only)"),
    "random_text_init": ("final loss, text head not initialised from the stage A head", "final loss, random text-head init"),
}
TEXT_TO_BRAIN = {  # metric: Dice of 70th-percentile masks (paper definition)
    "head_v2": ("latent MSE + map correlation terms (final head v2)", "head v2: latent MSE + correlation terms (final)"),
    "decoder_bce": ("voxel-wise BCE through the frozen decoder (submitted)", "BCE via decoder (submitted)"),
    "latent_mse": ("MSE on the 384-d latent", "latent MSE (library)"),
    "latent_cosine": ("cosine on the 384-d latent", "latent cosine"),
}


def markdown(df: pd.DataFrame) -> str:
    lines = ["| " + " | ".join(df.columns) + " |", "|" + "|".join("---" for _ in df.columns) + "|"]
    lines += ["| " + " | ".join(str(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join(lines) + "\n"


def pm(df: pd.DataFrame, k: str, m: str) -> str:
    return f"{df.loc[k, m + '_mean']:.3f} ± {df.loc[k, m + '_sd']:.3f}"


def tex_table(df: pd.DataFrame, *, caption: str, label: str, align: str) -> str:
    body = [" & ".join(str(v).replace("±", "$\\pm$") for v in row) + " \\\\" for row in df.itertuples(index=False)]
    return "\n".join(["\\begin{table}[t]", "\\centering", "\\small", f"\\caption{{{caption}}}", f"\\label{{{label}}}",
                      f"\\begin{{tabular}}{{{align}}}", "\\toprule", " & ".join(df.columns) + " \\\\", "\\midrule", *body,
                      "\\bottomrule", "\\end{tabular}", "\\end{table}", ""])


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path, default=HERE / "outputs")
    out = p.parse_args().out
    con = pd.read_csv(out / "table_contrastive.csv").set_index("variant")
    t2b = pd.read_csv(out / "table_text_to_brain.csv").set_index("variant")
    meta = json.loads((out / "metrics.json").read_text())
    n_test = meta.get("n_rows", {}).get("test", "")
    model = Path(meta.get("model_dir", "")).name
    n_seeds = len(meta.get("seeds", [])) or 3
    rows = [("contrastive heads", label, "retrieval AUC (mean of both directions)", pm(con, k, "mean_auc"),
             f"{con.loc[k, 'delta_mean_auc_vs_paper']:+.3f}") for k, (label, _) in CONTRASTIVE.items()]
    rows += [("text-to-brain head", label, "Dice (70th-percentile masks)", pm(t2b, k, "dice_matched"),
              f"{t2b.loc[k, 'delta_dice_vs_final']:+.3f}") for k, (label, _) in TEXT_TO_BRAIN.items()]
    df = pd.DataFrame(rows, columns=["Stage", "Loss variant", "Metric", f"Mean ± sd ({n_seeds} seeds)", "Change vs final"])
    df.to_csv(out / "a5_loss_ablation_table.csv", index=False)
    (out / "a5_loss_ablation_table.md").write_text(markdown(df))

    c = pd.DataFrame([{"Variant": tex, "tau": "learn." if k == "learnable_temp" else ("--" if k == "cosine_only" else f"{con.loc[k, 'temperature']:.2f}"),
                       "Seeds": int(con.loc[k, "n_seeds"]), "t2i AUC": pm(con, k, "t2i_auc"), "i2t AUC": pm(con, k, "i2t_auc"),
                       "R@10": pm(con, k, "mean_recall@10"), "MRR": pm(con, k, "mean_mrr"),
                       "Delta AUC": f"{con.loc[k, 'delta_mean_auc_vs_paper']:.3f}"} for k, (_, tex) in CONTRASTIVE.items()])
    t = pd.DataFrame([{"Variant": tex, "Seeds": int(t2b.loc[k, "n_seeds"]), "Dice": pm(t2b, k, "dice_matched"),
                       "Dice null": f"{t2b.loc[k, 'dice_null_mean']:.3f}", "Pearson": pm(t2b, k, "pearson_matched"),
                       "Pearson null": f"{t2b.loc[k, 'pearson_null_mean']:.3f}", "Map AUC": pm(t2b, k, "map_recall_auc"),
                       "Delta Dice": f"{t2b.loc[k, 'delta_dice_vs_final']:.3f}"} for k, (_, tex) in TEXT_TO_BRAIN.items()])
    c.to_csv(out / "a5_loss_ablation_contrastive.csv", index=False)
    t.to_csv(out / "a5_loss_ablation_text_to_brain.csv", index=False)
    where = f"PubMed test, n={n_test}; heads retrained on the autoencoder latents of the {model} model"
    (out / "a5_loss_ablation_contrastive.tex").write_text(tex_table(
        c, caption=f"Contrastive-head loss ablation (mean $\\pm$ sd over {n_seeds} head seeds; {where}). Delta = mean AUC minus the final "
                   "model's symmetric InfoNCE.", label="tab:a5_loss_contrastive", align="lrrrrrrr"))
    (out / "a5_loss_ablation_text_to_brain.tex").write_text(tex_table(
        t, caption=f"Text-to-brain head objective ablation (mean $\\pm$ sd over {n_seeds} head seeds; {where}). Dice: 70th-percentile masks of "
                   "the calibrated generated map and the reconstructed true map; null = deranged pairing; Delta = Dice minus head v2.",
        label="tab:a5_loss_t2b", align="lrrrrrrr"))
    print(markdown(df))


if __name__ == "__main__":
    main()
