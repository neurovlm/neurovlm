#!/usr/bin/env python
"""Q-Former vs MLP projector, stage 1 and stage 2 (reviewer 2.1), from the validator outputs
scripts/outputs/validation/{released_excl,final}/metrics.json -> outputs/a5_mlp_projector.{png,svg}.

``--table`` instead writes the variant-study table and results_summary.md from outputs/metrics.json (run_analysis.py).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[4]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO / "scripts" / "pipeline"))
from _a5_common import NEUTRAL, PALETTE, apply_style, read_json, save_fig, write_table  # noqa: E402
from common import FIG_FULL_W, FIG_LETTER_W, PAPER_FIG_DIR  # noqa: E402  paper (5.15 in) and response letter (6.5 in) text widths

VALIDATION = REPO / "scripts" / "outputs" / "validation"
INK = "#0b0b0b"
# (validator key, panel title, y label)
CORE = [("recall_auc", "a  Text finds\n    its map", "retrieval, recall-curve AUC"),
        ("contrastive_sim_matched", "b  Text matches\n    its map", "cosine, generated text vs map"),
        ("bertscore_f1_matched", "c  Text matches\n    the paper", "BERTScore F1 vs the paper's text")]
# (validator component key, x label): the bar groups
STAGES = [("qformer_pubmed", "stage 1"), ("qformer", "stage 2")]
# (validation run, legend label, colour): adjacent bars within each group; gray = original, blue = adopted
MODELS = [("released_excl", "Q-Former", NEUTRAL), ("final", "MLP projector (final)", PALETTE[0])]
BAR_W, BAR_GAP = 0.38, 0.04  # bar width and gap between the two bars of a group (groups are 1 apart)

# variant-study table (--table)
LABELS = {"qformer": "Q-Former (released)", "mlp_s": "MLP-S", "mlp_m": "MLP-M (param-matched)", "mlp_l": "MLP-L", "mlp_xl": "MLP-XL",
          "mlp_m_notoken": "MLP-M, 1 token repeated", "mlp_m_lowlr": "MLP-M, lr 1e-4", "xattn_lite": "Cross-attention lite"}
ADOPTED = "mlp_m"  # projector of scripts/outputs/runs/final (13_qformer_pubmed.summary.json: width 704, depth 2)


def paired_bar_figure(out: Path, validation: Path = VALIDATION, paper: bool = False) -> list[Path]:
    """One panel per metric; in each, two bar groups (stage 1, stage 2) of two bars (Q-Former, MLP)."""
    apply_style(8.0)
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    plt.rcParams["savefig.bbox"] = "standard"  # no tight crop: saved at exactly FIG_LETTER_W (FIG_FULL_W with --paper)

    runs = {name: read_json(validation / name / "metrics.json")["components"] for name, _, _ in MODELS}
    values = {(name, st, key): float(runs[name][st][key]) for name, _, _ in MODELS for st, _ in STAGES for key, _, _ in CORE}
    fig, axes = plt.subplots(1, len(CORE), figsize=(FIG_FULL_W, 2.8) if paper else (FIG_LETTER_W, 2.65), layout="constrained")
    x = np.arange(len(STAGES))
    gap, fs = (0.1, 7) if paper else (BAR_GAP, 8)  # value labels need more room at the paper's width
    offsets = (np.arange(len(MODELS)) - (len(MODELS) - 1) / 2) * (BAR_W + gap)
    for ax, (key, title, ylabel) in zip(axes, CORE):
        top = 0.0
        for (name, _, color), dx in zip(MODELS, offsets):
            ys = [values[(name, st, key)] for st, _ in STAGES]
            ax.bar(x + dx, ys, width=BAR_W, color=color, linewidth=0, zorder=3)
            for xi, y in zip(x + dx, ys):
                ax.text(xi, y, f"{y:.2f}", ha="center", va="bottom", fontsize=fs, color=INK, zorder=4)
            top = max(top, *ys)
        ax.set_ylim(0, top * 1.16)  # headroom for the value labels
        ax.set_xticks(x, [lab for _, lab in STAGES])
        ax.tick_params(axis="x", length=0, labelsize=8, labelcolor=INK)
        ax.grid(axis="x", visible=False)
        ax.set_ylabel(ylabel, fontsize=8)
        ax.set_title(title, fontsize=9, loc="left", fontweight="bold", color=INK)
    fig.legend(handles=[Patch(facecolor=c, label=lab) for _, lab, c in MODELS], loc="outside lower center", ncol=len(MODELS),
               fontsize=8, handlelength=1.2, columnspacing=2.0)
    if paper:
        paths = [PAPER_FIG_DIR / "supp_projector_comparison.png"]
        fig.savefig(paths[0])
    else:
        paths = save_fig(fig, "a5_mlp_projector", out)
    plt.close(fig)
    print(json.dumps({f"{name} | {st} | {key}": round(v, 4) for (name, st, key), v in values.items()}, indent=1))
    return paths


def val(r: dict, key: str):
    v = r["test"]["loss"] if key == "test_loss" else r.get(key)
    return np.nan if v is None else float(v)


def fmt(v, nd=3):
    return "n/a" if v is None or (isinstance(v, float) and not np.isfinite(v)) else f"{v:.{nd}f}"


def table_outputs(out: Path) -> list[Path]:
    """Variant-study table and results summary from ``<out>/metrics.json``."""
    import pandas as pd

    summary = read_json(out / "metrics.json")
    results = summary["results"]
    if "qformer" not in results:
        raise SystemExit(f"no reference evaluation in {out / 'metrics.json'}; run run_analysis.py first")
    adoption = summary.get("adoption") or {}
    per = adoption.get("variants", {})
    smoke = bool(summary.get("smoke"))
    ref = results["qformer"]
    n = ref.get("n", "?")
    order = list(LABELS)
    cands = sorted((v for v in results if v != "qformer"), key=lambda v: (order.index(v) if v in order else len(order), v))
    params = {v: results[v]["params"] / 1e6 for v in results}
    verdict = adoption.get("verdict", "no verdict (reference or candidates missing)")

    rows = []
    for v in ["qformer", *cands]:
        r = results[v]
        arch = r.get("arch", {})
        a = per.get(v, {})
        o7 = "reference" if v == "qformer" else ("pass" if a.get("adopt") else ("fail: " + ", ".join(
            k.replace("validity_", "").replace("_matched", "").replace("frac_", "").replace("_outputs", "") for k in a.get("failed", []))) if a else "n/a")
        rows.append({"model": LABELS.get(v, v), "params (M)": params[v], "W x D": f"{arch.get('width')} x {arch.get('depth')}",
                     "retrieval AUC": val(r, "recall_auc"), "BERTScore F1": val(r, "bertscore_f1_matched"), "MiniLM cos": val(r, "semantic_sim_matched"),
                     "shared cos": val(r, "contrastive_sim_matched"), "degenerate": val(r, "validity_frac_degenerate"),
                     "duplicated": val(r, "validity_frac_duplicated_outputs"), "fabricated": val(r, "validity_fabrication_any"),
                     "test loss": val(r, "test_loss"), "O7": o7})
    table = pd.DataFrame(rows)
    paths = write_table(table, "a5_mlp_projector_table", out,
                        caption=(f"Residual-MLP projectors vs the released PubMed Q-Former (same inputs, training objective/data/schedule, greedy "
                                 f"decoding and validate\\_models metrics; PubMed test set without the leakage-audit papers, n={n}). "
                                 f"O7: pass if within the preset tolerances of the Q-Former. The final model uses {LABELS[ADOPTED]}."),
                        label="tab:a5_mlp_projector")

    L = ["# A5.1 results summary: MLP projector vs Q-Former", "",
         f"Written by `make_figure.py --table` on {time.strftime('%Y-%m-%d %H:%M')} from `{out.name}/metrics.json`.", ""]
    if smoke:
        L += ["> **SMOKE RUN.** The projectors were trained for 2 steps on 48 rows and scored on 8 short generations. The numbers check "
              "that the code paths run. They are not results, and the verdict below is not a decision.", ""]
    L += [f"**O7 verdict:** {verdict}", ""]
    if adoption.get("recommended"):
        L += [f"**Recommended projector:** `{adoption['recommended']}` ({params[adoption['recommended']]:.1f} M parameters): the smallest "
              f"variant that passes every O7 criterion. The final model uses `{ADOPTED}`.", ""]
    elif adoption.get("closest_if_none_pass"):
        c = adoption["closest_if_none_pass"]
        L += [f"No variant passes all criteria. Closest: `{c}`, which fails {', '.join(per[c]['failed']) or 'none'}.", ""]
    L += [f"Brain-to-text on the PubMed test set, n = {n} maps (leakage-audit papers excluded), validator decoding (greedy, 128 tokens). "
          "Tolerances (O7): AUC -0.01; BERTScore / MiniLM / shared cosine -0.02; degenerate +0.01; duplicated and fabricated +0.02.", "",
          "| model | params (M) | W x D | retrieval AUC | BERTScore F1 | MiniLM cos | shared cos | degenerate | duplicated | fabricated | test loss | O7 |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for _, r in table.iterrows():
        L.append(f"| {r['model']} | {r['params (M)']:.1f} | {r['W x D']} | {fmt(r['retrieval AUC'])} | {fmt(r['BERTScore F1'])} | {fmt(r['MiniLM cos'])} | "
                 f"{fmt(r['shared cos'])} | {fmt(r['degenerate'])} | {fmt(r['duplicated'])} | {fmt(r['fabricated'])} | {fmt(r['test loss'])} | {r['O7']} |")
    nulls = [f"{k.split('_')[0]} null {fmt(ref.get(k))}" for k in ("bertscore_f1_null", "semantic_sim_null", "contrastive_sim_null")]
    L += ["", f"Deranged-pair nulls (Q-Former): {', '.join(nulls)}.", ""]
    cc = ref.get("validator_crosscheck")
    if cc and "delta_ours_minus_validator" in cc:
        d = ", ".join(f"{k} {v:+.4f}" for k, v in cc["delta_ours_minus_validator"].items())
        L += ["## Reference cross-check", "",
              f"The reference was re-evaluated through this script and compared with `{cc['validator_metrics']}` (n = {cc.get('validator_n')}). "
              f"Over {cc.get('n_texts_compared')} compared texts, the identical fraction is {fmt(cc.get('identical_text_frac'), 3)} and the "
              f"prefix-consistent fraction is {fmt(cc.get('prefix_match_frac'), 3)}. The latter applies to greedy decoding with a shorter "
              f"token budget. Metric deltas (ours - validator): {d}"
              + ("." if cc.get("comparable") else f". These are **not comparable** (n {cc.get('ours_n')} vs {cc.get('validator_n')})."), ""]
        if smoke:
            L += ["(Smoke: 8 items, 16 new tokens, on CPU in fp32. The validator ran on GPU in bf16 with 128 tokens, so only the "
                  "prefix check is meaningful here.)", ""]
    L += ["## Training and runtime", "", "| variant | best epoch | best val loss | sec/step | train min | device |", "|---|---|---|---|---|---|"]
    for v in cands:
        t = results[v].get("training") or {}
        L.append(f"| {v} | {(t.get('best_epoch') or 0) + 1 if t.get('best_epoch') is not None else 'n/a'} | {fmt(t.get('best_val_loss'), 4)} | "
                 f"{fmt(t.get('sec_per_step_median'), 3)} | {fmt((t.get('train_seconds_total') or 0) / 60, 1)} | {t.get('device', 'n/a')} |")
    n_sp = (ref.get("soft_prompt") or {}).get("shape", ["?"])[0]
    L += ["", f"## Soft-prompt statistics (first {n_sp} test maps, before the rescale to the LM token norm)", "",
          f"LM input embeddings: mean token norm {fmt(ref.get('soft_prompt', {}).get('lm_embedding_norm_mean'), 3)}, element std "
          f"{fmt(ref.get('soft_prompt', {}).get('lm_embedding_elem_std'), 4)}.", "",
          "| model | shape | token norm (mean +- sd) | element std | within-prompt token cosine | across-sample std |", "|---|---|---|---|---|---|"]
    for v in ["qformer", *cands]:
        s = results[v].get("soft_prompt") or {}
        L.append(f"| {LABELS.get(v, v)} | {s.get('shape')} | {fmt(s.get('token_norm_mean'))} +- {fmt(s.get('token_norm_std'))} | {fmt(s.get('elem_std'), 4)} | "
                 f"{fmt(s.get('within_prompt_token_cosine'))} | {fmt(s.get('across_sample_std'), 4)} |")
    L.append("")
    (out / "results_summary.md").write_text("\n".join(L))
    return paths + [out / "results_summary.md"]


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path, default=HERE / "outputs")
    p.add_argument("--validation-dir", type=Path, default=VALIDATION, help="Parent of released_excl/ and final/ (validate_models.py outputs).")
    p.add_argument("--table", action="store_true", help="Write the variant-study table and results summary from <out>/metrics.json.")
    p.add_argument("--paper", action="store_true",
                   help="Draw at the paper's text width (FIG_FULL_W) and write only paper/figures/supp_projector_comparison.png.")
    args = p.parse_args(argv)
    paths = table_outputs(args.out) if args.table else paired_bar_figure(args.out, args.validation_dir, paper=args.paper)
    print("\n".join(str(path) for path in paths))
    return 0


if __name__ == "__main__":
    sys.exit(main())
