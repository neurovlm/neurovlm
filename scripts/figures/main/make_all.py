#!/usr/bin/env python
"""Render the paper figures (Figs. 2-6, S2-S4) from runs/final into scripts/figures/main/ and print a summary.

    .venv/bin/python scripts/figures/main/make_all.py                       # all figures
    .venv/bin/python scripts/figures/main/make_all.py --only fig2,figS4     # subset
    .venv/bin/python scripts/figures/main/make_all.py --released            # released weights -> main/released/
    .venv/bin/python scripts/figures/main/make_all.py --max-samples 64      # preview -> main/preview_n64/
    .venv/bin/python scripts/figures/main/make_all.py --paper               # also copy the PNGs into paper/figures/
    .venv/bin/python scripts/figures/main/make_all.py --model-dir scripts/outputs/runs/<run>   # -> main/<run>/

Flags after -- go to every script, e.g. -- --no-spin.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent

# key, script, output stem
FIGURES = [
    ("fig2", "fig2_network_examples.py", "fig2_network_examples"),
    ("fig3", "fig3_newgen.py", "fig3_newgen"),
    ("fig4", "fig4_generative_metrics.py", "fig4_generative_metrics"),
    ("fig5", "fig5_perf_quant.py", "fig5_perf_quant"),
    ("fig6", "fig6_autoencoder_qual.py", "fig6_autoencoder_qual"),
    ("figS2", "figS2_difumo_v_ale.py", "figS2_difumo_v_ale"),
    ("figS3", "figS3_recall_comparison.py", "figS3_recall_comparison"),
    ("figS4", "figS4_clip_ica.py", "figS4_clip_ica"),
]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--only", default=None, help="Comma-separated subset, e.g. fig2,fig5,figS3.")
    p.add_argument("--released", action="store_true", help="Use the released weights.")
    p.add_argument("--model-dir", type=Path, default=None, help="Run directory (default: scripts/outputs/runs/final).")
    p.add_argument("--out", type=Path, default=None, help="Output folder (default: figures/main/ or a sub-folder).")
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--force", action="store_true")
    p.add_argument("--paper", action="store_true")
    p.add_argument("--no-exclusions", action="store_true", help="Keep the leakage-audit items (comparison only).")
    p.add_argument("--device", default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--continue-on-error", action="store_true", help="Keep going when a script fails.")
    p.add_argument("extra", nargs=argparse.REMAINDER, help="Flags for every script (after --).")
    args = p.parse_args()

    wanted = [f for f in FIGURES if args.only is None or f[0] in {s.strip() for s in args.only.split(",")}]
    if not wanted:
        sys.exit(f"--only matched nothing; choose from {[f[0] for f in FIGURES]}")
    passthrough = []
    for flag, val in (("--max-samples", args.max_samples), ("--device", args.device), ("--seed", args.seed),
                      ("--model-dir", args.model_dir), ("--out", args.out)):
        if val is not None:
            passthrough += [flag, str(val)]
    for flag, on in (("--released", args.released), ("--force", args.force), ("--paper", args.paper),
                     ("--no-exclusions", args.no_exclusions)):
        if on:
            passthrough.append(flag)
    extra = [e for e in args.extra if e != "--"]

    # where the scripts write (as _shared.default_out), for the summary
    out = args.out or HERE
    if args.out is None:
        if args.released:
            out = out / "released"
        elif args.model_dir is not None and args.model_dir.resolve() != (HERE.parents[1] / "outputs" / "runs" / "final").resolve():
            out = out / args.model_dir.name
        if args.max_samples:
            out = out / f"preview_n{args.max_samples}"

    env = dict(os.environ)
    env.setdefault("PYTHONHASHSEED", "0")  # fixed set order (bert_score batches texts via a set; fig4 BERTScore)
    rows = []
    for key, script, stem in wanted:
        cmd = [sys.executable, str(HERE / script), *passthrough, *extra]
        print(f"\n=== {key}: {' '.join(cmd)}", flush=True)
        t0 = time.perf_counter()
        rc = subprocess.call(cmd, env=env)
        dt = time.perf_counter() - t0
        outputs = [pth.name for ext in ("png", "svg")
                   for pth in (out / f"{stem}.{ext}", *sorted(out.glob(f"{stem}_*.{ext}"))) if pth.exists()]
        rows.append((key, stem, "ok" if rc == 0 else f"FAILED ({rc})", dt, outputs))
        if rc != 0 and not args.continue_on_error:
            break

    print("\n" + "=" * 100)
    print(f"{'figure':<7} {'stem':<26} {'status':<12} {'seconds':>8}  outputs")
    print("-" * 100)
    for key, stem, status, dt, outputs in rows:
        print(f"{key:<7} {stem:<26} {status:<12} {dt:>8.1f}  {', '.join(outputs) or '-'}")
    print("=" * 100)
    print(f"output dir: {out}")
    sys.exit(0 if all(r[2] == "ok" for r in rows) else 1)


if __name__ == "__main__":
    main()
