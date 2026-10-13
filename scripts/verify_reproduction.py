#!/usr/bin/env python
"""Compare a reproduction's validation metrics with the revision's numbers (scripts/reference_metrics.json).

    .venv/bin/python scripts/verify_reproduction.py --name repro                       # validation/repro/metrics.json
    .venv/bin/python scripts/verify_reproduction.py --metrics path/to/metrics.json --figures-dir scripts/figures/main/repro

Every stage trains with deterministic kernels (common.setup_torch), so every primary (PASS/FAIL) metric of
validate_models.py must equal the same metric of runs/final: "exact" = |delta| <= --tol (1e-4), else FAIL. This holds on
the same GPU model and software stack (provenance.json of the run); PyTorch does not promise bitwise identity across GPU
architectures or library versions. A one-fold run (stage 20 with --folds 0) is compared with fold 0 of runs/final, a
five-fold run with the 5-fold mean. The `vs released` column is validate_models.py's own verdict (reproduced >= released -
tolerance). A run evaluated on other items than runs/final (another split or data set) is a problem, not a determinism
test. Overall PASS (exit code 0): every component exact and evaluated on the same items as runs/final.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent
FIG_STEMS = ["fig2_network_examples", "fig3_newgen", "fig4_generative_metrics", "fig5_perf_quant", "fig6_autoencoder_qual",
             "figS2_difumo_v_ale", "figS3_recall_comparison", "figS4_clip_ica", "figS4_clip_ica_p2"]  # every page, as drawn


def verdict(delta, tol: float) -> str:
    if not isinstance(delta, (int, float)) or math.isnan(delta):
        return "FAIL"
    return "exact" if abs(delta) <= tol + 1e-12 else "FAIL"


def component_verdicts(rows: list[dict]) -> dict[str, str]:
    out: dict[str, str] = {}
    for r in rows:
        out[r["component"]] = "FAIL" if "FAIL" in (out.get(r["component"]), r["verdict"]) else "exact"
    return out


def same_items(ref: dict, got: dict) -> list[str]:
    """Differences between the evaluated item sets (PubMed test split, CV pool, NeuroVault) of the reference and the run."""
    ex = (got.get("meta") or {}).get("exclusions") or {}
    out = []
    for key, r in ref.get("exclusions", {}).items():
        g = ex.get(key)
        if isinstance(r, dict) and isinstance(g, dict) and (g.get("before"), g.get("kept")) != (r.get("before"), r.get("kept")):
            out.append(f"{key}: {g.get('before')} items ({g.get('kept')} after exclusions) vs {r.get('before')} ({r.get('kept')}) in runs/final")
    return out


def compare(ref: dict, got: dict, tol: float) -> tuple[list[dict], list[str]]:
    rows, problems = [], []
    for msg in same_items(ref, got):
        problems.append("evaluated items differ from runs/final (another split or data set; deltas are not a determinism test): " + msg)
    for comp, spec in ref["components"].items():
        res = got.get("components", {}).get(comp)
        status = (got.get("comparisons", {}).get(comp) or {}).get("status", "-")
        if not isinstance(res, dict) or "error" in res:
            problems.append(f"{comp}: not evaluated ({(res or {}).get('error', 'missing') if isinstance(res, dict) else 'missing'})")
            rows.append(dict(component=comp, stage=spec["stage"], metric="-", reference=None, reproduced=None, delta=None,
                             verdict="FAIL", vs_released=status))
            continue
        targets = {k: v["final"] for k, v in spec["metrics"].items()}
        label = ""
        if comp == "cv":
            n = res.get("n_folds")
            if n == 1 and "0" in {str(k) for k, v in res.get("folds", {}).items() if isinstance(v, dict)}:
                targets, label = dict(spec["fold0"]), " (fold 0)"
            elif n != spec.get("n_folds", 5):
                problems.append(f"cv: {n} folds evaluated; compare one fold (0) or all {spec.get('n_folds', 5)}")
        for key in spec["metrics"]:
            r, t = res.get(key), targets[key]
            delta = (r - t) if isinstance(r, (int, float)) else None
            rows.append(dict(component=comp, stage=spec["stage"], metric=key + label, reference=t, reproduced=r, delta=delta,
                             verdict=verdict(delta, tol), vs_released=status))
    return rows, problems


def figures(fig_dir: Path, since: float | None) -> list[dict]:
    out = []
    for stem in FIG_STEMS:
        files = [fig_dir / f"{stem}.{ext}" for ext in ("png", "svg")]
        have = [f for f in files if f.exists()]
        mtime = max((f.stat().st_mtime for f in have), default=None)
        out.append(dict(stem=stem, files=[f.name for f in have], mtime=mtime,
                        fresh=None if since is None or mtime is None else mtime >= since))
    return out


def fmt(v, nd=4):
    """nd < 0: signed with |nd| decimals."""
    if v is None:
        return "-"
    if not isinstance(v, (int, float)):
        return str(v)
    return f"{v:+.{-nd}f}" if nd < 0 else f"{v:.{nd}f}"


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--name", help="Validation name: reads scripts/outputs/validation/<name>/metrics.json.")
    g.add_argument("--metrics", type=Path, help="A validate_models.py metrics.json.")
    g.add_argument("--write-released-reference", type=Path, metavar="PATH",
                   help="Only write the released column of --reference as a validate_models.py --reference file and exit "
                        "(so a new machine need not re-evaluate the released checkpoints).")
    p.add_argument("--reference", type=Path, default=SCRIPTS_DIR / "reference_metrics.json")
    p.add_argument("--figures-dir", type=Path, default=None, help="make_all.py output dir to list (e.g. scripts/figures/main/<run>).")
    p.add_argument("--since", type=float, default=None, help="Epoch seconds; figures older than this are flagged as not regenerated.")
    p.add_argument("--tol", "--det-tol", dest="tol", type=float, default=1e-4, help="Largest |delta| that counts as exact.")
    p.add_argument("--json-out", type=Path, default=None, help="Write the comparison here (default: next to metrics.json).")
    args = p.parse_args(argv)

    if args.write_released_reference:
        ref = json.loads(args.reference.read_text())
        comps = {c: {k: m["released"] for k, m in spec["metrics"].items()} for c, spec in ref["components"].items()}
        for c in ("text_to_brain", "adapter"):  # scored on calibrated maps, as validate_models.py --map-calibration global
            comps[c]["map_calibration"] = "global"
        out = {"meta": {"name": "released (from reference_metrics.json)", "released": True, "exclusions": {"enabled": True}},
               "components": comps, "sources": {"note": ref["sources"]["released"]}}
        args.write_released_reference.parent.mkdir(parents=True, exist_ok=True)
        args.write_released_reference.write_text(json.dumps(out, indent=1))
        print(f"released reference -> {args.write_released_reference}")
        return 0

    path = args.metrics or SCRIPTS_DIR / "outputs" / "validation" / args.name / "metrics.json"
    if not path.exists():
        sys.exit(f"{path} not found; run validate_models.py --run <run dir> --name <name> first")
    ref, got = json.loads(args.reference.read_text()), json.loads(path.read_text())
    rows, problems = compare(ref, got, args.tol)

    w = max(len(r["metric"]) for r in rows)
    print(f"reproduction: {path}\nreference:    {args.reference} (runs/final)\n")
    print(f"{'component':<15}{'stage':<6}{'metric':<{w + 2}}{'reference':>10}{'reproduced':>11}{'delta':>10}  {'verdict':<9}{'vs released'}")
    print("-" * (w + 75))
    for r in rows:
        print(f"{r['component']:<15}{r['stage']:<6}{r['metric']:<{w + 2}}{fmt(r['reference']):>10}{fmt(r['reproduced']):>11}"
              f"{fmt(r['delta'], -5):>10}  {r['verdict']:<9}{r['vs_released']}")
    verdicts = component_verdicts(rows)
    print("-" * (w + 75))
    print(f"reproduction (|delta| <= {args.tol:g}): " + ", ".join(f"{c} {v}" for c, v in verdicts.items()))
    for msg in problems:
        print(f"  ! {msg}")

    figs = None
    if args.figures_dir is not None:
        figs = figures(args.figures_dir, args.since)
        print(f"\nfigures in {args.figures_dir}:")
        for f in figs:
            when = time.strftime("%Y-%m-%d %H:%M", time.localtime(f["mtime"])) if f["mtime"] else "-"
            flag = "" if f["fresh"] is None else ("  regenerated" if f["fresh"] else "  STALE (older than --since)")
            print(f"  {f['stem']:<28} {', '.join(f['files']) or 'MISSING':<60} {when}{flag}")

    n_exact = sum(v == "exact" for v in verdicts.values())
    ok = n_exact == len(verdicts) and not problems
    print(f"\nreproduction {'PASS' if ok else 'FAIL'}: {n_exact}/{len(verdicts)} components exact"
          + ("" if not problems else f", {len(problems)} problem(s) above"))
    out = args.json_out or path.with_name("verify_reproduction.json")
    out.write_text(json.dumps({"metrics": str(path), "reference": str(args.reference), "pass": ok, "verdicts": verdicts,
                               "rows": rows, "problems": problems, "figures": figs, "tolerance": args.tol}, indent=1))
    print(f"-> {out}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
