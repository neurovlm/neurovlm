#!/usr/bin/env python
"""Get the data inputs: fetch them verbatim from Hugging Face (default), or regenerate them with stages 01-07.

    .venv/bin/python scripts/data/run_all.py                         # fetch_hf.py: every pinned input into data_dir
    .venv/bin/python scripts/data/run_all.py --inputs pipeline       # training + validation inputs only
    .venv/bin/python scripts/data/run_all.py --regenerate --skip-optional
    .venv/bin/python scripts/data/run_all.py --regenerate --only 06 07 --force --stage-args 06="--corpus corpus.txt"

--from-hf (default) runs scripts/data/fetch_hf.py (scripts/config/hf_manifest.json: pinned revisions, sha256 checked) and
no data stage. --regenerate is the provenance path: it fetches the pinned raw inputs, then runs 01-07 into
--artifact-dir (default scripts/outputs/data_regenerated/, never the data dir, so the released files are not
overwritten). --smoke, --force, --device and --artifact-dir are passed to every stage; 04 and 05 are optional.
"""
from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "pipeline"))  # scripts/pipeline/
from common import DEFAULT_OUT_ROOT, REPO_ROOT, get_logger, record_timing  # noqa: E402

HERE = Path(__file__).resolve().parent
STAGES = [  # (id, script, optional)
    ("01", "01_coordinate.py", False),
    ("02", "02_pubmed.py", False),
    ("03", "03_neurovault.py", False),
    ("04", "04_neurowiki.py", True),
    ("05", "05_cogatlas.py", True),
    ("06", "06_n_grams.py", False),
    ("07", "07_transform_text.py", False),
]
log = get_logger("data_run_all")


def fetch(args) -> int:
    """scripts/data/fetch_hf.py with the pinned manifest."""
    cmd = [sys.executable, str(HERE / "fetch_hf.py"), "--only", args.inputs]
    if args.data_dir:
        cmd += ["--data-dir", str(args.data_dir)]
    if args.overwrite:
        cmd.append("--overwrite")
    if args.check:
        cmd.append("--check")
    log.info("fetch: %s", " ".join(shlex.quote(c) for c in cmd[1:]))
    t0 = time.perf_counter()
    rc = subprocess.run(cmd, cwd=REPO_ROOT).returncode
    record_timing(args.out, stage="fetch_hf", script="fetch_hf.py", status="ok" if rc == 0 else f"failed (rc={rc})",
                  seconds=time.perf_counter() - t0, smoke=False)
    return rc


def regenerate(args) -> int:
    """Stages 01-07 into the artifact dir."""
    extra = {}
    for item in args.stage_args:
        sid, _, val = item.partition("=")
        extra.setdefault(sid, []).extend(shlex.split(val))

    common = ["--artifact-dir", str(args.artifact_dir)] if not args.smoke else []
    if args.smoke:
        common.append("--smoke")
    if args.force:
        common.append("--force")
    if args.device:
        common += ["--device", args.device]

    rows = []
    for sid, script, optional in STAGES:
        if args.only and sid not in args.only:
            continue
        if sid in args.skip or (optional and args.skip_optional):
            rows.append(dict(stage=sid, script=script, status="skipped", seconds=0.0))
            continue
        cmd = [sys.executable, str(HERE / script), *common, *extra.get(sid, [])]
        log.info("start %s: %s", sid, " ".join(shlex.quote(c) for c in cmd[1:]))
        t0 = time.perf_counter()
        rc = subprocess.run(cmd, cwd=REPO_ROOT).returncode
        secs = time.perf_counter() - t0
        status = "ok" if rc == 0 else f"failed (rc={rc})"
        rows.append(dict(stage=sid, script=script, status=status, seconds=secs))
        record_timing(args.out, stage=sid, script=script, status=status, seconds=secs, smoke=args.smoke)
        log.info("%s %s in %.1fs", sid, status, secs)
        if rc != 0 and not args.continue_on_error:
            break

    width = max((len(r["script"]) for r in rows), default=10)
    print("\n" + f"{'stage':<6}{'script':<{width + 2}}{'status':<16}{'seconds':>9}")
    print("-" * (width + 33))
    for r in rows:
        print(f"{r['stage']:<6}{r['script']:<{width + 2}}{r['status']:<16}{r['seconds']:>9.1f}")
    total = sum(r["seconds"] for r in rows)
    print("-" * (width + 33) + f"\n{'total':<{width + 24}}{total:>9.1f}")
    (args.out / ("run_all_smoke.json" if args.smoke else "run_all.json")).write_text(json.dumps(rows, indent=2))
    return 0 if all(r["status"] in ("ok", "skipped") for r in rows) else 1


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--from-hf", action="store_true", help="Fetch the pinned inputs from Hugging Face (default).")
    mode.add_argument("--regenerate", action="store_true", help="Run stages 01-07 (provenance) into --artifact-dir.")
    p.add_argument("--inputs", choices=["pipeline", "figures", "all"], default="all", help="fetch_hf.py --only.")
    p.add_argument("--data-dir", type=Path, default=None, help="fetch_hf.py --data-dir (default: the package data dir).")
    p.add_argument("--overwrite", action="store_true", help="fetch_hf.py --overwrite: replace files whose sha256 differs.")
    p.add_argument("--check", action="store_true", help="fetch_hf.py --check: verify only.")
    p.add_argument("--no-fetch", action="store_true", help="--regenerate without fetching the raw inputs first.")
    p.add_argument("--smoke", action="store_true", help="Passed to every stage (outputs go to <out>/smoke_artifacts).")
    p.add_argument("--force", action="store_true", help="Passed to every stage.")
    p.add_argument("--device", default=None, help="Passed to every stage (auto/cuda/cpu).")
    p.add_argument("--artifact-dir", type=Path, default=DEFAULT_OUT_ROOT / "data_regenerated",
                   help="Where --regenerate writes (default scripts/outputs/data_regenerated).")
    p.add_argument("--only", nargs="+", default=None, metavar="NN", help="Run only these stages (e.g. 01 06).")
    p.add_argument("--skip", nargs="+", default=[], metavar="NN", help="Skip these stages.")
    p.add_argument("--skip-optional", action="store_true", help="Skip 04 and 05.")
    p.add_argument("--stage-args", nargs="*", default=[], metavar='NN="--flag ..."', help="Extra arguments for one stage.")
    p.add_argument("--continue-on-error", action="store_true", help="Keep going when a stage fails.")
    p.add_argument("--out", type=Path, default=DEFAULT_OUT_ROOT / "data_run_all", help="Where the summary JSON goes.")
    args = p.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)

    if not args.regenerate:
        return fetch(args)
    if not args.no_fetch and not args.smoke:
        rc = fetch(args)
        if rc != 0 and not args.continue_on_error:
            return rc
    return regenerate(args)


if __name__ == "__main__":
    sys.exit(main())
