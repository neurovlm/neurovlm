#!/usr/bin/env python
"""Retrain the NeuroVLM paper models: data stages, then 08 -> 09 -> 20 -> 11 -> 12 -> 13 -> 14, each followed by
validate_models.py on the components it trained.

    python scripts/pipeline/train_pipeline.py --smoke                      # every stage, tiny, into runs/smoke/
    python scripts/pipeline/train_pipeline.py --run-name paper             # full retrain into runs/paper/
    python scripts/pipeline/train_pipeline.py --only 08,09 --run-name x    # subset (also --from 12 --to 14, --stages cv)
    python scripts/pipeline/train_pipeline.py --only 13,14 --run-name x --skip-data --stage-args "--inputs-from scripts/outputs/runs/final"

Dependencies: 09, 11 <- 08; 20 <- 09; 12 <- 11, 09, 08; 13 <- 09, 08; 14 <- 13, 12, 09, 08.
Artifacts stay in scripts/outputs/runs/<run_name>/ unless --install copies them into neurovlm.data.data_dir.
Timings go to scripts/outputs/pipeline_timings.json and runs/<run_name>/pipeline_timings.json.
Stages run with deterministic kernels (bit-identical reruns on the same GPU and software); --stage-args "--no-deterministic" turns that off.
Every input comes from the Hugging Face revisions pinned in scripts/config/hf_manifest.json. runs/<run_name>/provenance.json records the
manifest sha256, the pinned revisions, torch/CUDA/cuDNN, GPU and driver, the determinism settings, the git commit and the
command line (earlier invocations into the same run dir are kept under "earlier").
"""
from __future__ import annotations

import argparse
import os
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))  # scripts/pipeline/
from common import (CUBLAS_WORKSPACE_CONFIG, DEFAULT_OUT_ROOT, REPO_ROOT, SCRIPTS_DIR, get_logger, provenance, read_json,  # noqa: E402
                    write_json)

log = get_logger("train_pipeline")
PYTHON = sys.executable

# stage id -> (script, group, components validated afterwards, label), in dependency order
STAGES = {
    "08": ("08_autoencoder.py", "models", "ae", "autoencoder"),
    "09": ("09_projection_head.py", "models", "heads,networks,text_to_brain", "projection heads"),
    "20": ("20_pubmed_cv.py", "cv", "cv", "PubMed 5-fold CV"),
    "11": ("11_anatomical_atlases.py", "models", "", "anatomical atlases"),
    "12": ("12_neuroadapter.py", "models", "adapter", "NeuroAdapter"),
    "13": ("13_qformer_pubmed.py", "models", "qformer_pubmed", "stage 1 projector (grounded, PubMed)"),
    "14": ("14_qformer_anat.py", "models", "qformer", "stage 2 projector (canonical)"),
}
ORDER = list(STAGES)
# stage-14 outputs copied from the selected seed run (plus its last-epoch checkpoint)
STAGE14_FILES = ("canonical_semantic_banks.pt", "neuro_qformer_canonical", "history_14_qformer_anat.json", "generations_refs.jsonl",
                 "14_qformer_anat.summary.json")


def run(cmd: list[str], *, dry_run: bool, log_path: Path) -> tuple[int, float]:
    """Run ``cmd`` from the repo root and tee its output to ``log_path``; returns (exit code, seconds)."""
    log.info("$ %s", " ".join(shlex.quote(c) for c in cmd))
    if dry_run:
        return 0, 0.0
    t0 = time.perf_counter()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w") as fh:
        env = {**os.environ, "CUBLAS_WORKSPACE_CONFIG": os.environ.get("CUBLAS_WORKSPACE_CONFIG", CUBLAS_WORKSPACE_CONFIG)}
        env.setdefault("PYTHONHASHSEED", "0")  # fixed set order (bert_score batches texts via a set)
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, cwd=str(REPO_ROOT), env=env)
        for line in proc.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            fh.write(line)
            fh.flush()
        code = proc.wait()
    return code, time.perf_counter() - t0


def write_provenance(run_dir: Path, record: dict) -> None:
    """runs/<run>/provenance.json; a record of an earlier invocation into the same run dir moves to "earlier"."""
    path = run_dir / "provenance.json"
    old = read_json(path) if path.exists() else {}
    earlier = old.pop("earlier", [])
    if old and old.get("ts") != record.get("ts"):
        earlier.append(old)
    write_json(path, {**record, **({"earlier": earlier} if earlier else {})})


def select_stage14_seed(run_dir: Path, sweep: dict) -> dict:
    """Copy the seed run with the lowest final-epoch validation loss into ``run_dir`` (old files are moved aside)."""
    ok = {s: r for s, r in sweep.items() if r["exit_code"] == 0 and r.get("val_loss_final") is not None}
    if not ok:
        raise RuntimeError(f"no stage-14 seed run finished: {sweep}")
    best = min(ok, key=lambda s: (ok[s]["val_loss_final"], s))
    src = Path(ok[best]["run_dir"])
    epochs = read_json(src / "14_qformer_anat.summary.json")["epochs"]
    names = (f"qformer_canonical_bottleneck_epoch-{epochs - 1}.pt",) + STAGE14_FILES
    aside = run_dir / f"stage14_replaced_{time.strftime('%Y%m%d-%H%M%S')}"
    for name in names:
        dst = run_dir / name
        if dst.exists() or dst.is_symlink():
            aside.mkdir(exist_ok=True)
            shutil.move(str(dst), str(aside / name))
    for name in names:
        s = src / name
        (shutil.copytree if s.is_dir() else shutil.copy2)(s, run_dir / name)
    record = {"rule": "lowest final-epoch stage-14 validation loss (the packaged checkpoint's), same validation split for every "
                      "seed; test metrics and generated titles are not used", "selected_seed": best, "selected_run": str(src),
              "packaged_files": list(names), "replaced_files_moved_to": str(aside) if aside.exists() else None,
              "seeds": sweep, "ts": time.strftime("%Y-%m-%dT%H:%M:%S")}
    write_json(run_dir / "stage14_seed_selection.json", record)
    log.info("stage 14 seed selection: seed %d (val loss %.4f) of %s -> %s", best, ok[best]["val_loss_final"],
             {s: round(r["val_loss_final"], 4) for s, r in ok.items()}, run_dir)
    return record


def sweep_seeds_14(cmd: list[str], seeds: list[int], run_name: str, run_dir: Path, dry_run: bool) -> tuple[int, float, dict]:
    """Train stage 14 once per seed into runs/<run_name>_s<seed>/ (inputs from ``run_dir``), then package the best seed."""
    sweep, secs, code = {}, 0.0, 0
    for seed in seeds:
        srun = f"{run_name}_s{seed}"
        srun_dir = DEFAULT_OUT_ROOT / "runs" / srun
        scmd = list(cmd)
        scmd[scmd.index("--run-name") + 1] = srun
        scmd += ["--seed", str(seed), "--inputs-from", str(run_dir)]  # last, so it wins over a --stage-args --inputs-from
        scode, ssecs = run(scmd, dry_run=dry_run, log_path=run_dir / "logs" / f"14_s{seed}.log")
        summ = srun_dir / "14_qformer_anat.summary.json"
        val = read_json(summ).get("val_loss_final") if (scode == 0 and summ.exists() and not dry_run) else None
        sweep[seed] = {"run_dir": str(srun_dir), "exit_code": scode, "seconds": ssecs, "val_loss_final": val}
        secs += ssecs
        log.info("stage 14 seed %d: exit %d, %.1f s, final val loss %s", seed, scode, ssecs, val)
    if not dry_run:
        try:
            sel = select_stage14_seed(run_dir, sweep)
        except RuntimeError as exc:
            log.error("%s", exc)
            code = 1
        else:
            sweep = {"selected_seed": sel["selected_seed"], "runs": sweep}
    return code, secs, sweep


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--stages", default="models,cv", help="Comma list of groups: models (08,09,11,12,13,14), cv (20).")
    p.add_argument("--only", help="Comma list of stage ids (e.g. 08,09); overrides --stages/--from/--to.")
    p.add_argument("--from", dest="from_", help="First stage id.")
    p.add_argument("--to", help="Last stage id.")
    p.add_argument("--smoke", action="store_true", help="Tiny settings for every stage; run dir runs/smoke; validation on 64 samples.")
    p.add_argument("--force", action="store_true", help="Recompute even if cached.")
    p.add_argument("--dry-run", action="store_true", help="Print the commands without running them.")
    p.add_argument("--run-name", help="Run directory name (default: timestamp, or 'smoke' with --smoke).")
    p.add_argument("--no-validate", dest="validate", action="store_false", help="Skip validate_models.py after each stage.")
    p.add_argument("--install", action="store_true", help="Copy the artifacts into neurovlm.data.data_dir (notebook filenames).")
    p.add_argument("--skip-data", action="store_true", help="Skip scripts/data/run_all.py.")
    p.add_argument("--stage-args", default="", help="Extra arguments for every stage (quoted string).")
    p.add_argument("--validate-args", default="", help="Extra arguments for validate_models.py (quoted string).")
    p.add_argument("--projector", choices=["mlp", "qformer", "auto"], default="mlp",
                   help="Brain-to-text projector for 13 and 14: mlp (paper model), qformer (the notebook's Q-Former), "
                        "auto (13: mlp, 14: whatever 13 packaged). Artifact names keep the qformer prefix either way.")
    p.add_argument("--mlp-width", type=int, help="Passed to stage 13 (its default: 704).")
    p.add_argument("--mlp-depth", type=int, help="Passed to stage 13 (its default: 2).")
    p.add_argument("--early-stop-patience", type=int, help="Passed to stage 13 (its default: 3 for mlp, 0 = off for qformer).")
    p.add_argument("--min-epochs", type=int, help="Passed to stage 13 (its default: 5).")
    p.add_argument("--text-head", choices=["v2", "v1"], help="Passed to stage 09 (v2 = its default, v1 = the notebook's BCE head).")
    p.add_argument("--seeds-14", help="Comma list of stage-14 seeds; the seed with the lowest final validation loss is "
                                      "packaged into runs/<run_name>/ (see stage14_seed_selection.json).")
    p.add_argument("--device", default="auto")
    args = p.parse_args(argv)

    if args.only:
        selected = [s.strip() for s in args.only.split(",") if s.strip()]
    else:
        groups = {g.strip() for g in args.stages.split(",")}
        selected = [s for s in ORDER if STAGES[s][1] in groups]
        if args.from_:
            selected = [s for s in selected if ORDER.index(s) >= ORDER.index(args.from_)]
        if args.to:
            selected = [s for s in selected if ORDER.index(s) <= ORDER.index(args.to)]
    unknown = [s for s in selected if s not in STAGES]
    if unknown:
        p.error(f"unknown stage ids {unknown}; known: {list(STAGES)}")
    run_name = args.run_name or ("smoke" if args.smoke else time.strftime("%Y%m%d-%H%M%S"))
    run_dir = DEFAULT_OUT_ROOT / "runs" / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    timings = {"run_name": run_name, "smoke": args.smoke, "projector": args.projector, "stages": {}, "ts": time.strftime("%Y-%m-%dT%H:%M:%S")}
    extra_stage, extra_val = shlex.split(args.stage_args), shlex.split(args.validate_args)
    flags = (["--smoke"] if args.smoke else []) + (["--force"] if args.force else [])
    log.info("run directory: %s | stages: %s", run_dir, selected)
    prov = {**provenance(deterministic="--no-deterministic" not in extra_stage), "run_name": run_name, "stages": selected}
    if not args.dry_run:
        write_provenance(run_dir, prov)
        log.info("provenance -> %s (hf_manifest.json sha256 %s)", run_dir / "provenance.json", prov["hf_manifest"]["sha256"][:12])

    def save_timings():
        if not args.dry_run:
            write_json(DEFAULT_OUT_ROOT / "pipeline_timings.json", timings)
            write_json(run_dir / "pipeline_timings.json", timings)

    if not args.skip_data:
        cmd = [PYTHON, str(SCRIPTS_DIR / "data" / "run_all.py")] + flags
        code, secs = run(cmd, dry_run=args.dry_run, log_path=run_dir / "logs" / "data.log")
        timings["stages"]["data"] = {"seconds": secs, "exit_code": code, "command": " ".join(cmd)}
        if code != 0:
            log.error("data stages failed (exit %d); continuing with existing artifacts", code)

    failed = False
    for sid in selected:
        script, _, components, label = STAGES[sid]
        cmd = [PYTHON, str(SCRIPTS_DIR / "models" / script), "--run-name", run_name, "--device", args.device] + flags
        if args.install:
            cmd.append("--install")
        if sid == "14" or (sid == "13" and args.projector != "auto"):
            cmd += ["--projector", args.projector]
        if sid == "09" and args.text_head:
            cmd += ["--text-head", args.text_head]
        if sid == "13":
            for flag, val in (("--mlp-width", args.mlp_width), ("--mlp-depth", args.mlp_depth),
                              ("--early-stop-patience", args.early_stop_patience), ("--min-epochs", args.min_epochs)):
                if val is not None:
                    cmd += [flag, str(val)]
        cmd += extra_stage
        log.info("stage %s: %s", sid, label)
        if sid == "14" and args.seeds_14:
            seeds = [int(s) for s in args.seeds_14.split(",") if s.strip()]
            code, secs, sweep = sweep_seeds_14(cmd, seeds, run_name, run_dir, args.dry_run)
            entry = {"script": script, "label": label, "seconds": secs, "exit_code": code, "seeds_14": sweep}
        else:
            code, secs = run(cmd, dry_run=args.dry_run, log_path=run_dir / "logs" / f"{sid}.log")
            entry = {"script": script, "label": label, "seconds": secs, "exit_code": code}
        log.info("stage %s (%s) finished in %.1f s (exit %d)", sid, label, secs, code)
        if code != 0:
            log.error("stage %s failed with exit code %d", sid, code)
            failed = True
        elif args.validate and components:
            vcmd = [PYTHON, str(SCRIPTS_DIR / "pipeline" / "validate_models.py"), "--run", str(run_dir), "--components", components, "--device", args.device, "--name", run_name]
            if args.smoke:
                vcmd += ["--max-samples", "64", "--folds", "0"]
            vcode, vsecs = run(vcmd + extra_val, dry_run=args.dry_run, log_path=run_dir / "logs" / f"{sid}.validate.log")
            entry["validation_seconds"], entry["validation_exit_code"] = vsecs, vcode
        timings["stages"][sid] = entry
        save_timings()
        if failed:
            break

    total = sum(v.get("seconds", 0.0) + v.get("validation_seconds", 0.0) for v in timings["stages"].values())
    timings["total_seconds"] = total
    save_timings()
    log.info("%-6s %-38s %-26s %10s %10s %6s", "stage", "what", "script", "train_s", "valid_s", "exit")
    for sid, v in timings["stages"].items():
        log.info("%-6s %-38s %-26s %10.1f %10.1f %6s", sid, v.get("label", "data stages"), v.get("script", "run_all"), v.get("seconds", 0.0),
                 v.get("validation_seconds", 0.0), f"{v.get('exit_code')}/{v.get('validation_exit_code', '-')}")
    log.info("total %.1fs -> %s", total, DEFAULT_OUT_ROOT / "pipeline_timings.json")
    if not args.dry_run:
        write_provenance(run_dir, {**prov, "finished": time.strftime("%Y-%m-%dT%H:%M:%S"), "exit_status": 1 if failed else 0,
                                   "fetch_hf_exit_status": timings["stages"].get("data", {}).get("exit_code"),  # data step = fetch_hf.py
                                   "stage_exit_codes": {sid: v.get("exit_code") for sid, v in timings["stages"].items() if sid != "data"}})
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
