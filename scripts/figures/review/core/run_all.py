#!/usr/bin/env python
"""Rerun every analysis in core/ one after the other (one GPU job at a time), re-plot its figure, then rebuild ../reviewer_0N/.
scripts/figures/review/core/run_all.sh runs this script with .venv.

    cd neurovlm && nohup bash scripts/figures/review/core/run_all.sh > /dev/null 2>&1 &
    bash scripts/figures/review/core/run_all.sh a3_neurovault_bounds a7_ood_confidence   # only these directories
    FIGURES_ONLY=1 bash scripts/figures/review/core/run_all.sh                           # only re-plot (CPU, ~2 min)

Needs the final model in scripts/outputs/runs/final, its validation (read by a2_granularity_tradeoff and a5_mlp_projector)
and the paper figures (read by a7_failure_cases_percentiles): bash scripts/pipeline/reproduce.sh --run-name final.
Logs: core/logs/<dir>.log and core/logs/status.tsv (dir, script, exit code, seconds); a failed step does not stop the queue.
~2.5 h on one RTX 5090; A5_FULL=1 adds the 8-h A5.1 projector study. DEVICE (default cuda) is passed as --device;
DAPT_CORPUS (NeuroQwen3's PubMed pretraining parquet, optional) lets a4_memorization_check count the papers it holds.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
CORE = Path("scripts/figures/review/core")
LOGS = ROOT / CORE / "logs"
PY = sys.executable
FIGURES_ONLY = os.environ.get("FIGURES_ONLY", "0") == "1"
A5_FULL = os.environ.get("A5_FULL", "0") == "1"
MODEL = ["--model-dir", "scripts/outputs/runs/final"]
DEVICE = ["--device", os.environ.get("DEVICE", "cuda")]
DAPT_CORPUS = Path(os.environ.get("DAPT_CORPUS", str(Path.home() / "efficient_ai/project/pubmed/pubmed_neuroimaging.parquet")))
NO_GPU = {"CUDA_VISIBLE_DEVICES": ""}

# (directory, script, flags[, environment overrides]) in queue order; the comments name the reviewer points answered
STEPS = [
    # 2.4 / 1.2 leakage audit (CPU; --write-exclusions scripts/eval_exclusions would also rewrite the exclusion lists)
    ("a4_leakage_audit", "run_analysis.py", ["--force"]),
    ("a4_leakage_audit", "make_figure.py", []),
    # 1.2 / 2.2 / 1.5 counts (CPU; per-stage sizes from the final run's summaries)
    ("a4_counts_table", "run_analysis.py", [*MODEL, "--force"]),
    ("a4_counts_table", "make_figure.py", []),
    # 1.2 / 2.2 data-flow diagram (CPU)
    ("a4_dataflow_diagram", "run_analysis.py", [*MODEL, "--force"]),
    ("a4_dataflow_diagram", "make_figure.py", []),
    # 3.8 table captions (CPU; uses the cached PMC-OA download in outputs/pubget_data)
    ("a4_caption_audit", "run_analysis.py", ["--skip-download", "--n-jobs", "12"]),
    ("a4_caption_audit", "caption_usability.py", []),
    ("a4_caption_audit", "more_bad_examples.py", []),
    ("a4_caption_audit", "make_figure.py", []),
    # 2.m5 memorization: do test captions copy their abstracts more than training captions? (GPU ~5 min, then cached)
    ("a4_memorization_check", "run_analysis.py", [*MODEL, *DEVICE] + (["--dapt-corpus", str(DAPT_CORPUS)] if DAPT_CORPUS.is_file() else [])),
    # 1.1 retrieval metrics and hard negatives
    ("a1_retrieval_metrics_table", "run_analysis.py", [*MODEL, *DEVICE, "--force"]),
    ("a1_retrieval_metrics_table", "make_figure.py", []),
    # 1.1 / 2.5 / 3.2 text-to-brain baselines (also writes the Neurosynth cache the a6_* steps reuse)
    ("a1_text_to_brain_baselines", "run_text_to_brain.py", [*MODEL, *DEVICE]),
    ("a1_text_to_brain_baselines", "make_figure.py", []),
    # 1.1 / 3.3 brain-to-text generation vs retrieval
    ("a1_brain_to_text_baselines", "run_analysis.py", [*MODEL, *DEVICE, "--force"]),
    ("a1_brain_to_text_baselines", "make_figure.py", []),
    # 3.3 granularity / canonical projection (final model)
    ("a2_granularity_tradeoff", "run_analysis.py", [*MODEL, *DEVICE, "--force"]),
    ("a2_granularity_tradeoff", "make_figure.py", []),
    # 1.3 / 2.3 / 2.8 NeuroVault bounds
    ("a3_neurovault_bounds", "run_analysis.py", [*MODEL, *DEVICE, "--force", "--n-jobs", "12", "--qc-count-all"]),
    ("a3_neurovault_bounds", "make_figure.py", []),
    # 3.4 / 3.7 context-free encoder (heads retrained on the final run's latents)
    ("a5_encoder_swap", "run_analysis.py", [*MODEL, *DEVICE, "--force"]),
    ("a5_encoder_swap", "make_figure.py", []),
    # 3.5 loss ablation
    ("a5_loss_ablation", "run_analysis.py", [*MODEL, *DEVICE, "--force"]),
    ("a5_loss_ablation", "make_figure.py", []),
    # 2.1 MLP projector: the figure reads scripts/outputs/validation/; A5_FULL=1 also reruns the A5.1 variant study
    # (8 h, a5_mlp_projector/run_variants.sh, which ends with make_figure.py --table)
    *([("a5_mlp_projector", "run_variants.sh", [])] if A5_FULL else []),
    ("a5_mlp_projector", "make_figure.py", []),
    # 3.5 latent dimensions
    ("a6_latent_interpretation", "run_analysis.py", [*MODEL, *DEVICE, "--force"]),
    ("a6_latent_interpretation", "select_dimensions.py", []),
    ("a6_latent_interpretation", "make_figure.py", []),
    # 3.6 network-, anatomy- and cognition-only text to brain
    ("a6_modality_restricted_t2b", "run_analysis.py", [*MODEL, *DEVICE, "--force"]),
    ("a6_modality_restricted_t2b", "make_figure.py", []),
    # 3.5 perturbation probes
    ("a6_perturbation_probes", "run_analysis.py", [*MODEL, *DEVICE, "--force"]),
    ("a6_perturbation_probes", "make_figure.py", []),
    # 3.6 [NETWORK] / [REGION] / [FUNCTION] correctness (final model)
    ("a6_steering_tokens", "run_correctness.py", [*MODEL, *DEVICE]),
    ("a6_steering_tokens", "plot_correctness.py", []),
    # 2.12 / 3.9 failure cases and example percentiles (reads the paper figures' *_text.json in scripts/figures/main/)
    ("a7_failure_cases_percentiles", "run_analysis.py", [*MODEL, *DEVICE, "--gen-batch", "64"]),
    ("a7_failure_cases_percentiles", "make_figure.py", []),
    # 1.4 OOD detection (CPU)
    ("a7_ood_confidence", "run_analysis.py", [*MODEL], NO_GPU),
    ("a7_ood_confidence", "make_figure.py", []),
    # 1.8 memory and latency (needs an idle GPU)
    ("a8_edge_benchmark", "run_analysis.py", ["--device", "auto", "--require-idle-gpu", "--force"]),
    ("a8_edge_benchmark", "make_figure.py", []),
]
ANALYSES = list(dict.fromkeys(step[0] for step in STEPS))


def step(sub: str, script: str, flags: list[str], env: dict | None = None) -> None:
    """Run one script of core/<sub>/, append its output to logs/<sub>.log, record (dir, script, exit code, seconds)."""
    if FIGURES_ONLY and script not in ("make_figure.py", "plot_correctness.py"):
        return
    log = LOGS / f"{sub}.log"
    with log.open("a") as fh:
        fh.write(f"===== {time.strftime('%F %T')} {script} {' '.join(flags)}\n")
    print(f"[{time.strftime('%H:%M')}] {sub}/{script} {' '.join(flags)}", flush=True)
    runner = ["bash"] if script.endswith(".sh") else [PY]
    t0 = time.time()
    with log.open("a") as fh:
        rc = subprocess.run([*runner, str(CORE / sub / script), *flags], cwd=ROOT, stdout=fh, stderr=subprocess.STDOUT,
                            env={**os.environ, **(env or {})}).returncode
    with (LOGS / "status.tsv").open("a") as fh:
        fh.write(f"{sub}\t{script}\t{rc}\t{int(time.time() - t0)}\n")
    if rc != 0:
        print(f"[{time.strftime('%H:%M')}] {sub}/{script} FAILED (exit {rc}, see {log})", flush=True)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
                                     epilog="analyses, in queue order: " + ", ".join(ANALYSES))
    parser.add_argument("analyses", nargs="*", metavar="DIR", help="only these core/ directories (default: every one)")
    args = parser.parse_args(argv)
    unknown = sorted(set(args.analyses) - set(ANALYSES))
    if unknown:
        parser.error(f"unknown analyses {unknown}")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ.setdefault("PYTHONHASHSEED", "0")  # fixed set order (bert_score batches texts via a set)
    LOGS.mkdir(parents=True, exist_ok=True)
    (LOGS / "status.tsv").touch()
    for sub, script, flags, *env in STEPS:
        if not args.analyses or sub in args.analyses:
            step(sub, script, flags, *env)
    with (LOGS / "build_reviewer_dirs.log").open("a") as fh:
        proc = subprocess.Popen([PY, "scripts/figures/review/build_reviewer_dirs.py"], cwd=ROOT, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True)
        for line in proc.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            fh.write(line)
        proc.wait()
    print(f"[{time.strftime('%H:%M')}] done; status: {LOGS / 'status.tsv'}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
