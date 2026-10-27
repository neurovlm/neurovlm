#!/usr/bin/env python
"""One-command reproduction of the NeuroVLM revision: locked environment -> data inputs from Hugging Face -> training ->
validation -> paper figures -> comparison with the revision's numbers (scripts/config/reference_metrics.json).
scripts/pipeline/reproduce.sh builds .venv when it is missing and runs this script with it; run from anywhere.

    bash scripts/pipeline/reproduce.sh                    # all five CV folds, ~2.5 h on one RTX 5090 (plus the download)
    bash scripts/pipeline/reproduce.sh --cv-folds 0       # one CV fold, ~2 h
    bash scripts/pipeline/reproduce.sh --smoke            # every stage tiny (~5 min), code path only
    bash scripts/pipeline/reproduce.sh --home DIR         # isolated data dir (an existing ~/.cache/neurovlm stays untouched)

Data: scripts/data/fetch_hf.py downloads every input verbatim from the Hugging Face org neurovlm (and the third-party
models) at the commits pinned in scripts/config/hf_manifest.json and checks each sha256 (~23 GB on first use). A file
already in the data dir that differs from the pinned one stops the run; replace it with fetch_hf.py --overwrite.

Steps are resumable: the fetch skips verified files, model stages skip outputs that exist (delete the run directory to
retrain). Logs and per-step wall times: scripts/outputs/reproduce/<run>/.
"""
from __future__ import annotations

import argparse
import datetime
import difflib
import os
import platform
import re
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PY = sys.executable
OUTPUTS = ROOT / "scripts" / "outputs"
LOCK = ROOT / "scripts" / "config" / "requirements.lock"
ALL_FOLDS = "0,1,2,3,4"
# optional local copies of the inputs fetch_hf.py would otherwise download (checked against the manifest's sha256)
EXTERNAL_FILES = ("synth.parquet", "text_synth_less.parquet", "neurovault_collections.parquet", "neurovault_images.parquet",
                  "neurovault_abstracts.parquet", "neurovault_abstracts_missing.csv", "neurovault_images")


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--external-dir", metavar="DIR", default=os.environ.get("NEUROVLM_EXTERNAL_DIR") or None,
                   help="optional local copies, linked into the data dir when missing there before the fetch (they must match "
                        "the manifest's sha256): synth.parquet, text_synth_less.parquet, the neurovault_* tables and "
                        "neurovault_images/. Default: $NEUROVLM_EXTERNAL_DIR.")
    p.add_argument("--home", metavar="DIR",
                   help="run with HOME=DIR, so the package data dir (~/.cache/neurovlm) is DIR/.cache/neurovlm; the Hugging Face "
                        "cache stays where it was (HF_HOME is pinned before HOME changes).")
    p.add_argument("--data", choices=("released", "scratch"), default="released",
                   help="released (default, the only training input): the pinned HF files. scratch: also rerun data stages 01-07 "
                        "into scripts/outputs/data_regenerated/ for provenance (run_all.py --regenerate); training still uses the "
                        "released inputs.")
    p.add_argument("--run-name", metavar="NAME", default="repro",
                   help="run directory scripts/outputs/runs/NAME, validation NAME, figures scripts/figures/main/NAME (default: "
                        "repro). final: the model the response analyses read, figures in scripts/figures/main/")
    p.add_argument("--cv-folds", metavar="LIST", default=ALL_FOLDS,
                   help="CV folds that stage 20 trains and validate_models.py evaluates (default 0,1,2,3,4; ~9 min per fold on one "
                        "RTX 5090). The comparison takes fold 0 alone (vs fold 0 of runs/final) or all five (vs the 5-fold mean); "
                        "Fig. 5a needs all five, else it draws the single-split model.")
    p.add_argument("--full", action="store_true", help="same as --cv-folds 0,1,2,3,4 (the default)")
    p.add_argument("--smoke", action="store_true",
                   help="scripts/pipeline/smoke_pipeline.sh + figure previews on 64 samples, fold 0 only; FAILs of the comparison "
                        "are expected")
    p.add_argument("--offline", action="store_true",
                   help="HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 (the HF cache must already hold every model and dataset)")
    p.add_argument("--device", metavar="DEV", default="cuda", help="cuda (default), cpu")
    p.add_argument("--seeds-14", metavar="LIST", default="1,2,3,4,5",
                   help="stage-14 seeds; the lowest validation loss is packaged (default 1,2,3,4,5, the paper model's recipe)")
    p.add_argument("--skip-env", action="store_true", help="do not compare the environment with scripts/config/requirements.lock")
    p.add_argument("--skip-data", action="store_true", help="skip the fetch")
    p.add_argument("--skip-train", action="store_true", help="skip training")
    p.add_argument("--skip-validate", action="store_true", help="skip validation")
    p.add_argument("--skip-figures", action="store_true", help="skip the figures")
    args = p.parse_args(argv)
    if args.full:
        args.cv_folds = ALL_FOLDS
    if not re.fullmatch(r"[0-4](,[0-4])*", args.cv_folds):
        p.error("--cv-folds: comma-separated folds 0-4, e.g. 0 or 0,1,2,3,4")
    return args


class Reproduction:
    """The steps, in order: check_environment, configure, link_external, fetch_data, then smoke() or train, validate,
    figures and verify. Each step() runs one command from the repository root, tees its output to <name>.log in the log
    directory and appends its wall time and exit code to timings.tsv."""

    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.run = "smoke" if args.smoke else args.run_name
        self.cv_folds = "0" if args.smoke else args.cv_folds
        self.logdir = OUTPUTS / "reproduce" / self.run
        self.logdir.mkdir(parents=True, exist_ok=True)
        self.t_start = int(time.time())
        self.data_dir = Path()
        self.nilearn_dir = Path()

    def log(self, message: str) -> None:
        line = f"[reproduce {time.strftime('%H:%M:%S')}] {message}"
        print(line, flush=True)
        with (self.logdir / "reproduce.log").open("a") as fh:
            fh.write(line + "\n")

    def step(self, name: str, *cmd) -> int:
        cmd = [str(c) for c in cmd]
        self.log(f"step {name}: {' '.join(shlex.quote(c) for c in cmd)}")
        t0 = time.time()
        with (self.logdir / f"{name}.log").open("w") as fh:
            proc = subprocess.Popen(cmd, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            for line in proc.stdout:
                sys.stdout.write(line)
                sys.stdout.flush()
                fh.write(line)
            rc = proc.wait()
        seconds = int(time.time() - t0)
        with (self.logdir / "timings.tsv").open("a") as fh:
            fh.write(f"{name}\t{seconds}\t{rc}\t{_now()}\n")
        self.log(f"step {name}: exit {rc} after {seconds}s")
        return rc

    # 1. environment: CPython 3.12.12 + the exact package set of scripts/config/requirements.lock (reproduce.sh built it)
    def check_environment(self) -> None:
        if self.args.skip_env:
            return
        if shutil.which("uv") is None:
            self.log("WARNING: uv not found; the environment was not compared with scripts/config/requirements.lock")
            return
        frozen = subprocess.run(["uv", "pip", "freeze", "--python", PY], capture_output=True, text=True, check=True).stdout
        want = [line for line in LOCK.read_text().splitlines() if not line.startswith("#")]
        have = [line for line in frozen.splitlines() if not line.startswith("-e ")]
        diff = list(difflib.unified_diff(want, have, "requirements.lock", "uv pip freeze", lineterm=""))
        (self.logdir / "freeze.diff").write_text("".join(line + "\n" for line in diff))
        if diff:
            self.log(f"WARNING: {PY} differs from scripts/config/requirements.lock (see {self.logdir / 'freeze.diff'}); "
                     "remove .venv to rebuild it")
        else:
            self.log("environment matches scripts/config/requirements.lock")

    # 2. caches and data dir
    def configure(self) -> None:
        args = self.args
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")  # deterministic cuBLAS in every step (common.py's default)
        os.environ.setdefault("PYTHONHASHSEED", "0")  # fixed set/dict order (bert_score sorts texts via a set; ~1e-7 on BERTScore F1)
        os.environ.setdefault("HF_HOME", str(Path.home() / ".cache" / "huggingface"))
        if args.home:
            home = Path(args.home).resolve()
            (home / ".cache" / "neurovlm").mkdir(parents=True, exist_ok=True)
            os.environ.setdefault("NILEARN_DATA", str(home / "nilearn_data"))
            os.environ["HOME"] = str(home)
        if args.offline:
            os.environ["HF_HUB_OFFLINE"] = "1"
            os.environ["HF_DATASETS_OFFLINE"] = "1"
        self.data_dir = Path(_output([PY, "-c", "from neurovlm.data import data_dir; print(data_dir)"], check=True))
        self.nilearn_dir = Path(os.environ.get("NILEARN_DATA", str(Path(os.environ["HOME"]) / "nilearn_data")))
        self.log(f"data dir {self.data_dir} | HF cache {os.environ['HF_HOME']} | nilearn data {self.nilearn_dir} | run {self.run} | "
                 f"data {args.data} | CV folds {self.cv_folds} | device {args.device}")
        uname = platform.uname()
        gpu = _output(["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"])
        versions = _output([PY, "-c", "import sys, torch; print('python', sys.version.split()[0]); print('torch', torch.__version__, "
                                      "'cuda', torch.version.cuda, 'cudnn', torch.backends.cudnn.version(), 'gpu', "
                                      "torch.cuda.get_device_name(0) if torch.cuda.is_available() else '-')"])
        text = (f"date: {_now()}\nhost: {uname.system} {uname.release} {uname.machine}\nuv: {_output(['uv', '--version']) or '-'}\n"
                f"{gpu or 'nvidia-smi: -'}\n{versions or 'python: -'}\n")
        (self.logdir / "environment.txt").write_text(text)
        print(text, end="", flush=True)

    def link_external(self) -> None:
        """Optional local copies; fetch_hf.py checks them against the manifest afterwards."""
        if not self.args.external_dir:
            return
        external = Path(self.args.external_dir).resolve()
        for name in EXTERNAL_FILES:
            target = self.data_dir / name
            if not (target.exists() or target.is_symlink()) and (external / name).exists():
                target.symlink_to(external / name)
                self.log(f"linked {name}")

    # 3. data: every input verbatim from Hugging Face at the commits pinned in scripts/config/hf_manifest.json, sha256 checked
    # (neuro_vectors.pt = embedded_text/pubmed_images.pt, latent_specter2_adhoc.pt, synth.parquet, the NeuroVault scrape, ...).
    # A file in the data dir that differs from the pinned one stops the run: fetch_hf.py --overwrite replaces it.
    # --data scratch also reruns data stages 01-07 into scripts/outputs/data_regenerated/ (provenance; not used for training).
    def fetch_data(self) -> None:
        args = self.args
        if not args.skip_data:
            if self.step("data", PY, "scripts/data/fetch_hf.py", "--data-dir", self.data_dir) != 0:
                sys.exit(f"fetch_hf.py failed: pinned inputs missing or different (see {self.logdir / 'data.log'})")
            if args.data == "scratch":
                flags = ["--regenerate", "--no-fetch"] + (["--smoke"] if args.smoke else [])
                if self.step("data_regenerate", PY, "scripts/data/run_all.py", *flags) != 0:
                    self.log("WARNING: data stage regeneration failed (provenance only)")
        # from here on the Hugging Face libraries read only the HF cache, whose refs fetch_hf.py pinned to the manifest commits
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["HF_DATASETS_OFFLINE"] = "1"

    def prefetch_atlases(self) -> None:
        """Fig. 4's parallel spin-test workers race on the first neuromaps download into an empty HOME: fetch once here."""
        code = subprocess.run([PY, "-c", "from neuromaps.datasets import fetch_fsaverage, fetch_regfusion; "
                                         "fetch_fsaverage(density='3k'); fetch_regfusion('fsaverage')"], cwd=ROOT).returncode
        if code != 0:
            self.log("WARNING: neuromaps atlas download failed; the Fig. 4 spin test needs network access or a populated ~/neuromaps-data")

    # 3-6. smoke: every stage tiny, figure previews (no review step: Fig. 5b's a3 input is not computed for smoke models)
    def smoke(self) -> int:
        device = self.args.device
        if self.step("smoke_pipeline", "bash", "scripts/pipeline/smoke_pipeline.sh", "--device", device) != 0:
            self.log("smoke pipeline exit != 0 (validation FAILs on 64 samples are expected; check for tracebacks)")
        self.step("smoke_validate", PY, "scripts/pipeline/validate_models.py", "--run", "scripts/outputs/runs/smoke", "--name", "smoke_all",
                  "--max-samples", "64", "--folds", self.cv_folds, "--device", device, "--generation-batch", "64")
        if not self.args.skip_figures:
            self.prefetch_atlases()
            self.step("smoke_figures", PY, "scripts/figures/main/make_all.py", "--model-dir", "scripts/outputs/runs/smoke",
                      "--max-samples", "64", "--continue-on-error", "--device", device)
        self.step("verify", PY, "scripts/pipeline/verify_reproduction.py", "--name", "smoke_all",
                  "--figures-dir", "scripts/figures/main/smoke/preview_n64", "--since", self.t_start)
        self.log(f"smoke done in {int(time.time()) - self.t_start}s (metrics of 1-epoch models FAIL by design; tracebacks would mean "
                 "a broken path)")
        return 0

    # 4. training: 08 09 11 12 13 14 (stage 14: one run per seed, best validation loss packaged), then the CV stage 20 on
    # --cv-folds (folds already in the run directory are kept)
    def train(self) -> None:
        if self.args.skip_train:
            return
        common = ["scripts/pipeline/train_pipeline.py", "--run-name", self.run, "--skip-data", "--no-validate", "--device", self.args.device]
        rc = self.step("train_models", PY, *common, "--stages", "models", "--seeds-14", self.args.seeds_14)
        if rc != 0:
            sys.exit(rc)
        rc = self.step("train_cv", PY, *common, "--only", "20", "--stage-args", f"--folds {self.cv_folds}")
        if rc != 0:
            sys.exit(rc)

    # 5. validation (PubMed test with the leakage-audit exclusions) against the released numbers of reference_metrics.json
    def validate(self) -> None:
        if self.args.skip_validate:
            return
        reference = "scripts/outputs/validation/released_reference/metrics.json"
        if not (ROOT / reference).exists():
            subprocess.run([PY, "scripts/pipeline/verify_reproduction.py", "--write-released-reference", reference], cwd=ROOT, check=True)
        if self.step("validate", PY, "scripts/pipeline/validate_models.py", "--run", f"scripts/outputs/runs/{self.run}", "--name", self.run,
                     "--reference", reference, "--folds", self.cv_folds, "--generation-batch", "128", "--device", self.args.device) != 0:
            self.log("validate_models.py reports a FAIL vs the released checkpoints")

    # 6. paper figures from this run. Fig. 5b reads the reviewer analysis a3_neurovault_bounds, which fig5_perf_quant.py
    # requires to be computed with this run's autoencoder and InfoNCE heads (~30 s). core/run_all.py a3_neurovault_bounds
    # runs it for runs/final only and into the committed outputs, so it runs here with run_all.py's flags into the log dir.
    # Without the NeuroVault inputs, or if it fails, Fig. 5 is drawn as a, c, d.
    def figures(self) -> str:
        device = self.args.device
        fig_dir = "scripts/figures/main" if self.run == "final" else f"scripts/figures/main/{self.run}"  # make_all.py draws runs/final next to the scripts
        if self.args.skip_figures:
            return fig_dir
        model_dir = f"scripts/outputs/runs/{self.run}"
        a3_out = self.logdir / "a3_neurovault_bounds"
        self.prefetch_atlases()
        a3_ok = False
        if (self.data_dir / "neurovault_images.parquet").exists() and (self.data_dir / "neurovault_images").is_dir():
            a3_ok = self.step("a3_neurovault_bounds", PY, "scripts/figures/review/core/a3_neurovault_bounds/run_analysis.py",
                              "--model-dir", model_dir, "--out", a3_out, "--device", device, "--force", "--n-jobs", "12", "--qc-count-all") == 0
        else:
            self.log(f"WARNING: NeuroVault inputs missing in {self.data_dir}: no a3 bounds, Fig. 5 without panel b")
        figures_without_5 = _output([PY, "-c", "import sys; sys.path.insert(0, 'scripts/figures/main'); from make_all import FIGURES; "
                                              "print(','.join(k for k, _, _ in FIGURES if k != 'fig5'))"], check=True)
        self.step("figures", PY, "scripts/figures/main/make_all.py", "--model-dir", model_dir, "--only", figures_without_5,
                  "--continue-on-error", "--device", device)
        fig5_ok = a3_ok and self.step("figure5", PY, "scripts/figures/main/fig5_perf_quant.py", "--model-dir", model_dir,
                                      "--bounds-dir", a3_out, "--device", device) == 0
        if not fig5_ok:
            self.step("figure5_acd", PY, "scripts/figures/main/fig5_perf_quant.py", "--model-dir", model_dir, "--panels", "acd", "--device", device)
        return fig_dir

    # 7. comparison with runs/final
    def verify(self, fig_dir: str) -> int:
        rc = self.step("verify", PY, "scripts/pipeline/verify_reproduction.py", "--name", self.run, "--figures-dir", fig_dir, "--since", self.t_start)
        self.log(f"done in {int(time.time()) - self.t_start}s; step times in {self.logdir / 'timings.tsv'}")
        return rc


def _now() -> str:
    return datetime.datetime.now().astimezone().isoformat(timespec="seconds")


def _output(cmd: list[str], *, check: bool = False) -> str | None:
    """stdout of ``cmd`` run from the repository root, None when it is missing or fails (unless ``check``)."""
    try:
        return subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        if check:
            raise
        return None


def main(argv=None) -> int:
    reproduction = Reproduction(parse_args(argv))
    reproduction.check_environment()
    reproduction.configure()
    reproduction.link_external()
    reproduction.fetch_data()
    if reproduction.args.smoke:
        return reproduction.smoke()
    reproduction.train()
    reproduction.validate()
    return reproduction.verify(reproduction.figures())


if __name__ == "__main__":
    sys.exit(main())
