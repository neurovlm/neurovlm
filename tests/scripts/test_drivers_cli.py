"""The reproduction driver and the review-analysis queue parse their arguments without running anything."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
REPRODUCE = REPO_ROOT / "scripts" / "pipeline" / "reproduce.py"
RUN_ALL = REPO_ROOT / "scripts" / "figures" / "review" / "core" / "run_all.py"


def test_reproduce_help_lists_the_options():
    out = subprocess.run([sys.executable, str(REPRODUCE), "--help"], capture_output=True, text=True, check=True).stdout
    for option in ("--smoke", "--cv-folds", "--home", "--external-dir", "--data", "--seeds-14", "--skip-figures"):
        assert option in out


def test_reproduce_rejects_bad_folds_and_data_modes():
    bad_folds = subprocess.run([sys.executable, str(REPRODUCE), "--cv-folds", "7"], capture_output=True, text=True)
    assert bad_folds.returncode == 2 and "--cv-folds" in bad_folds.stderr
    bad_data = subprocess.run([sys.executable, str(REPRODUCE), "--data", "other"], capture_output=True, text=True)
    assert bad_data.returncode == 2 and "--data" in bad_data.stderr


def test_run_all_help_lists_the_analyses_and_rejects_unknown_ones():
    out = subprocess.run([sys.executable, str(RUN_ALL), "--help"], capture_output=True, text=True, check=True).stdout
    assert "a3_neurovault_bounds" in out and "a8_edge_benchmark" in out
    unknown = subprocess.run([sys.executable, str(RUN_ALL), "nope"], capture_output=True, text=True)
    assert unknown.returncode == 2 and "nope" in unknown.stderr
