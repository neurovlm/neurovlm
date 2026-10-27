"""The training pipeline's smoke test: scripts/pipeline/smoke_pipeline.sh, every stage for one batch on the CPU.

Needs the pinned training inputs (``python scripts/data/fetch_hf.py --only pipeline``) and the AAL atlas, and takes
about five minutes, so it carries the ``pipeline_smoke``, ``slow`` and ``requires_data`` markers and is deselected
from the offline suite. .github/workflows/smoke.yml runs it on every pull request that touches the scripts or the
library:

    pytest tests/scripts/test_smoke_pipeline.py -m pipeline_smoke -v

``NEUROVLM_SMOKE_ARGS`` replaces the arguments passed to smoke_pipeline.sh (default: --skip-data --no-validate
--device cpu); ``SMOKE_COVERAGE=1`` runs the stages under coverage (see smoke_pipeline.sh).
"""

from __future__ import annotations

import json
import os
import shlex
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "pipeline" / "smoke_pipeline.sh"
RUN_DIR = REPO_ROOT / "scripts" / "outputs" / "runs" / "smoke"
STAGES = ("08", "09", "20", "11", "12", "13", "14")

pytestmark = [pytest.mark.pipeline_smoke, pytest.mark.slow, pytest.mark.requires_data]


def test_smoke_pipeline_runs_every_stage():
    args = shlex.split(os.environ.get("NEUROVLM_SMOKE_ARGS", "--skip-data --no-validate --device cpu"))
    subprocess.run(["bash", str(SCRIPT), *args], check=True, cwd=REPO_ROOT)

    timings = json.loads((RUN_DIR / "pipeline_timings.json").read_text())
    assert timings["smoke"] is True
    assert {stage: timings["stages"][stage]["exit_code"] for stage in STAGES} == {stage: 0 for stage in STAGES}
    provenance = json.loads((RUN_DIR / "provenance.json").read_text())
    assert provenance["exit_status"] == 0
