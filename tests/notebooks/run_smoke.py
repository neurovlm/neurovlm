"""Execute a jupytext .py:percent notebook mirror as a script, in smoke mode.

Usage: python run_smoke.py <path/to/notebook.py>

Run as a subprocess (not in-process) so each notebook gets a clean interpreter
and one notebook's monkeypatches or crashes can't leak into another's.
"""

from __future__ import annotations

import runpy
import sys
import traceback
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: run_smoke.py <notebook.py>", file=sys.stderr)
        return 2
    target = sys.argv[1]

    sys.path.insert(0, str(REPO_ROOT))
    from tests.notebooks import smoke_bootstrap  # noqa: F401  (applies patches on import)

    try:
        runpy.run_path(target, run_name="__main__")
    except SystemExit as exc:
        return 0 if exc.code in (None, 0) else exc.code
    except BaseException:
        traceback.print_exc()
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
