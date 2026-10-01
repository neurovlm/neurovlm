"""Export a selected notebook to Python for testing without tracked mirrors."""

from __future__ import annotations

import argparse
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def export_notebook(notebook: Path, output: Path | None = None) -> Path:
    """Convert notebook cells to a percent-format script using Jupytext."""
    import jupytext

    notebook = notebook.resolve(strict=True)
    if notebook.suffix != ".ipynb":
        raise ValueError(f"Expected an .ipynb notebook: {notebook}")
    if output is None:
        try:
            relative = notebook.relative_to(REPO_ROOT / "docs")
        except ValueError:
            relative = Path(notebook.name)
        output = REPO_ROOT / "docs/generated/notebooks" / relative.with_suffix(".py")
    output = output.resolve()
    if output == notebook:
        raise ValueError("Output must differ from the input notebook")
    output.parent.mkdir(parents=True, exist_ok=True)
    jupytext.write(jupytext.read(notebook), output, fmt="py:percent")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("notebook", type=Path, help="Path to an .ipynb notebook")
    parser.add_argument("-o", "--output", type=Path, help="Optional output path")
    args = parser.parse_args()
    try:
        output = export_notebook(args.notebook, args.output)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    print(output)


if __name__ == "__main__":
    main()
