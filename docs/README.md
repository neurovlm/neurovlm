# Docs Development

## Notebooks

Every notebook under `docs/` is paired with a jupytext `.py:percent` mirror
(e.g. `01_tutorials/00_quickstart.ipynb` <-> `01_tutorials/00_quickstart.py`).
The `.py` mirror is the diff-friendly, reviewable source; edit either file and
run `jupytext --sync <path-to-either-file>` to resync the other. Both files
are committed. See `../tests/README.md` for how these mirrors get smoke-tested
(`scripts/smoke_notebooks.sh`).

## Build

Build locally with `uv`:

```bash
uv venv docs/.venv
uv pip install -p docs/.venv/bin/python -r docs/requirements.txt
docs/.venv/bin/sphinx-build -b html docs docs/_build/html
```

Output will be in `docs/_build/html`.

Clean and rebuild:

```bash
docs/.venv/bin/sphinx-build -M clean docs docs/_build
docs/.venv/bin/sphinx-build -b html -E -a docs docs/_build/html
```

If docs dependencies got into a bad state, recreate the docs environment:

```bash
rm -rf docs/.venv docs/_build docs/generated
UV_CACHE_DIR=.uv-cache uv venv docs/.venv
UV_CACHE_DIR=.uv-cache uv pip install -p docs/.venv/bin/python -r docs/requirements.txt
docs/.venv/bin/sphinx-build -b html -E -a docs docs/_build/html
```
