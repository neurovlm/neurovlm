# Docs Development

## Notebooks

Edit and commit the `.ipynb` notebooks. Generate Python scripts on demand with
Jupytext (included in `pip install -e ".[test]"`):

```bash
python scripts/export_notebook.py docs/01_tutorials/00_quickstart.ipynb
```

The script prints the output path under `docs/generated/notebooks/`, which is
ignored by Git. Use `-o /tmp/quickstart.py` to choose another output path.
When running an export, use the original notebook's directory as the working
directory so relative data paths resolve correctly. IPython magics and shell
commands are commented out by Jupytext's Python export; notebooks that rely
on them need an IPython environment or adaptation before script execution.

Smoke tests generate temporary scripts automatically; no committed `.py`
duplicates are needed. Test one notebook with:

```bash
scripts/smoke_notebooks.sh -k 00_quickstart
```

See `../tests/README.md` for test requirements and limitations.

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
