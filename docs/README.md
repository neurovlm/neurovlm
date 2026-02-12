# Docs Development

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
