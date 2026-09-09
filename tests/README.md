# NeuroVLM tests

The suite is organized by scope and subsystem:

```text
tests/
├── conftest.py
├── unit/
│   ├── api/
│   ├── data/
│   ├── metrics/
│   ├── models/
│   ├── pipelines/
│   └── resources/
├── integration/
│   ├── api/
│   ├── evaluation/
│   └── training/
└── notebooks/          # smoke-executes every docs/**.ipynb via its jupytext mirror
```

Unit tests cover isolated functions, model definitions, loaders with mocked
resources, metrics, and run infrastructure. Integration tests exercise complete
inference, evaluation, and small offline training workflows.

## Install test dependencies

```bash
pip install -e ".[test]"
```

## Run tests

Run the deterministic offline suite:

```bash
pytest -m "not network and not requires_data and not requires_pretrained and not requires_specter and not slow"
```

Run only unit tests:

```bash
pytest tests/unit
```

Run only integration tests:

```bash
pytest tests/integration
```

Run every test, including tests that may download data or pretrained models:

```bash
pytest
```

Generate coverage:

```bash
pytest --cov=neurovlm --cov-report=term-missing
```

## Verify training and notebooks after a refactor

Two convenience scripts double-check the parts pytest's fast suite doesn't
reach on its own:

```bash
scripts/smoke_training.sh     # every training pipeline, real 1-epoch runs on tiny synthetic CPU data (~15s)
scripts/smoke_notebooks.sh    # every notebook, executed end-to-end against real cached data (public|experimental|figures|all)
```

`smoke_training.sh` just runs `tests/integration/training` — already part of
the default `pytest` run, but useful as a single fast command to point at
after a refactor.

`smoke_notebooks.sh` is different: it's **not** part of the default `pytest`
collection filter, requires network access and the same local
data/model cache the notebooks themselves use (`neurovlm.data.fetch_data`),
and can take significant wall time (some figure-reproduction notebooks
legitimately run for tens of minutes even at smoke scale). Every notebook
under `docs/` is paired with a jupytext `.py:percent` mirror (kept in sync via
`jupytext --sync <notebook>.ipynb`); `tests/notebooks/test_notebooks.py` runs
each mirror as a subprocess with `NEUROVLM_SMOKE=1`, which caps training to a
single epoch over a couple of batches (see `tests/notebooks/smoke_bootstrap.py`)
so real data/model code paths get exercised in seconds instead of hours. This
proves the notebook's code still runs after a refactor; it does not validate
numerical quality.

Some notebooks are marked `skip` with a documented reason in
`tests/notebooks/test_notebooks.py::KNOWN_UNRUNNABLE` — e.g. they depend on a
manually curated artifact (`corpus.txt`), a pre-scraped raw dataset never
folded into the release bundle, a gated multi-billion-parameter LLM, or are
Colab-only. These aren't refactor regressions; keep the reasons in sync with
`docs/figures/README.md` and `docs/experimental/README.md` if the underlying
data situation changes.

## Markers

- `unit`: isolated, offline behavior
- `integration`: multi-component behavior
- `slow`: tests unsuitable for the quick suite
- `network`: requires an external service
- `requires_data`: requires downloaded datasets
- `requires_pretrained`: requires released model weights
- `requires_specter`: requires a Hugging Face SPECTER model
- `notebook_smoke`: executes a notebook's jupytext `.py` mirror end-to-end at smoke scale

New tests should be deterministic, use `tmp_path` for artifacts, mock network
access unless explicitly marked, and test behavior rather than notebook prose
or frozen experiment output.
