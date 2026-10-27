# NeuroVLM tests

One directory per subsystem:

```text
tests/
├── conftest.py
├── api/          # NeuroVLM client and the structured runtime
├── data/         # data loaders and the atlas-free datasets
├── evaluation/   # evaluation helpers and the model comparison
├── metrics/
├── models/       # model definitions, loading, serialization, losses
├── pipelines/    # run bookkeeping (artifacts, checkpoints, provenance)
├── resources/    # pinned-resource loaders
├── scripts/      # scripts/: helpers and the pipeline smoke test
├── training/     # trainers: helpers and one-epoch runs on tiny synthetic data
└── notebooks/    # exports and smoke-executes docs/**.ipynb
```

Tests that need the released weights or data carry the `network` or `requires_*`
markers; everything else runs offline in seconds with small stand-ins for the
models and corpora.

## Install test dependencies

```bash
pip install -e ".[test]"
```

## Run tests

Run the deterministic offline suite:

```bash
pytest -m "not network and not requires_data and not requires_pretrained and not requires_specter and not slow"
```

Run one subsystem:

```bash
pytest tests/models
```

Run every test, including tests that may download data or pretrained models:

```bash
pytest
```

Generate coverage (the CI command; `.coveragerc` limits the report to the training and inference code, see its
header for the measured and omitted modules):

```bash
pytest -m "not network and not requires_data and not requires_pretrained and not requires_specter and not slow" \
    --cov=neurovlm --cov-config=.coveragerc --cov-report=term-missing
```

The offline suite reaches the inference client, the loaders and SPECTER through small stand-ins: `tests/unit/api/
test_client_offline.py` injects fake models and corpora through the `load_model`, `load_dataset`, `load_latent` and
`load_masker` names of `neurovlm.core.client`; `tests/unit/resources/test_resource_loaders_offline.py` replaces
`_download_from_hf` with a writer of small fixture files; `tests/unit/models/test_specter_offline.py` replaces the
`transformers` and `adapters` factories. Tests that need the released weights or data keep their `requires_*` and
`network` markers.

## Reproduction smoke test

`tests/scripts/test_smoke_pipeline.py` runs the paper's training pipeline at smoke
scale (`scripts/pipeline/smoke_pipeline.sh`: every stage for one batch on CPU, from
the pinned Hugging Face inputs) and checks that every stage exited 0. It needs the
inputs and about five minutes, so it is deselected from the offline suite
(`pipeline_smoke`, `slow`, `requires_data`); CI runs it in
`.github/workflows/smoke.yml` on every pull request that touches `scripts/` or
`src/`. See `scripts/README.md`, "Tests and CI".

```bash
.venv/bin/python scripts/data/fetch_hf.py --only pipeline           # pinned inputs, sha256-checked
pytest tests/scripts/test_smoke_pipeline.py -m pipeline_smoke -v    # ~5 min
```

`tests/notebooks/test_notebooks.py` is the older smoke test of the `docs/` notebooks
(the pre-revision analyses, most of them marked unrunnable); it does not cover
`scripts/`.

## Verify training and notebooks after a refactor

Two commands double-check the parts pytest's fast suite doesn't reach on its own:

```bash
python -m pytest tests/training -v              # every training pipeline, real 1-epoch runs on tiny synthetic CPU data (~15s)
python -m pytest tests/notebooks -v              # every notebook, executed end-to-end against real cached data
```

The training tests are already part of the default `pytest` run;
the first command is a single fast check to point at after a refactor.

The notebook smoke tests are different: they are **not** part of the default `pytest`
collection filter, require network access and the same local
data/model cache the notebooks themselves use (`neurovlm.data.fetch_data`),
and can take significant wall time (some figure-reproduction notebooks
legitimately run for tens of minutes even at smoke scale). Every notebook
under `docs/` is converted with Jupytext to a temporary `.py:percent` script;
`tests/notebooks/test_notebooks.py` runs each export as a subprocess from the
original notebook's directory with `NEUROVLM_SMOKE=1`, which caps training to a
single epoch over a couple of batches (see `tests/notebooks/smoke_bootstrap.py`)
so real data/model code paths get exercised in seconds instead of hours. This
proves the notebook's code still runs after a refactor; it does not validate
numerical quality. `-k` selects a subset: `-k "01_tutorials or 02_data or 03_models"`
(public docs), `-k experimental`, `-k figures`, or one notebook, `-k 00_quickstart`.

To only generate a notebook's Python script, run:

```bash
python tests/notebooks/export_notebook.py docs/01_tutorials/00_quickstart.ipynb
```

Exports go to Git-ignored `docs/generated/notebooks/` by default; `-o` chooses
an explicit output path. The notebooks are the committed source of truth.

Some notebooks are marked `skip` with a documented reason in
`tests/notebooks/test_notebooks.py::KNOWN_UNRUNNABLE` — e.g. they depend on a
manually curated artifact (`corpus.txt`), a pre-scraped raw dataset never
folded into the release bundle, a gated multi-billion-parameter LLM, or are
Colab-only. These aren't refactor regressions; keep the reasons in sync with
`docs/figures/README.md` and `docs/experimental/README.md` if the underlying
data situation changes.

## Markers

- `slow`: tests unsuitable for the quick suite
- `network`: requires an external service
- `requires_data`: requires downloaded datasets
- `requires_pretrained`: requires released model weights
- `requires_specter`: requires a Hugging Face SPECTER model
- `notebook_smoke`: exports and executes a notebook end-to-end at smoke scale
- `pipeline_smoke`: runs the training pipeline at smoke scale (`tests/scripts/test_smoke_pipeline.py`)

New tests should be deterministic, use `tmp_path` for artifacts, mock network
access unless explicitly marked, and test behavior rather than notebook prose
or frozen experiment output.
