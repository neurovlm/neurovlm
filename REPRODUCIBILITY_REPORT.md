# Reproducibility & Smoke-Test Report

**Branch:** `tests-reproducibility-training`
**Purpose:** verify that the codebase refactor didn't break training or any notebook, and build lasting infrastructure to re-check this after future refactors.

## TL;DR

- Deterministic test suite: **369 passed**, 0 regressions (up from 368 — added a regression test for a real bug found below).
- All 46 notebooks under `docs/` accounted for: **25 pass**, **19 documented skip** (genuinely missing data/models, not bugs), **2 fail** (a local machine SSL issue, not a code bug — see below).
- **6 real bugs found and fixed**, most notably a checkpoint-resume bug affecting all 5 training pipelines.
- New, reusable infrastructure: every notebook now has a jupytext `.py` mirror, plus a smoke-test harness and two convenience scripts, so this whole check can be re-run in one command after the next refactor.
- Re-verified from a clean run after this report was first drafted — **results are stable and reproducible** (see "Re-verification" below).

---

## What was built

### 1. jupytext mirrors for every notebook

Every `docs/**/*.ipynb` is now paired with a `.py:percent` mirror (e.g. `00_quickstart.ipynb` ↔ `00_quickstart.py`). The `.py` file is a plain, diff-friendly Python script — same code, markdown cells as comments — kept in sync via:

```bash
jupytext --sync path/to/notebook.ipynb   # or the .py file, either direction
```

Both files are committed. Edit either one, then sync before committing.

### 2. Notebook smoke-test harness (`tests/notebooks/`)

- `smoke_bootstrap.py` — when `NEUROVLM_SMOKE=1`, monkeypatches all 8 `train_*` entry points to cap `epochs=1`, and patches `torch.utils.data.DataLoader.__iter__` to yield at most 2 batches. This lets a notebook's _real_ training code run against _real_ cached data in seconds instead of hours, without touching the notebook's own source.
- `run_smoke.py` — executes one `.py` mirror as a script inside that patched environment.
- `test_notebooks.py` — a pytest module that discovers every notebook, runs each as an isolated subprocess (own interpreter, `MPLBACKEND=Agg`, cwd set to the notebook's own directory to match Jupyter's convention), and reports pass/fail with the tail of stdout/stderr on failure. Notebooks that structurally cannot run in this checkout are marked `skip` with a documented reason in `KNOWN_UNRUNNABLE` (see below) rather than silently excluded.

This is **not** part of the default `pytest` run — it needs network access and the same local data cache the notebooks themselves use, and can take significant wall time. Run it explicitly:

```bash
pytest tests/notebooks -v                       # everything
scripts/smoke_notebooks.sh                       # everything (convenience wrapper)
scripts/smoke_notebooks.sh public                # 01_tutorials + 02_data + 03_models only
scripts/smoke_notebooks.sh experimental          # docs/experimental only
scripts/smoke_notebooks.sh figures               # docs/figures only
```

### 3. Training smoke script

```bash
scripts/smoke_training.sh
```

Runs `tests/integration/training` — real 1-epoch training runs on tiny synthetic CPU data for every pipeline (MLP autoencoder/contrastive/text-to-brain, CNN autoencoder/contrastive/text-to-brain, brain-to-text generation), including checkpoint save/resume/reload round-trips. This suite already existed and is part of the default `pytest` run; the script is just a fast, single-purpose entry point. ~15 seconds.

---

## Real bugs found and fixed

These were found by actually executing the code, not by inspection — each one reproduced a genuine crash before the fix.

### 1. Checkpoint-resume path doubling (systemic — all 5 training pipelines)

**File:** `src/neurovlm/pipelines/checkpoints.py`, `CheckpointManager._resolve_resume_path`

Every training module (`autoencoder.py`, `contrastive.py`, `text_to_brain.py`, `brain_to_text.py`, `mlp.py`) reloads the best checkpoint after training via:

```python
manager.load_resume(run.run_dir / "checkpoints" / "best.pt", ...)
```

That path is already correct relative to the working directory. But `_resolve_resume_path` assumed _any_ relative path was a bare filename needing re-anchoring under `run_dir`, so it re-prepended `run_dir` a second time — producing paths like `runs/run-001/runs/run-001/checkpoints/best.pt`. This only manifests when `output_root` is itself a relative path (exactly what `docs/03_models/08_autoencoder.ipynb` uses); every existing test used an absolute `tmp_path`, which is why this had never been caught.

**Fix:** treat a relative path as already-correct if it exists as given, before falling back to re-anchoring:

```python
if not path.is_absolute() and not path.exists():
    ...
```

**Regression test added:** `tests/unit/pipelines/test_infrastructure.py::test_checkpoint_load_resume_accepts_fully_qualified_relative_run_dir_path` — verified it fails on the pre-fix code and passes after.

### 2. Notebooks bypassing public data loaders

Several notebooks read raw files directly out of `data_dir` instead of using the package's public loaders — but `fetch_data()` / the HF-backed loaders cache into the HuggingFace hub cache, not `data_dir`, so those raw paths never exist on a fresh checkout:

| Notebook                                                                                                                                                           | Was reading                                             | Fixed to                             |
| ------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------- | ------------------------------------ |
| `03_models/08_autoencoder.ipynb`                                                                                                                                   | `data_dir/"coordinates.parquet"`                        | `load_dataset("pubmed_coordinates")` |
| `figures/20_pubmed_cv.ipynb`                                                                                                                                       | `data_dir/"coordinates.parquet"`                        | `load_dataset("pubmed_coordinates")` |
| `03_models/08_autoencoder.ipynb`, `03_models/09_projection_head.ipynb` (×2)                                                                                        | manually rebuilding a masker from `data_dir/"mask.npz"` | `load_masker()`                      |
| `experimental/evaluation/15_qualatative_networks.ipynb`, `figures/13_network_labeling.ipynb`, `figures/16_qualatative_auto.ipynb`, `figures/19_ica_networks.ipynb` | `gzip.open(data_dir/"networks_arrays.pkl.gz")`          | `load_dataset("networks")`           |

### 3. `torch.load` missing `weights_only=False` under PyTorch 2.6+

**File:** `docs/03_models/08_autoencoder.ipynb`

`torch.load(data_dir / "autoencoder.pt")` loads a full pickled `nn.Module`, not a state dict. PyTorch 2.6 flipped the default `weights_only` to `True`, so this now raises `UnpicklingError`. Every other `torch.load` call in the same notebook already passed `weights_only=False`; this one was just missed. Fixed for consistency.

### 4. Hardcoded `device="cuda"` (crashes on any non-CUDA machine)

Five call sites across `01_tutorials/00_quickstart.ipynb`, `figures/24_brain_to_text_pubmed.ipynb`, and `figures/19_ica_networks.ipynb` (8 individual `.to("cuda")`/`device="cuda"` occurrences total) hardcoded CUDA even though sibling cells in the _same_ notebooks already used the portable pattern. Standardized all of them to:

```python
device = "cuda" if torch.cuda.is_available() else "cpu"
```

### 5. `display()` used without importing it

Six notebooks called `display(...)` (a name Jupyter/IPython auto-injects into the kernel namespace) without importing it — works fine in a real notebook, `NameError` in any plain-script execution:

`experimental/data_preparation/extract_neuroscience_terms_from_text.ipynb`, `extract_neuroscience_terms_colab_hf.ipynb`, `prepare_network_gold_terms_for_llm_corpus.ipynb`, `create_networks_test_set_csv.ipynb`, `figures/24_brain_to_text_pubmed.ipynb`, `03_models/13_qformer_pubmed.ipynb`.

Fixed by adding `from IPython.display import display` — harmless in a real kernel (it's already available), and makes the notebook portable to script execution.

### 6. Empty-DataFrame column crash

**File:** `experimental/data_preparation/prepare_network_gold_terms_for_llm_corpus.ipynb`

When there are zero "missing" network gold terms (the actual, correct state on a fully up-to-date checkout), `pd.DataFrame([])` produces a DataFrame with **zero columns**, and a later `added_terms_df["normalized_term"]` lookup raised `KeyError`. Fixed by declaring `columns=[...]` explicitly on construction so the empty case still has the right schema.

### Smoke-scale hooks added (not bugs, but needed for testability)

- `02_data/04_neurowiki.ipynb` — embeds an entire wiki corpus in a plain Python loop (no `DataLoader` to cap). Added `if os.environ.get("NEUROVLM_SMOKE"): df = df.head(32)`.
- `experimental/data_preparation/extract_neuroscience_terms_from_text.ipynb` — already had a `MAX_DOCS` scale knob for exactly this purpose, just left at `None` (full corpus). Wired it to `2 if os.environ.get("NEUROVLM_SMOKE") else None`.

Both only change behavior when `NEUROVLM_SMOKE=1` is set; real/manual runs are untouched.

---

## Full notebook results

### `docs/01_tutorials/` — 7/7 pass

All pass: `00_quickstart`, `01_introduction`, `02_contrastive`, `03_generative_text-to-brain`, `04_generative_brain-to-text`, `05_custom_corpus`, `06_atlas_free_cnn`.

### `docs/02_data/` — 4 pass, 3 skip

| Notebook            | Result   | Reason                                                                          |
| ------------------- | -------- | ------------------------------------------------------------------------------- |
| `01_coordinate`     | pass     |                                                                                 |
| `02_pubmed`         | pass     |                                                                                 |
| `03_neurovault`     | **skip** | needs pre-scraped raw Neurovault artifacts never folded into the release bundle |
| `04_neurowiki`      | pass     | (smoke-scale hook added)                                                        |
| `05_cogatlas`       | pass     |                                                                                 |
| `06_n_grams`        | **skip** | needs a manually curated `corpus.txt` (human-in-the-loop step, not regenerable) |
| `07_transform_text` | **skip** | needs `meta-llama/Meta-Llama-3.1-8B-Instruct` (gated, 8B params)                |

### `docs/03_models/` — 2 pass, 3 skip, 2 fail (environment, not code)

| Notebook                | Result   | Reason                                                                |
| ----------------------- | -------- | --------------------------------------------------------------------- |
| `08_autoencoder`        | pass     | (3 bugs fixed here — see above)                                       |
| `09_projection_head`    | pass     | (2 bugs fixed here)                                                   |
| `10_n_grams`            | **skip** | needs `ngram_emb.pt` from the skipped `06_n_grams`                    |
| `11_anatomical_atlases` | **fail** | local SSL trust-store gap — see "The 2 SSL failures" below            |
| `12_neuroadapter`       | **fail** | blocked transitively by `11`'s failure (bug found and fixed here too) |
| `13_qformer_pubmed`     | **skip** | needs output of the skipped `07_transform_text`                       |
| `14_qformer_anat`       | **skip** | needs `text_synth_less.parquet`, no producer anywhere in this repo    |

### `docs/experimental/` — 10 pass, 7 skip

| Notebook                                                     | Result   | Reason                                                                      |
| ------------------------------------------------------------ | -------- | --------------------------------------------------------------------------- |
| `cnn/evaluation/autoencoder_comparison`                      | pass     |                                                                             |
| `cnn/evaluation/contrastive_comparison`                      | pass     |                                                                             |
| `cnn/evaluation/text_to_brain_comparison`                    | pass     |                                                                             |
| `cnn/training/architecture_background`                       | pass     |                                                                             |
| `cnn/training/autoencoder`                                   | pass     |                                                                             |
| `cnn/training/contrastive_and_text_to_brain`                 | pass     |                                                                             |
| `cnn/training/contrastive_pubmed`                            | pass     |                                                                             |
| `data_preparation/create_networks_test_set_csv`              | pass     | (display() fix)                                                             |
| `data_preparation/extract_neuroscience_terms_colab_hf`       | **skip** | Colab-only: shell magics, Drive mount, needs A100                           |
| `data_preparation/extract_neuroscience_terms_from_text`      | pass     | (display() fix + smoke hook; uses local Ollama)                             |
| `data_preparation/mesh`                                      | **skip** | reads a path outside this repo's tracked tree                               |
| `data_preparation/prepare_network_gold_terms_for_llm_corpus` | pass     | (display() fix + empty-DataFrame bug fixed)                                 |
| `evaluation/14_llm_concepts`                                 | **skip** | missing upstream artifact + gated 8B LLM; also hung >2h with zero output    |
| `evaluation/15_qualatative_networks`                         | **skip** | needs `networks_emb.pt`, only produced inside the skipped `14_llm_concepts` |
| `evaluation/17_qualatative_generative`                       | **skip** | needs `neurovault.pt` from the skipped `02_data/03_neurovault`              |
| `evaluation/18_quant_gen_a`                                  | **skip** | needs `neuro_vault_brain_to_text.csv`, no producer anywhere                 |
| `evaluation/18_quant_gen_b`                                  | **skip** | documented as needing an unpublished baseline                               |

### `docs/figures/` — 2 pass, 6 skip

| Notebook                  | Result   | Reason                                                                                           |
| ------------------------- | -------- | ------------------------------------------------------------------------------------------------ |
| `11_autoencoder`          | **skip** | needs `neurovault.pt` from the skipped `02_data/03_neurovault`                                   |
| `12_neurovault_decoding`  | pass     |                                                                                                  |
| `13_network_labeling`     | pass     | (loader-bypass bug fixed)                                                                        |
| `16_qualatative_auto`     | **skip** | needs `network_examples.csv`, hand-curated, no producer                                          |
| `19_ica_networks`         | **skip** | needs `networks_emb.pt` (same chain as experimental/15); device + loader bugs fixed anyway       |
| `20_pubmed_cv`            | **skip** | needs `ngram_labels.npy` from the skipped `10_n_grams`; loader bug fixed anyway                  |
| `23_versus_others`        | **skip** | needs external NiCLIP/NeuroConText baselines not vendored here                                   |
| `24_brain_to_text_pubmed` | **skip** | full-corpus generation + BERTScore, genuinely >900s even at smoke scale; device bug fixed anyway |

---

## The 2 SSL failures — not a code bug

`03_models/11_anatomical_atlases.ipynb` calls `nilearn.datasets.fetch_atlas_aal()`, which downloads from `www.gin.cnrs.fr`. It fails with:

```
ssl.SSLCertVerificationError: unable to get local issuer certificate
```

Diagnosed directly: `curl` to the same URL succeeds (macOS's system keychain trusts it), but a raw `openssl s_client` handshake using **this conda environment's own CA bundle** (`.conda/ssl/cacert.pem`) fails identically — that bundle is missing the GEANT/Hellenic-Academic-and-Research-Institutions root that `gin.cnrs.fr`'s certificate chains to. This is a local machine/environment configuration gap, confirmed independent of the neurovlm codebase.

`12_neuroadapter.ipynb` fails only as a consequence — it consumes `images_synth.pt`/`synth.parquet`, which `11_anatomical_atlases.ipynb` is supposed to produce. (Its own hardcoded-cuda bug was found and fixed regardless, ahead of it even reaching the missing-file check.)

These are left as **real test failures**, not `skip`s — on a correctly configured machine they should pass, and skipping would hide a future genuine regression. Two fixes, not applied (would touch machine config, not the repo):

- `SSL_CERT_FILE=$(python -m certifi)` before running, to point at Python's own trusted bundle instead of conda's, or
- `conda install -f ca-certificates` to refresh the conda bundle.

---

## Re-verification

After this work was initially completed, notebooks were opened and manually saved in Jupyter. Before finalizing this report, a full fresh check was run to confirm nothing was lost:

- `git show` on the manual-save commit confirmed the only changes were kernelspec metadata (`display_name`, Python version string) in 4 notebooks — cosmetic, no code cells touched.
- Re-synced the corresponding `.py` mirrors (`jupytext --sync`) to pick up that metadata.
- Re-ran the entire `tests/notebooks` suite from a clean state: **25 passed, 19 skipped, 2 failed** — identical to the original result. Nothing regressed.
- Re-ran the deterministic suite: **369 passed**, unchanged.

## How to re-run this whole check next time

```bash
# Fast, deterministic, offline — part of normal CI
pytest -m "not network and not requires_data and not requires_pretrained and not requires_specter and not slow"

# Training pipelines, real 1-epoch runs, ~15s
scripts/smoke_training.sh

# Every notebook, real execution against real cached data, ~20-30 min
scripts/smoke_notebooks.sh
```

If you edit a notebook, edit either the `.ipynb` or its paired `.py` file and run `jupytext --sync <path>` before committing, so the two never drift.
