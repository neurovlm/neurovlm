# Scripts

These scripts reproduce every figure of the paper and of the response to reviewers, from pinned inputs to the drawn
figures. Run them from the repository root.

## Reproduce

```bash
bash scripts/pipeline/reproduce.sh --run-name final   # model, validation, paper figures (~2.5 h on one RTX 5090)
bash scripts/figures/review/core/run_all.sh           # response analyses and figures (~2.5 h)
bash scripts/pipeline/reproduce.sh --smoke            # every stage at smoke scale (~5 min), code path only
```

`reproduce.sh` needs [uv](https://docs.astral.sh/uv/) and an NVIDIA driver >= 580. It builds `.venv` from
`config/requirements.lock` (CPython 3.12.12, torch 2.14.1 built for CUDA 13.0) and fetches every input with
`data/fetch_hf.py` (about 23 GB on first use). It then trains stages 08, 09, 11, 12, 13 and 14 (five seeds) and the
five cross-validation folds of stage 20. It validates the model on the test set, draws the paper figures and compares
every primary metric with the paper model (`pipeline/verify_reproduction.py` against `config/reference_metrics.json`).
A successful run ends with "reproduction PASS: 8/8 components exact". Logs and step times go to
`scripts/outputs/reproduce/<run>/`. Steps resume: verified inputs and trained stages are skipped.

| option | effect |
|---|---|
| `--run-name NAME` | run directory `scripts/outputs/runs/NAME` (default `repro`). `final` is the model the response analyses read, and its figures go to `scripts/figures/main/`. |
| `--home DIR` | data dir under `DIR`, so an existing `~/.cache/neurovlm` is left untouched |
| `--external-dir DIR` | local copies of inputs, checked against the manifest. Put the AAL atlas in `DIR/nilearn_data/aal_SPM12/`, because nilearn's AAL server fails TLS verification. |
| `--cv-folds 0` | one cross-validation fold (about 40 min less). Fig. 5a then draws the single-split model. |
| `--data scratch` | also reruns data stages 01-07 as provenance. Training still reads the pinned inputs. |
| `--smoke` | every stage tiny, then figure previews on 64 samples. The metric comparison fails by design. |
| `--offline`, `--device cpu`, `--skip-env`, `--skip-data`, `--skip-train`, `--skip-validate`, `--skip-figures` | see `--help` |

Docker runs the same command in the locked environment (mounts in the header of `config/Dockerfile`):

```bash
docker build -f scripts/config/Dockerfile -t neurovlm-repro .
docker run --gpus all --shm-size 16g neurovlm-repro --run-name final
```

## Layout

```
scripts/
  README.md          this file
  config/            hf_manifest.json (every Hugging Face input with its revision and sha256), reference_metrics.json
                     (the paper model's validation metrics), requirements.lock, Dockerfile
  pipeline/          reproduce.sh (one command), train_pipeline.py (the stages in order), validate_models.py,
                     verify_reproduction.py, smoke_pipeline.sh (CI), push_hf_release.py, common.py (shared helpers)
  data/              fetch_hf.py (the pinned inputs) and data stages 01-09 (provenance), see data/README.md
  models/            training stages 00-20 and their shared code, see models/README.md
  eval_exclusions/   leakage-audit items dropped from every evaluation, see eval_exclusions/README.md
  figures/main/      one script per paper figure (Figs. 2-6, S2-S4), see figures/main/README.md
  figures/review/    the response: one analysis per directory in core/, RESPONSE.md and the letter,
                     see figures/review/README.md
  outputs/           (git-ignored) runs/<run>/, validation/<name>/, logs
```

## Data

Training, validation and figures read only the files pinned in `config/hf_manifest.json`. `data/fetch_hf.py`
downloads each one verbatim from the Hugging Face org `neurovlm` (and the third-party models) at its pinned revision
and checks its sha256. `--only pipeline|figures|all` selects a subset (default all), and `--check` verifies without
downloading. A local file that differs from the pinned one is reported and replaced only with `--overwrite`. The
scripts read Hugging Face files through `common.hf_pinned`, and each review directory has the same helpers in its
`_hf_pins.py`.

Data stages 01-07 (`data/run_all.py --regenerate`) rebuild the released inputs from their sources. They are optional
provenance. They write to `scripts/outputs/data_regenerated/` and training never reads them. Stages 08 and 09 repeat the
LLM text generation as a proof of concept. Details are in [`data/README.md`](data/README.md).

Three dataset repos were published on 2026-10-09 for inputs that had no public copy: `neurovlm/neurovault_maps`,
`neurovlm/synthetic_pairs` and `neurovlm/pubmed_ale_maps`. E-mail addresses were redacted from
`neurovault_collections.parquet`, so an older local copy is a mismatch. Replace it with `--overwrite` or start from an
empty `--home`.

## Training

`pipeline/train_pipeline.py` runs the stages in order into `scripts/outputs/runs/<run>/` and records
`provenance.json` (command, git commit, manifest sha256, library versions, GPU, driver, determinism settings, exit
codes). Design notes for each stage are in [`models/README.md`](models/README.md).

| stage | script (`models/`) | trains | time |
|---|---|---|---|
| 08 | `08_autoencoder.py` | autoencoder, 101 epochs | 10 min |
| 09 | `09_projection_head.py` | text-to-brain head v2 and the InfoNCE projection heads | 46 s |
| 11 | `11_anatomical_atlases.py` | anatomical atlases from training papers only (CPU) | 8-12 min |
| 12 | `12_neuroadapter.py` | anatomically centered projection head, anchored recipe | 103 s |
| 13 | `13_qformer_pubmed.py` | stage 1 projector (residual MLP, early stopping) | 17 min |
| 14 | `14_qformer_anat.py` | stage 2 projector, five seeds, lowest validation loss packaged | 5 min per seed |
| 20 | `20_pubmed_cv.py` | five-fold cross-validation of 08 and 09 | 48 min |
| 00 | `00_neuroqwen_dapt.py` | NeuroQwen3-0.6B continued pretraining (opt-in, needs the 1.2 M-paper corpus) | 10-11 h (estimate) |

Times are for one RTX 5090 with deterministic kernels. Validation adds 19 min and the paper figures 21 min.

```bash
.venv/bin/python scripts/pipeline/train_pipeline.py --run-name x                     # every stage, validated after each
.venv/bin/python scripts/pipeline/train_pipeline.py --only 14 --run-name x --seeds-14 1,2,3,4,5
.venv/bin/python scripts/pipeline/train_pipeline.py --from 12 --to 14 --dry-run
```

Every stage takes `--smoke` (one epoch, two batches, 64 samples), `--force` (stages skip existing outputs otherwise),
`--seed` and `--no-deterministic`. `--stage-args` and `--validate-args` pass flags to every stage and validation.
Inputs come from `--inputs-from <run dir>`, then this run, then `~/.cache/neurovlm`, then the released checkpoints at
their pinned revisions. `--install` copies a run into the package data dir.

## Validation

```bash
.venv/bin/python scripts/pipeline/validate_models.py --released          # released checkpoints -> validation/released_excl/
.venv/bin/python scripts/pipeline/validate_models.py --run scripts/outputs/runs/final --name final \
    --reference scripts/outputs/validation/released_excl/metrics.json --generation-batch 128
```

`validate_models.py` scores a run on the paper's test splits and writes `scripts/outputs/validation/<name>/`
(`metrics.json`, `report.md`). It covers eight components: the autoencoder (`ae`), the projection heads (`heads`), the
network labels (`networks`), text to brain (`text_to_brain`), the anatomical head (`adapter`), the stage 1 and stage 2
projectors (`qformer_pubmed`, `qformer`) and the cross-validation models (`cv`). A component passes when every primary
metric is at least the released checkpoint's minus a tolerance (AUC 0.01, cosine 0.02, accuracy 0.02). Every evaluation
drops the leakage-audit items of [`eval_exclusions/`](eval_exclusions/README.md): 16 of 2,987 test papers (n = 2,971),
7 of 312 NeuroVault papers with their 29 maps, and 42 of 29,868 cross-validation papers.

### Final run

The paper model is `scripts/outputs/runs/final` with its validation in `scripts/outputs/validation/final`. It is one
`reproduce.sh` run from scratch (2026-10-10), with stage 14 then retrained under its current recipe (level-bank
projection, per-batch map augmentation, title tokens weighted 5, see [`models/README.md`](models/README.md#salience-fig-2)).
Four later runs from scratch reproduce its weights byte for byte ([Determinism and proof](#determinism-and-proof)).
Against the released checkpoints on the PubMed test set (n = 2,971), all eight components pass.

| component | metric | paper | released | final |
|---|---|---|---|---|
| ae | voxel AUROC / bits-per-pixel gain (median %) | 0.975 / - | 0.9749 / 30.11 | 0.9740 / 29.67 |
| heads | PubMed text-to-image / image-to-text AUC | - | 0.8364 / 0.8334 | 0.8356 / 0.8331 |
|  | NeuroVault text-to-image / image-to-text AUC | - | 0.8342 / 0.7757 | 0.8447 / 0.7853 |
| networks | accuracy / macro AUC | - | 0.8067 / 0.9634 | 0.8319 / 0.9619 |
| text_to_brain (head v2) | Dice / Pearson / top-10 % Dice | 0.47 / - / - | 0.4949 / 0.3208 / 0.2581 | 0.5308 / 0.3200 / 0.2809 |
|  | recall AUC / contrastive cosine | - | 0.8648 / 0.2299 | 0.9525 / 0.3449 |
| adapter (anatomical head) | recall AUC / contrastive cosine | - | 0.7922 / 0.1567 | 0.9447 / 0.3113 |
|  | Dice / Pearson | 0.47 / - | 0.4690 / 0.2748 | 0.5288 / 0.3373 |
| qformer (stage 2) | recall AUC / contrastive cosine | - | 0.9396 / 0.2658 | 0.9527 / 0.2807 |
|  | BERTScore F1 / MiniLM cosine | 0.56 / - | 0.5567 / 0.4681 | 0.5530 / 0.4633 |
| qformer_pubmed (stage 1) | recall AUC / contrastive cosine | - | 0.8186 / 0.1867 | 0.8957 / 0.2339 |
|  | BERTScore F1 / MiniLM cosine | - | 0.5730 / 0.3988 | 0.5770 / 0.4310 |
| cv | contrastive / generated text-to-brain mean AUC | - | 0.8230 / 0.7393 | 0.8239 / 0.7415 |

## Generation

Captions come from the frozen NeuroQwen3-0.6B, decoded as in the library. Every script that captions applies two rules
([`models/README.md`](models/README.md#generation)):

- Tokens outside the Latin script are suppressed. Driven by soft prefix tokens, the multilingual LM otherwise drifted
  into Chinese or Korean in about 5 % of the stage 2 captions.
- [NETWORK] and [REGION] titles are restricted to the names of the run's stage 14 canonical bank
  (`<run>/title_vocabulary.json`). `--free-titles` decodes free titles.

bf16 greedy captions depend on the batch size, so validation and Fig. 4 always generate in batches of 128.

Generated maps are multiplied by one factor per model and clipped at 1 (`--map-calibration global`). The factor is
fitted on the validation split so that the median map peaks at 1, like the training maps
([`models/README.md`](models/README.md#map-calibration)).

## Determinism and proof

Training is deterministic by default (`common.setup_torch`). It uses `torch.use_deterministic_algorithms(True)`,
`CUBLAS_WORKSPACE_CONFIG=:4096:8`, seeded Python, NumPy, torch and DataLoader generators, and flash attention with its
deterministic backward in the frozen NeuroQwen3. `reproduce.sh` and the launchers export `PYTHONHASHSEED=0`, because
`bert_score` orders texts through a set. `--no-deterministic` turns this off. It costs 7-12 % per step in stages 13 and
14. Before it, stage 13 recall AUC varied by 0.022 between repeats of the same seed.

The proof is three pairs of full `reproduce.sh` runs. Each run started from a fresh copy of the tree, a fresh `.venv`
and an empty `--home`, with five folds and five stage 14 seeds, on one RTX 5090 (driver 591.44, WSL2). Each took
2 h 22 min to 2 h 34 min.

| runs | date | recipe | result |
|---|---|---|---|
| c, d | 2026-10-10 | previous stage 14 recipe, random hash seed | 77 model and data files and 20 seed-run files identical. Four BERTScore values differed by at most 6e-7 (hash seed, then fixed). Run c became `runs/final`. |
| e, f | 2026-10-11 | current stage 14 recipe, `PYTHONHASHSEED=0` | the 97 files identical to each other and to `runs/final`. Every metric value equal, the nine figure PNGs identical, 8/8 exact. |
| g, h | 2026-10-12 | plus the Latin-script suppression | as e and f, plus identical captions and 13 identical figure pages. 12 equal the committed PNGs, Fig. S3 differs by design (below). |

```bash
bash scripts/pipeline/reproduce.sh --home "$(mktemp -d)" --external-dir DIR --run-name final_v3   # as in runs g and h
cat scripts/outputs/reproduce/final_v3/verify.log                                               # metrics vs the paper model
```

Files that differ are not model files. Their differences are clock fields and absolute paths in run records, the random
temporary name inside torch's atomic-save checkpoint containers (members identical), the object ids in the AAL atlas
pickle (values equal), and the dates and element ids of SVG files. The expected weights, on the same GPU model,
driver and libraries:

| file | sha256 in runs e, f, g, h and `runs/final` |
|---|---|
| `autoencoder.safetensors` (08) | `151d62eebc805ea79007415355c8b26069af7d9c8ee545b7fddb6a5922731f55` |
| `neuro_qformer_pubmed/model.safetensors` (13) | `4fc17243c5514021ca8d8543112826107da3c87899531c279698531bbeb5cb37` |
| `neuro_qformer_canonical/model.safetensors` (14) | `eaf5ee494e96e40f9c75f2b9bd4aed0edcbe4589b8fafe9d0365402a5515a5a9` |
| `canonical_semantic_banks.pt` (14) | `f28293a0595afab0a137ba8cc96b8023b5562e7b464f7071a3d5230da7f20771` |
| `autoencoder_kfold-4.safetensors` (20) | `c07c28c515fdd3d29466323bfdcb8dcb219ad5277527f67219e65392b9ca36e1` |

The canonical titles and the map calibration came on 2026-10-13, after runs g and h. They change captions, maps and
metrics but no training stage, so the weight evidence stands. `validation/final` and `reference_metrics.json` were
regenerated with them, and a repeat of the two-run proof is planned after the NeuroQwen3-0.6B retrain.

## Response to reviewers

`figures/review/` holds the response. `RESPONSE.md` is the response text, and `response_letter.tex` sets it as the
letter with its figures and tables. Each analysis in `core/` has a `run_analysis.py` (recomputes `outputs/`), a
`make_figure.py` (redraws from `outputs/` on the CPU) and a README. They read the final model at
`scripts/outputs/runs/final`, its validation and the paper figures, so run `reproduce.sh --run-name final` first.

```bash
bash scripts/figures/review/core/run_all.sh                       # every analysis, one GPU job at a time (~2.5 h)
FIGURES_ONLY=1 bash scripts/figures/review/core/run_all.sh        # redraw every response figure (CPU, ~2 min)
.venv/bin/python scripts/figures/review/build_reviewer_dirs.py    # reviewer_01/ .. 03/ (run_all.sh ends with it)
.venv/bin/python scripts/figures/review/check_letter_sync.py      # the letter carries RESPONSE.md (0 mismatches)
cd scripts/figures/review && pdflatex response_letter.tex && pdflatex response_letter.tex
```

[`figures/review/README.md`](figures/review/README.md) maps each comment to its analysis and explains the letter
build, including the copy with the marked manuscript.

## Tests and CI

Two workflows run on pull requests to `main`. Both use the locked environment with the CPU build of torch,
`PYTHONHASHSEED=0` and no secrets.

- `.github/workflows/tests.yml` runs the offline test suite (`tests/`). Tests that download data or weights carry the
  markers `network` or `requires_*` and are deselected.
- `.github/workflows/smoke.yml` fetches the training inputs of `config/hf_manifest.json` (54 files, 8.1 GB, pinned
  and sha256-checked) and the AAL atlas (sha256-checked), then runs the pipeline smoke test
  `tests/scripts/test_smoke_pipeline.py` (`SMOKE_COVERAGE=1 bash scripts/pipeline/smoke_pipeline.sh --skip-data
  --no-validate --device cpu`, every training stage for one batch on the CPU, every stage must exit 0). It checks that
  the pipeline runs end to end, not its numbers, and reports the coverage of the reproduction code (`.coveragerc`).

Local run of both jobs after the move to `pipeline/` and `config/` (fresh clone, fresh `.venv` from the lock, empty
`HOME`, 16 CPU cores, 2026-10-14):

| job | step | time | result |
|---|---|---|---|
| tests | environment from the lock | 15 s | torch 2.14.1+cpu |
| tests | offline suite | 13 s | 423 passed, 1 skipped (CUDA), 97 deselected, `neurovlm` coverage 95.3 % (2026-10-26, training and inference scope of `.coveragerc`) |
| smoke | environment, AAL atlas, inputs (`fetch_hf.py`, 54 train files, empty cache) | 141 s | 54 ok, 0 mismatch |
| smoke | `smoke_pipeline.sh` (00, 08, 09, 20, 11, 12, 13, 14) | 247 s | every stage exit 0, reproduction-code coverage 36.6 % (scope of `.coveragerc` before 2026-10-26, which also counted the figure scripts and validation; the next CI smoke run re-measures it) |

## Publishing the release

`pipeline/push_hf_release.py` publishes `runs/final` on the Hugging Face org `neurovlm`, one commit per repo. Without
`--execute` it is a dry run that only reads the Hub. `--execute` refuses on a dirty git tree, a file whose sha256 is not
the validated run's (`EXPECTED_SHA256`, cross-checked with the table above), an e-mail address in an upload, or
weights the installed `neurovlm` loaders cannot read. `--update-manifest` then re-pins the pushed repos in
`config/hf_manifest.json`. Run each command without `--execute` first.

```bash
.venv/bin/python scripts/pipeline/push_hf_release.py --only models --execute --update-manifest
.venv/bin/python scripts/pipeline/push_hf_release.py --only neuroqwen --neuroqwen-dir scripts/outputs/runs/neuroqwen/neuroqwen \
    --execute --update-manifest
.venv/bin/python scripts/pipeline/push_hf_release.py --only corpus --execute --yes --update-manifest
```

## Known limits

- Bitwise identity was shown on one GPU model, driver and set of library versions. PyTorch guarantees it only there,
  so another GPU may give different bits (`provenance.json` records the GPU and driver).
- The proof runs used a warm Hugging Face cache, so the first download (about 23 GB) is not in their times.
- nilearn's AAL server fails TLS verification, so the proof runs took the AAL atlas from `--external-dir`.
- Fig. S3 adds the NiCLIP and NeuroConText baselines from an external checkout (`--external-root`). Runs without it
  draw NeuroVLM only.
- The published NeuroQwen3-0.6B was trained before stage 00 removed the evaluation papers from its corpus, which holds
  2,906 of the 2,971 test papers. Stages 13 and 14 still use it. `figures/review/core/a4_memorization_check` tests
  whether captions copy the abstracts.
- The stage 13 and 14 packages hold the MLP projector. `neurovlm` loads them since `src/neurovlm/models/qformer.py`
  gained `ResidualMLPProjector` (October 2026): `NeuroQFormer.from_state_dict_payload` reads `config.json["projector"]`.
  Older installs fail on `NeuroQformer` and `NeuroQformer-PubMed` with a state-dict size error.
