# Scripts

Headless ports of the paper notebooks: data preprocessing, model training, validation, and the figures of the paper
and of the response to reviewers. Run everything from the repository root:

```bash
.venv/bin/python scripts/data/fetch_hf.py                         # every input from Hugging Face, pinned, sha256-checked
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
bash scripts/smoke_pipeline.sh --device cpu                       # every stage, tiny -> runs/smoke
.venv/bin/python scripts/train_pipeline.py --run-name paper       # full retrain -> runs/paper
.venv/bin/python scripts/validate_models.py --run scripts/outputs/runs/paper --name paper \
    --reference scripts/outputs/validation/released_excl/metrics.json --generation-batch 128
.venv/bin/python scripts/figures/main/make_all.py --model-dir scripts/outputs/runs/paper
```

```
scripts/
  common.py              shared CLI, timing, device, determinism, HF pinning and exclusion helpers
  hf_manifest.json       every Hugging Face file read here: repo, pinned revision, sha256
  data/                  fetch_hf.py (pinned inputs) and the 02_data ports (01..07, run_all.py), see data/README.md
  eval_exclusions/       leakage-audit items dropped from every evaluation, see eval_exclusions/README.md
  models/                03_models ports (08 09 11 12 13 14), 20 (figures/20 cross-validation), shared code in _*.py
  train_pipeline.py      data -> 08 -> 09 -> 20 -> 11 -> 12 -> 13 -> 14, validating after each stage
  validate_models.py     paper metrics of a run vs the released checkpoints (PASS/FAIL)
  reproduce.sh           one-command reproduction (requirements.lock, Dockerfile), see Reproducibility
  verify_reproduction.py a reproduction's metrics vs runs/final (reference_metrics.json)
  smoke_pipeline.sh      train_pipeline.py --smoke (smoke_training.sh, smoke_notebooks.sh: library tests, tests/README.md)
  figures/main/          one script per paper figure (Figs. 2-6, S2-S4), see figures/main/README.md
  figures/review/        response to reviewers, see figures/review/README.md
  outputs/               (git-ignored) runs/<run>/, validation/<name>/, timings
```

## Conventions

* Artifacts go to `scripts/outputs/runs/<run_name>/` (`--run-name`, default a timestamp; `--smoke` -> `runs/smoke`),
  never to `~/.cache/neurovlm` unless `--install` (copies them into `neurovlm.data.data_dir` under the notebook names).
* Inputs resolve as `--inputs-from <run dir>` > this run > `data_dir` (`~/.cache/neurovlm`) > released HF checkpoints
  at their pinned revisions.
* Every Hugging Face file is read at the revision pinned in `hf_manifest.json` and checked against its sha256:
  `common.hf_pinned` (a file, or a repo's revision), `common.hf_pinned_dir` (a snapshot, for bert_score) and
  `hf_cache_pinned` (`data/data_utils.py`, `figures/main/_shared.py`; the library loaders take no revision, so it checks
  that the HF cache serves the pinned one). Each review directory has the same helpers in its `_hf_pins.py`. A model id
  given on the command line (e.g. `07_transform_text.py --model`) must be in the manifest.
* Every stage has `--smoke` (1 epoch, 2 batches, 64 samples), `--epochs`/`--batch-size`, skip-if-exists unless
  `--force`, the notebook's seeds (`--seed`), deterministic kernels (`--no-deterministic` turns them off, see
  [Determinism](#determinism)), timings in `scripts/outputs/<stage>/timings.jsonl` and `<stage>.summary.json` in the run.

## Data

`scripts/data/fetch_hf.py` downloads every input verbatim from the Hugging Face org `neurovlm` (and the third-party
models) at the revision pinned in `scripts/hf_manifest.json` and checks its sha256. `--only pipeline|figures|all`
(default all, ~23 GB), `--check` verifies without downloading, `--overwrite` replaces files that differ. A local file
that differs from the pinned one is reported and never replaced without `--overwrite`. `data/run_all.py` does the same
by default (`--from-hf`). `--regenerate` reruns stages 01-07 as provenance into `scripts/outputs/data_regenerated/` and
never touches the data dir.

Three dataset repos were pushed on 2026-10-09 for inputs that were not published before: `neurovlm/neurovault_maps`
(the NeuroVault scrape, its maps and stage 03's outputs), `neurovlm/synthetic_pairs` (`synth.parquet`,
`text_synth_less.parquet`) and `neurovlm/pubmed_ale_maps` (the released ALE maps). E-mail addresses were redacted from
`neurovault_collections.parquet` before publishing, so an author's unredacted local copy is reported as a mismatch by
`--check`; replace it with `fetch_hf.py --overwrite` or start from a fresh `reproduce.sh --home`.

Stage 01 reproduces the released ALE maps with `--ijk-mode trunc` (default). They were built from the current coordinate
table with nimare <= 0.15, which truncates MNI coordinates to voxel indices; nimare 0.22 rounds them, so most foci move
by one voxel. `--ijk-mode round` gives the nimare 0.22 behaviour. Details in [`data/README.md`](data/README.md).

## Stages

| stage | notebook | outputs (run dir) | notes |
|---|---|---|---|
| 00 | efficient_ai 01_dapt (NeuroQwen3-0.6B) | neuroqwen/NeuroQwen3-0.6B/ (HF model folder), eval_filter.json, eval_filter_removed.csv, history.jsonl, export_check.json | opt-in, not in `train_pipeline.py`; evaluation papers removed, see [Stage 00: NeuroQwen](#stage-00-neuroqwen) |
| 08 | 03_models/08_autoencoder | pmids_split.pt, autoencoder.{pt,safetensors}, latent_neuro.pt | per-epoch voxel AUROC on the GPU (`--no-fast-eval` = library CPU path) |
| 09 | 03_models/09_projection_head | proj_head_text_mse (head v2), proj_head_text_bce (notebook head), proj_head_text_infonce, proj_head_image_infonce | [Text-to-brain head v2](#text-to-brain-head-v2); `--text-head v1` = notebook head; `--only-text-head` = head v2 into an existing run |
| 20 | figures/20 (cells 2-6) | {autoencoder,proj_head_text_mse,proj_head_text_infonce,proj_head_image_infonce}_kfold-i.pt | `--folds`; validation draw seeded (1000+fold, 2000+fold) |
| 11 | 03_models/11_anatomical_atlases | df_region_terms.parquet, images_agg.pt, images_synth.pt, synth_text.parquet | CPU (joblib); training papers only, see [Stage 11 split leak (fixed)](#stage-11-split-leak-fixed) |
| 12 | 03_models/12_neuroadapter | adapter_anat_full.{pt,safetensors}, adapter_anat_full.lineage.json | anatomical head; `--recipe anchored` (default) or `notebook`, see [Anatomical head v2](#anatomical-head-v2-stage-12-anchored) |
| 13 | 03_models/13_qformer_pubmed | neuro_qformer_pubmed/, qformer_grounded_permissive_neuro_epoch-*.pt, history_13_qformer_pubmed.json, probes.jsonl | stage 1 projector (grounded, PubMed); residual MLP with early stopping, see [Projector choice](#projector-choice) |
| 14 | 03_models/14_qformer_anat | neuro_qformer_canonical/, canonical_semantic_banks.pt, qformer_canonical_bottleneck_epoch-29.pt, generations_refs.jsonl | stage 2 projector (canonical); synthesised maps max-scaled and binarised; training semantics on the row's own level bank, per-batch map augmentation, title tokens weighted 5; see [Salience](#salience-fig-2) |

Stages 13 and 14 keep the `qformer` file names (scripts, checkpoints, packages, validator components `qformer_pubmed`
and `qformer`) whichever projector they hold; the kind is in the package's `config.json["projector"]`.

Differences from the notebooks:
* 11 uses training-split papers only (`--splits all` = notebook) and writes `synth_text.parquet` instead of overwriting
  its input `synth.parquet`.
* 12, 13 and 14 take the heads, adapter and grounded projector from the run directory when present (12 `--base
  released`, 13 `--heads released`, 14 `--grounded-qformer`/`--adapter <file>` restore the notebook wiring). With run
  heads, 13 also reads the run's `latent_neuro.pt` (`--latents auto`); the notebook's HF latents only match the released heads.
* `synth.parquet` (11) and `text_synth_less.parquet` (14) have no producer in this tree; they are published as
  `neurovlm/synthetic_pairs`.

## Validation (`validate_models.py`)

```bash
.venv/bin/python scripts/validate_models.py --released                 # released reference -> validation/released_excl/
.venv/bin/python scripts/validate_models.py --run scripts/outputs/runs/final --name final \
    --reference scripts/outputs/validation/released_excl/metrics.json   # or compare with runs/final: scripts/verify_reproduction.py
```

Components (paper test splits): `ae` voxel AUROC and bits-per-pixel gain; `heads` retrieval AUC, recall@k, MRR;
`networks` 8-class labelling accuracy and AUC; `text_to_brain` (text head + decoder) and `adapter` (anatomical head)
Dice, Pearson, top-10 % Dice, recall AUC, contrastive cosine, each vs a shuffled null; `qformer_pubmed` and `qformer`
brain-to-text recall AUC, contrastive cosine, BERTScore F1, MiniLM cosine and output validity; `cv` the 5-fold CV models.
`--components` and `--max-samples` select a subset. Output: `scripts/outputs/validation/<name>/{metrics.json,report.md}`.
PASS = retrained >= released - tolerance on every primary metric (`--auc-tol 0.01`, `--cos-tol 0.02`, `--acc-tol 0.02`,
`--bpp-tol 2`). `--skip-generation` re-scores saved captions. The smoke pipeline's FAILs (64 samples) are expected.

**Exclusions (default on).** Every PubMed-test, NeuroVault and CV evaluation drops the leakage-audit items of
[`eval_exclusions/`](eval_exclusions/README.md): 16 of 2,987 PubMed test papers (n = 2,971), 7 of 312 NeuroVault papers
with their 29 maps (305 papers), and 42 of 29,868 CV papers (all in fold 0). Counts are in `metrics.json["meta"]`.
`--no-exclusions` evaluates the full sets (output names without `_excl`). `text_to_brain` and `adapter` score calibrated
maps (`--map-calibration global`, default, see [Map calibration](#map-calibration)), the released reference too, with
the released models' own factors.

## Generation

Captions come from the frozen, multilingual NeuroQwen3-0.6B. Reading the projector's soft prefix tokens instead of text,
it used to drift at low-margin steps into Chinese, Korean, Cyrillic or Thai tokens and byte fragments (165 of 2,971
stage-2 test captions held CJK characters, 1 at stage 1), although no training text holds such characters. Every script
that captions therefore sets `lm.generation_config.suppress_tokens = list(latin_script_suppress_tokens(tok))`
(`models/_common_models.py`; applied in `validate_models.Bundle.nvlm`, `figures/main/_shared.make_nvlm` and
`pin_neuroqwen` of the review analyses' `_hf_pins.py`): the 52,962 tokens with a character outside ASCII, Latin-1, Greek
and U+2000-2BFF, or that are not valid UTF-8 alone. Special tokens are kept, and no training text uses a suppressed
complete token. `model.generate` reads that setting, so `NeuroVLM.generate_text` and the decoding recipes (greedy or beam
search, repetition penalty) are unchanged. The validator strips only the `[NETWORK]` / `[REGION]` / `[FUNCTION]` tags the
LM sometimes writes (`strip_steering_tags`; the earlier `.*]` deleted each line up to its last `]`) and reports
`validity_frac_non_latin`, which must be 0 (FAIL otherwise). Together this changed 320 stage-2 captions and 1 stage-1
caption and moved the stage-2 metrics by at most 5e-4 (recall AUC 0.9544 -> 0.9542, BERTScore F1 0.5520 -> 0.5524).
ASCII junk in occasional second paragraphs (code fences, HTML tags, markdown bullets; 235 of 2,971 stage-2 captions) is
left as generated. Batch size: bf16 greedy captions depend on the generation batch (93 % of stage-2 fields differ
somewhere between batches 64, 128 and 512), so captions repeat bit for bit only at the same batch; validation and Fig. 4
use 128.

**Caption titles.** A [NETWORK] or [REGION] caption is "[LEVEL]Title\nbody". Decoded freely, the LM composed titles that are no
name it was trained on: 827 of the 2,971 stage-2 test [NETWORK] titles and 301 [REGION] titles were not a name of the run's
stage-14 canonical bank ("Cerebro-Cervico-Brain", "Medial Personal Relevance Cortex"). Generation therefore restricts the title
after a [NETWORK] / [REGION] steering token to those names (`models/_common_models.py`): `title_vocabulary(run_dir)` matches
every row of `canonical_semantic_banks.pt` to its curated stage-14 text (SPECTER2 + the run's `proj_head_text_infonce`, cosine
>= 0.999) and caches the sorted names as `<run_dir>/title_vocabulary.json` (48 network, 83 region, 163 function names in
runs/final; rebuilt when the banks change; the released banks get their own file). `install_title_constraint(lm, tok, names)`
wraps `generate` of that LM instance, because `NeuroVLM.generate_text` passes no logits processor and `generation_config`
cannot hold one: for every prompt row that ends with the [NETWORK] / [REGION] tag (its token embeddings in `inputs_embeds`) it
adds a logits processor that keeps each beam on a trie of the names' token ids until the newline that ends the title. The body,
[FUNCTION] captions and stage-1 captions (no tag) are decoded as before. It is applied where the Latin-script suppression is
(`validate_models.Bundle.nvlm`, `figures/main/_shared.make_nvlm`, `pin_neuroqwen` of the review analyses' `_hf_pins.py`, which
reads runs/final/title_vocabulary.json); `--free-titles` (validate_models.py and the figure scripts) or
`pin_neuroqwen(nvlm, free_titles=True)` decode free titles. The validator reports `validity_frac_noncanonical_title` (stage 2;
must be 0 unless `--free-titles`), and figure caches that hold captions carry `_titles-vocab` in their names. On the stage-2
test set this changed 832 [NETWORK] and 311 [REGION] captions (1,038 papers; no [FUNCTION] or stage-1 caption) and moved
the stage-2 metrics by at most 1.5e-3 (recall AUC 0.9542 -> 0.9527, contrastive cosine 0.2820 -> 0.2807, BERTScore F1
0.5524 -> 0.5530, MiniLM cosine 0.4625 -> 0.4633); generation time is unchanged.

### Stage 00: NeuroQwen

`models/00_neuroqwen_dapt.py` retrains the text decoder, NeuroQwen3-0.6B: continued pretraining of `Qwen/Qwen3-0.6B` on
PubMed titles and abstracts, ported from efficient_ai `project/notebooks/01_dapt.ipynb` (the recipe on the model card).
The corpus is `pubmed_neuroimaging.parquet`, 1,231,613 PubMed records from the neuroimaging query of efficient_ai
`project/scrape.py`. It is not on the Hugging Face Hub, so `hf_manifest.json` pins it by sha256 under `not_on_hub` and
the script takes it with `--corpus`. The base model is pinned in `files` (group `extra`).

Recipe: one document per paper, `<title>\n\n<abstract><|im_end|>`, packed like TRL's `wrapped` strategy (each run of 1,000
documents concatenated and cut into 1,024-token sequences), 16 sequences (16,384 tokens) per optimizer step, one epoch,
AdamW at lr 5e-6 with 10 % warmup and cosine decay, weight decay 0.01, gradient clipping 0.3, 1 % of the documents held
out for validation (seed 12345). A plain loop replaces TRL's `SFTTrainer` (TRL is not in `requirements.lock`). With
`--no-eval-filter` it rebuilds the notebook's split and token count exactly (1,219,296 training documents, 385,468,745
tokens).

Precision: with bf16 weights and lr 5e-6 most AdamW updates fall below bf16 resolution, so the published model differs
from Qwen3-0.6B in only 4.6 % of its weights. The script therefore keeps fp32 master weights and AdamW states and runs
the passes under bf16 autocast (the export stays bf16), which departs from the notebook; `--weights-bf16` restores the
notebook's behavior. Memory: weights, gradients and AdamW states of the 0.75 B parameters (untied head) take 12 GB in fp32
(5.6 GB with `--weights-bf16`), and each 1,024-token sequence needs about 4 GB of activations and logits. The script
therefore runs 2 sequences per forward pass with 8 accumulation steps (about 22 GB; 4 x 4 with `--weights-bf16`) instead
of the notebook's 8 x 2, the same 16 sequences per step.

Evaluation filter (one marked block in the script): it removes the validation and test papers of the pinned
`publications.parquet` split (by PMID, or by normalized title) and the NeuroVault evaluation papers (the 312 released
publications, which include the 7 of `eval_exclusions/neurovault_exclusions.csv`, by DOI or by title). The DOIs in
`publications.parquet` are not used because some of them belong to another paper. On the corpus the filter removes
6,144 of 1,231,613 documents: 5,832 by PMID (2,922 of the 2,987 test papers and 2,910 of the 2,987 validation papers),
10 more by title, and 302 more NeuroVault papers (303 of the 312 are in the corpus, one of them also a held-out PubMed
paper). The counts go to `neuroqwen/eval_filter.json` and the removed rows to `eval_filter_removed.csv`.
`--no-eval-filter` keeps everything and still writes the counts. The published `neurovlm/NeuroQwen3-0.6B` was trained
before this filter existed, on the full corpus.

Output head: the published model's `config.json` says `tie_word_embeddings: true`, but its weights hold a separately
trained `lm_head` (transformers 5 trained the two matrices untied). transformers 4.57 (this venv) loads it tied and
replaces the trained `lm_head` with the trained `embed_tokens`, which is what stages 13 and 14 have used. This script
trains the two untied, like the published run, and exports `tie_word_embeddings: false`, so every transformers version
keeps the trained head. `export_check.json` reloads the folder on the CPU and checks the head, the tokenizer and 10 greedy
tokens. transformers 4.57.6 prints a `fix_mistral_regex` warning for any folder it saved itself, and the Qwen tokenizer is
not affected.

```bash
.venv/bin/python scripts/models/00_neuroqwen_dapt.py --run-name neuroqwen \
    --corpus ~/efficient_ai/project/pubmed/pubmed_neuroimaging.parquet      # full run, RTX 5090: ~10-11 h
.venv/bin/python scripts/models/00_neuroqwen_dapt.py --smoke --device cpu   # 1 step on 256 documents, length 128: ~50 s
```

The full run trains 23,522 steps on 1,213,214 documents (384.8 M tokens). It computes the validation loss every 1,000
steps (`--eval-every`) and writes a resumable `neuroqwen/checkpoint.pt` every 1,000 steps (`--save-every`, `--resume`).
The 10-11 h are estimated, not measured: the author's local run of the same 16,384-token steps at length 2,048
(efficient_ai `05_dapt_pubmed_neuroimaging.ipynb`, bf16 weights, 0.73 steps/s, 8 h 51 min projected for 23,679 steps),
plus deterministic kernels, fp32 master weights and the 23 validation passes. `smoke_pipeline.sh` runs the smoke.
Stages 13 and 14 still load the published model from the Hub.

## Orchestrator (`train_pipeline.py`)

```bash
.venv/bin/python scripts/train_pipeline.py --only 08,09 --run-name x --no-validate
.venv/bin/python scripts/train_pipeline.py --from 12 --to 14 --dry-run
.venv/bin/python scripts/train_pipeline.py --only 14 --run-name x --seeds-14 1,2,3,4,5
```

`--stage-args "..."` is appended to every stage and `--validate-args "..."` to every validation; `--projector` goes to
13 and 14, `--mlp-width/--mlp-depth/--early-stop-patience/--min-epochs` to 13, `--text-head` to 09. `--seeds-14` trains
stage 14 once per seed in `runs/<run>_s<seed>/` and packages the seed with the lowest final-epoch validation loss into
`runs/<run>/` (`stage14_seed_selection.json`). Logs in `runs/<run>/logs/`, timings in `scripts/outputs/pipeline_timings.json`,
the run record in `runs/<run>/provenance.json` (see [Reproducibility](#reproducibility)). `--stage-args "--no-deterministic"`
turns deterministic kernels off in every stage.

## Final run (`runs/final`)

The paper model is one `scripts/reproduce.sh` run from scratch (proof run c, 2026-10-10: stages 08 09 11 12 13 14 with
five stage-14 seeds, then 20 with five folds), copied into `runs/final` with its validation in `validation/final`. A
second run from scratch (d) gave byte-identical weights ([`REPRODUCTION.md`](REPRODUCTION.md)). Stage 14 was then
retrained in `runs/final` with a new recipe (2026-10-10, [Salience](#salience-fig-2)):
`train_pipeline.py --only 14 --run-name final --seeds-14 1,2,3,4,5 --force --skip-data`, on run c's inputs. Two later
runs from scratch with the new recipe (e and f, 2026-10-10/11) each equal the current `runs/final` byte for byte: all 77
model and data files, stage 14 included, and the five stage-14 seed runs. The same thing without the wrapper is
`train_pipeline.py --run-name final --seeds-14 1,2,3,4,5` followed by `train_pipeline.py --only 20 --run-name final
--skip-data`. Stage 13 stopped after epoch 7 (best validation loss at epoch 4, packaged).

Stage 14 trains five seeds and packages the one with the lowest final validation loss (seed 3). Each training row's
semantics are projected onto its own level bank (`[NETWORK]` titles onto the network bank), as at inference. Every
training map is re-binarised per batch, with draws from a generator seeded from the training seed. Title tokens weigh 5
in the training loss; the validation loss is unweighted. Validation split fixed by `--split-seed 123`; titles scored
after the selection, never used for it (beam 5, maps rendered as in Fig. 2); deterministic run:

| seed | val loss, epoch 30 | title accuracy (119 network maps) | salience-class accuracy (15 maps) | Du SAL/PMN map title |
|---|---|---|---|---|
| 1 | 1.3144 | 0.739 | 0.47 | Salience Network |
| 2 | 1.3225 | 0.765 | 0.40 | Salience Network |
| **3 (selected)** | **1.3005** | **0.748** | **0.53** | **Salience Network** (Fig. 2) |
| 4 | 1.3144 | 0.765 | 0.53 | Salience Network |
| 5 | 1.3186 | 0.798 | 0.53 | Salience Network |

Five-seed mean title accuracy 0.763; the previous recipe, scored the same way, gave 0.743 (0.723-0.765), with the Du
map right in 2 of 5 seeds. The packaged seed gets one map fewer right than the previous one (89 vs 90 of 119). Validation
losses are not comparable across recipes (previous seed 3: 1.3183), because the training inputs differ.

Validation (`validation/final/report.md`; PubMed test n = 2,971; reference = released checkpoints): **8 of 8 PASS**.

| component | metric | paper | released | final |
|---|---|---|---|---|
| ae | PubMed voxel AUROC / bpp gain (median %) | 0.975 / - | 0.9749 / 30.11 | 0.9740 / 29.67 |
| heads | PubMed t->i / i->t AUC | - | 0.8364 / 0.8334 | 0.8356 / 0.8331 |
|  | NeuroVault t->i / i->t AUC | - | 0.8342 / 0.7757 | 0.8447 / 0.7853 |
| networks | accuracy / macro AUC | - | 0.8067 / 0.9634 | 0.8319 / 0.9619 |
| text_to_brain (head v2) | Dice / Pearson / top-10 % Dice | 0.47 / - / - | 0.4949 / 0.3208 / 0.2581 | 0.5308 / 0.3200 / 0.2809 |
|  | recall AUC / contrastive cosine | - | 0.8648 / 0.2299 | 0.9525 / 0.3449 |
| adapter (anatomical head v2) | recall AUC / contrastive cosine | - | 0.7922 / 0.1567 | 0.9447 / 0.3113 |
|  | Dice / Pearson | 0.47 / - | 0.4690 / 0.2748 | 0.5288 / 0.3373 |
| qformer (stage 2) | recall AUC / contrastive cosine | - | 0.9396 / 0.2658 | 0.9544 / 0.2823 |
|  | BERTScore F1 / MiniLM cosine | 0.56 / - | 0.5567 / 0.4681 | 0.5520 / 0.4622 |
| qformer_pubmed (stage 1) | recall AUC / contrastive cosine | - | 0.8186 / 0.1867 | 0.8957 / 0.2339 |
|  | BERTScore F1 / MiniLM cosine | - | 0.5730 / 0.3988 | 0.5770 / 0.4310 |
| cv | contrastive / generated t->b mean AUC | - | 0.8230 / 0.7393 | 0.8239 / 0.7415 |

The previous stage-14 recipe gave 0.9486 / 0.2744 and 0.5524 / 0.4654. The retrain changed no other component; stage
13's captions are byte-identical and its BERTScore F1 moved by -6.6e-7 (scorer noise).

The text_to_brain and adapter rows score maps calibrated by the global factor (2026-10-13, [Map
calibration](#map-calibration)); with each map divided by its maximum they were 0.5308 / 0.3223 / 0.2814 and 0.9499 /
0.3421 (head v2), 0.9435 / 0.3127 and 0.5288 / 0.3422 (anatomical head). No other component changed.

### Text-to-brain head v2

The notebook's text head (`ProjHead(768, 512, 384)`, BCE against the raw map through the frozen decoder) lost to a
ridge regression from SPECTER2 to the autoencoder latent (validation Dice 0.489 vs 0.524): trained only through the
decoder, it lands off the latent manifold and decodes to near-average maps. Head v2 (same module): loss = latent MSE + 1.0 (1 - r of decoder logits) + 0.3 (1 - r of sigmoid
maps) against the decoded true latent; input noise `l2n(x + 0.5 N(0, I)/sqrt(768))`; output bias initialised at the mean
training latent; AdamW (lr 1e-3, wd 0.01), batch 512, up to 80 epochs, the epoch with the best validation Dice kept.

| model (validation, n = 2,961) | Dice | top-10 % Dice | Pearson |
|---|---|---|---|
| ridge, alpha 0.3 | 0.5242 | 0.2746 | 0.3067 |
| notebook head | 0.4891 | 0.2553 | 0.3166 |
| **head v2 (stage 09 in `runs/final`)** | **0.5322** | **0.2808** | **0.3173** |

Test (once, n = 2,971; `figures/review/core/a1_text_to_brain_baselines`): Dice / top-10 % Dice / Pearson 0.531 / 0.281
/ 0.322 for head v2, 0.526 / 0.275 / 0.311 for ridge, 0.492 / 0.256 / 0.321 for the notebook head. Stage 09 still trains
the notebook head (`proj_head_text_bce`) to initialise the InfoNCE text head; the CV models (20) keep the notebook recipe.

### Map calibration

Every generated map (sigmoid output of head v2 + decoder, or of the anatomical head) is multiplied by one factor per
model and clipped at 1 before it is drawn, encoded or scored (`_common_models.calibrate_maps`, `--map-calibration
global`, the default of `validate_models.py` and every figure script). The factor is fitted on the PubMed validation
split (2,961 papers after the leakage-audit exclusions): the median over the validation maps of 1 / map peak, so the
median generated map peaks at 1 like every training map. `_common_models.map_calibration` fits it once and caches it in
`runs/final/map_calibration.json` (head v2 19.16, anatomical head 6.13) with the sha256 of the model files, refitting
when they change (released models: `scripts/outputs/released_map_calibration.json`). One factor for all maps keeps their
relative scale: a weak prediction stays weak, half of the maps reach 1. `--rescale-max` (= `--map-calibration max`)
divides each map by its own maximum instead, the previous rule; `none` gives the raw maps.

Why a factor and not a probability calibration: head v2's raw maps peak at 0.05 (validation median), the training maps
at 1 because each was divided by its maximum. Read as expected activation of the training maps, the raw values are too
low at the low end (0.014 observed at 0.001; mean 0.004 vs 0.019) and about right to slightly high at the peaks (0.30 at
0.36), so a low peak mostly means low confidence: on the test split the raw peak correlates with each map's Dice,
Pearson r and retrieval rank (Spearman 0.38, 0.41, 0.54). Isotonic regression or affine scaling of the logits, fitted on
the validation split, calibrates the values (expected calibration error 0.015 -> 0.0001 / 0.0009) but leaves the peaks
at 0.10 / 0.08, and the encoder, trained on peak-1 maps, then retrieves worse (test recall AUC 0.807 / 0.801).

| test split, head v2 (n = 2,971) | Dice | Pearson r | recall AUC | contrastive cosine | ECE | peak ~ rank (Spearman) |
|---|---|---|---|---|---|---|
| raw maps | 0.531 | 0.322 | 0.887 | 0.296 | 0.015 | 0.54 |
| each map / its maximum (previous rule) | 0.531 | 0.322 | 0.950 | 0.342 | 0.047 | - |
| affine logits, 2 parameters | 0.531 | 0.314 | 0.801 | 0.228 | 0.001 | 0.33 |
| isotonic regression | 0.531 | 0.326 | 0.807 | 0.242 | 0.0001 | 0.50 |
| **global factor, clipped at 1** | **0.531** | **0.320** | **0.953** | **0.345** | 0.054 | **0.63** |

Dice uses each map's own percentile thresholds, which no monotone calibration changes. Against the per-map maximum the
factor gains recall AUC and loses a little at the top of the ranking (contrastive recall@10 0.350 vs 0.367) and in
Pearson r (clipping). The anatomical head behaves the same way (recall AUC 0.945 vs 0.944, Pearson r 0.337 vs 0.342).
Not covered by the rule: stage 12 selects its epoch on recall AUC of maps divided by their maximum (the rule
`runs/final` was trained with); stage 14 divides each synthesised map by its maximum before binarising it at a
percentile, where clipping would tie voxels at 1; the CV component keeps figures/20 cell 9's per-map maximum (the folds
have no validation split).

### Anatomical head v2 (stage 12, anchored)

With the notebook recipe, the anatomical head started from head v2 drifts away from PubMed while fitting the synthetic
pairs (PubMed-validation recall AUC 0.911 at epoch 1, 0.783 at epoch 2,000; release 0.792). The anchored recipe
(`--recipe anchored`, default) keeps the notebook's pairs, model, optimiser, schedule and 2,000 epochs, adds head v2's
loss on 512 random PubMed training pairs per step (`--anchor-weight 1.0`, `--anchor-batch 512`) and keeps the epoch with
the best PubMed-validation recall AUC (maps divided by their maximum, every 20 epochs). `runs/final`: best epoch 200,
validation recall AUC 0.948. `--recipe notebook` restores the notebook.

### Salience (Fig. 2)

Stage 14 trains on the anatomical head's dense synthesised maps (mean 0.20) but names binary atlas maps.
`--synth-binarize-pct 95` (default) divides each synthesised map by its maximum and binarises it at its 95th percentile
before the encoder, as NeuroVLM does for any continuous brain image: 1,106 of 1,338 canonical rows then pass the
self-consistency filter (858 without), all 16 Salience rows among them.

With that alone, the Salience title still depended on the seed: the Du SAL/PMN map was "Salience Network" in 2 of 5
seeds. The 14 Salience training rows are renderings of near-identical texts (Dice 0.87). Training projected their
image semantics onto the all bank, where "Salience Network" got 0.076 of the weight, while `[NETWORK]` inference uses
the network bank; and the Du map's autoencoder latent is slightly closer to the Frontoparietal prototype than to the
Salience one (cosine 0.896 vs 0.877). The current defaults train against the row's own level bank (`--train-bank
level`), re-binarise each training map per batch at a percentile drawn from 90-97 with 10 % voxel dropout in half the
maps (`--map-aug`), and weight title tokens 5 (`--title-weight 5.0`); `--train-bank all --no-map-aug --title-weight 1`
is the previous recipe.
In a ten-seed test before the change, the Du map was titled "Salience Network" in 10 of 10 seeds, and the 119-map title
accuracy went from 0.743 (five seeds, previous recipe) to 0.760 (ten seeds).

| Fig. 2 row | network title, `runs/final` | previous recipe |
|---|---|---|
| Visual | Object Perception Network | Visual System |
| Language | Language Network | Language Network |
| Auditory | Auditory Network | Auditory Network |
| Somatomotor | Somatosensory Network | Somatosensory Network |
| Interoception | Interoceptive Network | Introspective Network |
| Default Mode | Default Mode Network | Default Mode Network |
| Frontoparietal Control | Frontoparietal Control Network | Frontoparietal Control Network |
| Salience (Du SAL/PMN) | Salience Network | Frontoparietal Network |

Exact reference names: 6 of 8 (7 counting Somatosensory for the somatomotor row), previously 4 (5). All eight now name
the right system. The Salience row reads "Salience Network / Anterior Cingulate Cortex / Executive Control".

### Stage 11 split leak (fixed)

The notebook mines region n-grams over all 30,784 PubMed papers and pools their maps into the AAL parcel maps from which
the synthetic (text, map) pairs are composed. Those pairs train the anatomical head (12), which makes stage 14's training
latents, so validation and test maps entered training as parcel averages (1,623 val and 1,596 test of 16,346 pooled
papers). `--splits train` (default) keeps training papers only (23,894 papers, 14,768 pooled, 0 val / 0 test, asserted
and recorded in the summary); `--splits all` = notebook, outputs suffixed `_allsplits`. Stage 13 does not use 11.

| PubMed test (n = 2,971) | released | before the fix | 12 on `--splits all` maps | after the fix |
|---|---|---|---|---|
| anatomical head recall AUC | 0.7925 | 0.8120 | 0.7905 | 0.7894 |
| anatomical head Dice / Pearson | 0.4690 / 0.2748 | 0.4633 / 0.2771 | 0.4664 / 0.2788 | 0.4692 / 0.2817 |
| stage 2 projector recall AUC | 0.9396 | 0.9466 | - | 0.9448 |

The leak itself moves the anatomical head by <= 0.003; the larger before -> after drop comes from replacing the released
synthetic pairs with this port's. These numbers predate head v2 and calibration; final numbers are above.

## Projector choice

Stages 13 and 14 use a residual MLP projector (`--projector mlp`, default; `--projector qformer` = the notebook's
Q-Former): `LN(raw) ++ LN(semantic) -> Linear(768, W) -> D pre-LN residual blocks -> Linear(W, 32 x 1024) + per-query
embedding -> LayerNorm(1024)` (`ResidualMLPProjector` in `_common_models.py`), with the Q-Former's `forward(raw,
semantic)`, so `NeuroQFormer` and `generate_text` use it unchanged. `--mlp-width 704 --mlp-depth 2`: 28.34 M parameters
(Q-Former 28.46 M). fp32 master weights, best-validation epoch packaged (`--package-checkpoint auto`), early stopping
(`--early-stop-patience 3 --min-epochs 5`; the LR schedule still spans 30 epochs). Packages without
`config.json["projector"]` are Q-Formers (`_common_models.load_projector` builds either).

On the proof run's inputs (PubMed test n = 2,971) the Q-Former stays below the release at stage 2 and the MLP passes
it (reviewer analysis: [`figures/review/core/a5_mlp_projector/`](figures/review/core/a5_mlp_projector/README.md)):

| stage | model | recall AUC | contrastive cos | BERTScore F1 | MiniLM cos |
|---|---|---|---|---|---|
| stage 1 (`qformer_pubmed`) | released | 0.819 | 0.187 | 0.573 | 0.399 |
|  | Q-Former (proof) | 0.860 | 0.213 | 0.574 | 0.417 |
|  | MLP (proof_mlp) | 0.907 | 0.244 | 0.575 | 0.431 |
| stage 2 (`qformer`) | released | 0.940 | 0.266 | 0.557 | 0.468 |
|  | Q-Former (proof; five seeds 0.889-0.916) | 0.916 | 0.242 | 0.556 | 0.473 |
|  | MLP (proof_mlp) | 0.945 | 0.273 | 0.556 | 0.471 |

## Runtime

One RTX 5090 (32 GB), 16 cores, WSL2, caches populated, MLP projector. *measured* = a full run on an idle GPU; *est.* =
from clean per-epoch rates. Details: `scripts/outputs/time_estimates.json`.

| step | what | wall time | source |
|---|---|---|---|
| data | `fetch_hf.py`, ~23 GB on first use (`run_all.py --regenerate`, 01..07: 14 min, est.) | download | |
| 08 | autoencoder, 101 epochs (6.1 s/epoch) | 10 min | measured |
| 09 | notebook head (0.093 s/epoch) + head v2 + InfoNCE heads (0.017 s/epoch) | 46 s | measured |
| 20 | 5-fold CV (per fold 08 + heads); `--folds 0` = one fold, ~10 min | 48 min | measured |
| 11 | anatomical atlases (CPU, joblib) | 8-12 min | measured / est. |
| 12 | anatomical head, 2,000 epochs, anchored | 103 s | measured |
| 13 | stage 1 projector, MLP, early stop after 7 epochs (Q-Former, 30 epochs: 68 min) | 17 min | measured |
| 14 | stage 2 projector, 30 epochs, 8.3 s/epoch (per seed; previous recipe 4.2-4.5 min) | 5 min | measured |
| **training** | 08..14 with five stage-14 seeds (60 min) + 20 (48 min); data from Hugging Face, not regenerated | **1.8 h** | measured per stage |
| validation | all components, generation batch 128 | 19 min | measured |
| paper figures | `figures/main/make_all.py`, cold caches (Fig. 4: 13 min) | 21 min | measured |
| smoke | `smoke_pipeline.sh --device cpu`, cached captions and reference (cold: 35-40 min) | 7 min | measured |

Keep one GPU job at a time: on WSL2 a second large CUDA process on a full card slows both down 5-20x instead of failing.
Cached between runs: stage 13 tokens and stage 11 blocks (`runs/<run>/cache/`), data products (`~/.cache/neurovlm`),
captions (`validation/<name>/df_gen_*.parquet`), figure panels (`figures/main/cache/`) and `validation/released_excl/`.
Regenerating the PubMed summaries with an LLM (`07_transform_text.py --regenerate`) adds 4.3 h (Qwen3-0.6B).

## Determinism

Every stage runs deterministic kernels by default (`common.setup_torch`): `torch.use_deterministic_algorithms(True)`,
`cudnn.deterministic = True`, `cudnn.benchmark = False`, seeded python/numpy/torch, seeded DataLoader generators and
worker init. `common.py` sets `CUBLAS_WORKSPACE_CONFIG=:4096:8` before torch loads; `train_pipeline.py` and
`reproduce.sh` pass it to every stage. TF32 matmul is unchanged (on). In stages 13 and 14 the frozen NeuroQwen3 runs
without cuDNN attention, so SDPA uses flash attention with its deterministic backward (`_common_models.lm_sdpa_backends`).
Stage 13 takes its seed from `--seed` (default 17). Stage 14's old `--deterministic` (TF32 off, the notebook's
precision) is now `--notebook-precision`. `--no-deterministic` turns all of this off. Cost: about 7-12 % per step in
stages 13 and 14; the runtimes of 08, 12, 13, 14, 20 and training above were measured with it (the two proof runs
of [REPRODUCTION.md](REPRODUCTION.md); 14: the 2026-10-10 retrain), the others before it became the default.

PyTorch guarantees bitwise identical results only on the same GPU architecture with the same driver and library
versions. `requirements.lock` and `scripts/Dockerfile` pin the libraries and `provenance.json` records the GPU and
driver. A different GPU model may give different bits.

## Reproducibility

`bash scripts/reproduce.sh [--cv-folds LIST] [--smoke] [--home DIR] [--data released|scratch] [--external-dir DIR]` builds
`.venv` from `requirements.lock` (CPython 3.12.12, torch 2.14.1 = CUDA 13.0 build, driver >= 580), fetches every input
with `fetch_hf.py`, runs 08 09 11 12 13 14 (five stage-14 seeds) -> 20 (five folds by default, ~9 min each) ->
validation -> the a3 NeuroVault bounds (Fig. 5b) -> figures, then `verify_reproduction.py` compares every primary metric
with runs/final (`reference_metrics.json`). `--home DIR` puts the data dir under `DIR` (the HF cache stays where it is).
`--data scratch` also reruns data stages 01-07 into `scripts/outputs/data_regenerated/` for provenance; training still
reads the pinned files. `--external-dir` takes optional local copies of inputs (linked in when missing, checked against
the manifest) and `nilearn_data/aal_SPM12/` (nilearn's AAL server fails TLS verification, 2026-10). Docker: `docker
build -f scripts/Dockerfile -t neurovlm-repro .` (mounts in the file).

The verifier calls a metric *exact* when |delta| <= 1e-4 (`--tol`) and FAIL otherwise, for every component, 13 and 14
included; there is no run-to-run allowance. PASS = every component exact and evaluated on the same items as runs/final.

`train_pipeline.py` writes `runs/<run>/provenance.json`: command line, git commit, `hf_manifest.json` sha256, pinned
revisions, Python/torch/CUDA/cuDNN/transformers versions, GPU, driver, determinism settings and per-stage exit codes.
`validate_models.py` writes the same record (without exit codes) to `metrics.json["provenance"]`.

Two full `reproduce.sh` runs with the current stage-14 recipe (e and f, 2026-10-10/11), each from a fresh copy of the
tree, a fresh `.venv` and an empty `--home`, with five CV folds and five stage-14 seeds, took 2 h 25 min and 2 h 22 min on
one RTX 5090. Both verifiers: 8/8 components exact.

| output | files | byte-identical, e vs f | byte-identical, e vs `runs/final` |
|---|---|---|---|
| model and data files of the run / of the five stage-14 seed runs | 77 / 20 | 77 / 20 | 77 / 20 |
| generated texts | 2 | 2 | 2 |
| figure PNGs | 9 | 9 | 8 of the committed PNGs; Fig. S3 is committed with the NiCLIP / NeuroConText baselines, which the runs did not have |
| validation metrics | 2,495 values | all metric values equal, BERTScore included | all metric values equal |

Against the previous, non-deterministic final, stages 08, 09, 11, 12 and 20 are unchanged (40/40 CV files byte-identical);
13 was retrained deterministically (recall AUC 0.8957, before 0.8969) and 14 with the current recipe (0.9544, before
0.9487). An earlier pair of runs with the previous stage-14 recipe (c and d, 2026-10-09/10) was byte-identical in the
same files apart from four BERTScore values (delta <= 6e-7, hash seed, since fixed with `PYTHONHASHSEED=0`). Details
and limits: [REPRODUCTION.md](REPRODUCTION.md).
