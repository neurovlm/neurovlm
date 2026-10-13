# Reproduction from scratch, 2026-10-12

Two full runs of `scripts/reproduce.sh` (g and h, 2026-10-12) with the final recipe, each from a fresh copy of the
working tree, a fresh `.venv` and an empty data dir, on one RTX 5090. The recipe:

- stage 14 as in [Stage 14 recipe change](#stage-14-recipe-change-2026-10-10);
- captions decoded with every non-Latin-script and byte-fragment token suppressed (README,
  [Generation](README.md#generation)), with validation at generation batch 128;
- `PYTHONHASHSEED=0` ([Hash seed](#hash-seed)).

Results:

- All 77 model and data files of the run, and the 20 of the five stage-14 seed runs, are byte-identical between g and h.
  They are also byte-identical to `runs/final` and `runs/final_s1` ... `_s5`.
- `metrics.json` is identical between g and h in every value, BERTScore included. It equals `validation/final` in every
  metric value, and `validity_frac_non_latin` is 0 for both captioning stages.
- The generated captions are byte-identical across g, h and `validation/final`.
- Both verifiers report 8/8 components exact.
- The 13 PNG pages of the nine figures are byte-identical between g and h. 12 of the 13 equal the committed PNGs; Fig. S3
  differs by design ([Figures](#figures)).

The two earlier proofs are kept below: runs e and f (2026-10-11), with the same models and captions from before the
token suppression, and runs c and d (2026-10-10), with the previous stage-14 recipe. Mechanisms: README
[Data](README.md#data), [Determinism](README.md#determinism), [Reproducibility](README.md#reproducibility).

**Generation recipe of 2026-10-13.** The generation recipe changed on 2026-10-13: [NETWORK] and [REGION] caption titles
are restricted to the canonical vocabulary ([Caption titles](README.md#generation)), and generated maps are scaled by one
global calibration factor per model ([Map calibration](README.md#map-calibration)). Runs g and h predate both changes, so
their caption, metric and figure comparisons describe the 2026-10-12 generation recipe. A repeated two-run proof under
the new recipe was interrupted by a machine restart before it finished. It will be repeated after the NeuroQwen3-0.6B
retrain (`scripts/models/00_neuroqwen_dapt.py`, then stages 13 and 14). The model-stage evidence of runs g and h is
unchanged, because no training stage changed on 2026-10-13.

## How the proof was run

```bash
(cd tree_g && bash scripts/reproduce.sh --home home_g --run-name final_v3 --external-dir external_aal)
(cd tree_h && bash scripts/reproduce.sh --home home_h --run-name final_v3 --external-dir external_aal)
```

| | runs g and h (h started after g had exited) |
|---|---|
| code | an rsync of the working tree (HEAD c660203 plus uncommitted changes) without `.venv`, `scripts/outputs`, `scripts/figures/main/cache`, the review analyses' `outputs`, `.git` and `__pycache__`. tree_h was copied from tree_g after run g, also without run g's figures. The sha256 manifests of tree_g before run g and of tree_h are identical (519 files), so run g changed no code file. `reproduce.sh` was unchanged |
| Python env | a new `.venv` per tree from `requirements.lock` (uv 0.9.26, CPython 3.12.12, shared uv package cache); `uv pip freeze` = lock |
| data dir | `--home` = an empty directory, so `~/.cache/neurovlm` started empty |
| inputs | `fetch_hf.py`: 119 manifest entries checked, 15 written to the data dir, 104 served by the shared `~/.cache/huggingface` at the pinned commit; 0 mismatches. HF offline after the fetch |
| external dir | `nilearn_data/aal_SPM12/` only ([Known limits](#known-limits)) |
| CV folds / stage-14 seeds | 0-4 / 1-5 (defaults); stage 14 with the current recipe (`--train-bank level`, `--map-aug`, `--title-weight 5`) |
| generation | non-Latin-script and byte-fragment tokens suppressed (`generation_config.suppress_tokens`); validation with `--generation-batch 128`, as `reproduce.sh` passes |
| environment | `reproduce.sh` exports `CUBLAS_WORKSPACE_CONFIG=:4096:8`, `PYTHONHASHSEED=0` and `TOKENIZERS_PARALLELISM=false`; no other HF, CUDA, torch or nilearn variable is set |
| hardware, libraries | RTX 5090 32 GB, driver 591.44, WSL2 (Linux 6.6.87.2); torch 2.14.1+cu130, CUDA 13.0, cuDNN 9.24.0, transformers 4.57.6, huggingface_hub 0.36.2, numpy 2.5.3. The software, HF revision and determinism records in `provenance.json` are the same in g, h and `runs/final` |

| step (`timings.tsv`) | run g (s) | run h (s) |
|---|---:|---:|
| env (create + install) | 1 | 2 |
| data (`fetch_hf.py`) | 90 | 185 |
| training 08 / 09 / 11 / 12 / 13 | 639 / 57 / 148 / 117 / 1,158 | 730 / 66 / 170 / 125 / 1,222 |
| training 14 (five seeds) | 1,412 (275-289 per seed) | 1,440 (283-297 per seed) |
| training 08-14 | 3,468 | 3,682 |
| training 20 (five folds) | 2,898 | 3,024 |
| validation | 1,082 | 1,071 |
| a3 NeuroVault bounds | 31 | 29 |
| figures 2, 3, 4, 6, S2, S3, S4 | 1,222 | 1,162 |
| Fig. 5 (panels a-d) | 88 | 83 |
| total | 8,899 (2 h 28 min) | 9,256 (2 h 34 min) |

Every step exited 0, and so did `reproduce.sh`. During run h's fetch and stage 08, a low-priority disk comparison of
run g with `runs/final` was running. Stage-14 validation losses at epoch 30, seeds 1-5, were the same in both runs:
1.31444, 1.32248, **1.30051**, 1.31436 and 1.31858. Seed 3 was packaged in both.

## Results

*identical* = `cmp` equal. *values equal* = equal after loading (tensors with `torch.equal`, floats exactly).

| category | files | g vs h identical | g vs `runs/final` identical | the others |
|---|---:|---:|---:|---|
| `runs/final_v3` model and data files (.pt 45, .safetensors 28, .parquet 2, .jsonl 2) | 77 | 77 | 77 | |
| - stage 08 / 09 / 11 / 12 | 4 / 8 / 4 / 2 | all | all | |
| - stage 13 (12 checkpoints, `model.safetensors`, `probes.jsonl`, tokens cache) | 15 | 15 | 15 | |
| - stage 14 (`model.safetensors`, epoch-29 checkpoint, banks, `generations_refs.jsonl`) | 4 | 4 | 4 | |
| - stage 20 (five folds x 8) | 40 | 40 | 40 | |
| stage-14 seed runs `_s1` ... `_s5` (4 weight and data files each) | 20 | 20 | 20 (vs `final_s1` ... `_s5`) | |
| `cache/*.pkl` | 4 | 3 | 3 | `atlas_aal_spm12.pkl`: 16 / 18 bytes, values equal. The bytes are the legacy torch storage keys (object ids) of the pickled tensors |
| `training_runs/*/checkpoints/*.pt` | 33 | 0 | 0 | zip root directory only (torch's random atomic-save temp name); member lists equal, all 1,302 members identical |
| other `training_runs` files | 198 | 132 | 132 | 66 = 11 runs x 6 JSON files: clock fields, output path, executable, the sha256 of the 33 containers, `config_sha256` (the config holds the output path) |
| run JSON and stage logs (summaries, histories, lineage, configs, provenance, timings) | 27 | 0 | 0 | clock fields, epoch seconds and absolute paths only; g vs final also the provenance command history |
| `validation/final_v3` generated texts | 2 | 2 | 2 (vs `validation/final`) | |
| `validation/final_v3/metrics.json` | 1 | 0 | 0 | clock fields and paths; every metric value equal (below) |
| a3 NeuroVault bounds | 12 | 10 | 10 (vs the committed a3 outputs) | `config.json`, `timings.json`: paths and wall times |
| `figures/main/cache` | 16 | 1 | not compared (other run tag) | 11 `.pt` values equal (the file name carries the run tag); 2 `.pt` differ only in an absolute path; `model_sources` in its digest; `timings.jsonl` clock |

`metrics.json` has 2,510 values. Run g vs run h: 2,504 equal. The other 6 are the two `generation_seconds`, two
timestamps, the provenance command and the tree path of the exclusion list. Run g vs `validation/final`: 2,459 equal.
The other 51 are paths (`final_v3` vs `final`), timestamps, `generation_seconds`, the provenance command and the git
commit (the tree copies have no `.git`). No metric value differs. Bitwise:

| metric | run g | run h | `validation/final` |
|---|---:|---:|---:|
| stage 13 BERTScore F1, matched | 0.5770239233970642 | identical | identical |
| stage 13 BERTScore F1, null | 0.569153368473053 | identical | identical |
| stage 14 BERTScore F1, matched | 0.5524473190307617 | identical | identical |
| stage 14 BERTScore F1, null | 0.5473654270172119 | identical | identical |
| `validity_frac_non_latin`, stages 13 / 14 | 0 / 0 | identical | identical |

`verify_reproduction.py` compared each run with `reference_metrics.json` at a tolerance of 1e-4. Runs g and h both
report "reproduction PASS: 8/8 components exact" (ae, heads, networks, text_to_brain, adapter, qformer_pubmed,
qformer, cv), with every metric delta at +0.00000. The two `verify.log` files are identical once tree paths and figure
mtimes are removed. `validate_models.py` against the released checkpoints: 8/8 PASS in both runs.

### Figures

| figures, run g vs run h | files | identical | the others |
|---|---:|---:|---|
| PNG (nine figures; Fig. 2 four pages, Figs. 3 and S4 two each) | 13 | 13 | |
| SVG | 13 | 0 | equal after removing `dc:date` and the random element ids; embedded rasters identical |
| `*_text.json` | 8 | 0 | `cache_tag` and two absolute paths (Fig. 4 generation cache, Fig. S3 `external_root`) |

All nine figures were drawn in both runs. Fig. 5 has all four panels: a from the five CV folds, b from the run's a3
bounds.

The committed PNGs in `scripts/figures/main/` were redrawn on 2026-10-12 at the template widths (`FIG_FULL_W` =
5.15 in for the journal, `FIG_LETTER_W` = 6.5 in for the response letter, `scripts/common.py`). 12 of the 13 equal
runs g and h byte for byte:

| PNG | md5 (runs g and h) | committed |
|---|---|---|
| `fig2_network_examples` | `b0a32ca2641a83a4cc59daa0e1759284` | same |
| `fig2_network_examples_p2` | `857032e859a250475d7faf8724d50022` | same |
| `fig2_network_examples_p3` | `cc558179549088e8a634ff93d6279f1e` | same |
| `fig2_network_examples_p4` | `852974b5eed315a1f2e213f23e07c645` | same |
| `fig3_newgen` | `ab6badad305cc33cc26224dbbd703b6d` | same |
| `fig3_newgen_p2` | `1e6783023ee0fec0a4348b53f5b85514` | same |
| `fig4_generative_metrics` | `ad419d3c35f0210ee22becd55bf46065` | same |
| `fig5_perf_quant` | `94d97cd461e625a6d018a5b1c60565e7` | same |
| `fig6_autoencoder_qual` | `e83af840c93fa1f3e0e55587802614fc` | same |
| `figS2_difumo_v_ale` | `0e3fa64bf26a13f4e72b62db3fb6498e` | same |
| `figS3_recall_comparison` | `3fc6144e48f3b5ce277f32735e23c969` | `099bffa57df231049772bedc47a6d893`, by design: the committed figure adds the external NiCLIP and NeuroConText baselines (from `~/neurovlm`) on their fair subset (n = 366). The runs had no external checkout and draw NeuroVLM only (n = 2,971) |
| `figS4_clip_ica` | `3e1f38a74a3c0c8b2c69909a04ac89ef` | same |
| `figS4_clip_ica_p2` | `eab1cd5f16091ee298392a43200a3475` | same |

**Cache guard.** A metadata listing (path, type, size, mtime) of the author's `~/.cache/neurovlm` was identical before
run g, after run g and after run h (10,450 entries, 70,342,456,752 bytes). `runs/final`, `validation/final`, `src/`
and the working tree were only read.

### Hash seed

`validate_models.bertscore_f1_many` and `bert_score.score` order the texts by `sorted(set(texts), key=word count)`.
Ties follow set iteration order, which depends on Python's per-process hash seed. Batch composition and padding change,
and with them the last bits of F1. In runs c and d, four BERTScore values differed by up to 6e-7
([below](#earlier-record-runs-c-and-d-2026-10-10)). Fix: `PYTHONHASHSEED=0` in `reproduce.sh` and the launchers. Runs e
and f and runs g and h confirm it end to end: within each pair, the four values are bitwise equal to each other and to
`validation/final` at the time.

## Previous-recipe record: runs e and f (2026-10-11)

The second two-run proof ran before the token suppression. The procedure was the same: `--run-name final_v2`, trees
`tree_e` and `tree_f`, homes `home_e` and `home_f`, HEAD 51bedf4 plus uncommitted changes (452 files), the current
stage-14 recipe and `PYTHONHASHSEED=0`.

- All 77 model and data files and the 20 seed-run files were byte-identical between e and f, and to `runs/final`.
- The suppression does not touch the trained models: no stage trains through generated captions, and stage 14's probe
  captions do not use it.
- `metrics.json` was identical between e and f in every value, and equal to the `validation/final` of that time in
  every metric value.
- Both verifiers reported 8/8 exact.
- The nine figure PNGs (previous layouts) were byte-identical between e and f.

Wall times e / f: total 8,692 / 8,533 s (2 h 25 / 22 min); training 08-14 3,377 / 3,399 s; training 20 2,847 / 2,850 s;
validation 923 / 936 s; figures with Fig. 5 1,192 / 1,155 s. Their BERTScore F1 values, on the unsuppressed captions,
were 0.5770103931427002 / 0.5691438317298889 for stage 13 matched / null and 0.5519694685935974 / 0.54689621925354 for
stage 14.

The suppression then changed the captions and caption metrics. Stage 2:

- recall AUC 0.9544 -> 0.9542
- contrastive 0.2823 -> 0.2820
- BERTScore F1 0.5520 -> 0.5524
- MiniLM 0.4622 -> 0.4625

Stage 1 moved by at most 2e-5, and every other component is unchanged. `validation/final` and `reference_metrics.json`
were regenerated with the suppression, and runs g and h reproduce them.

## Earlier record: runs c and d (2026-10-10)

The first two-run proof, with the previous stage-14 recipe (`--train-bank all --no-map-aug --title-weight 1`) and a
random hash seed. Same procedure as above (`--run-name final_det`, trees `tree_c` / `tree_d`, homes `home_c` /
`home_d`); `fetch_hf.py` checked 109 manifest entries (15 written, 94 from the shared HF cache, 0 mismatches). Run c's
outputs became `runs/final` and `validation/final`; the previous, non-deterministic final is `runs/final_nondet_old`.
Wall times c / d: total 8,751 / 8,691 s (2 h 26 / 25 min); training 08-14 3,386 / 3,353 s, 20 2,908 / 2,868 s,
validation 1,039 / 1,015 s, figures with Fig. 5 1,274 / 1,255 s. Every step exited 0; `reproduce.sh` exited 1 in both
runs because the verifier compared with the previous final's `reference_metrics.json` and stages 13 and 14 were
retrained.

| run c vs run d | files | identical | the others |
|---|---:|---:|---|
| `runs/final_det` model and data files / seed runs `_s1` ... `_s5` | 77 / 20 | 77 / 20 | |
| `runs/final_det/cache/*.pkl` | 4 | 3 | AAL atlas pickle: legacy torch storage keys (object ids); values equal |
| `training_runs/*/checkpoints/*.pt` | 33 | 0 | zip root directory name (torch's random temp name); members identical |
| other `training_runs` files, run JSON | 198, 29 | 132, 0 | clock fields, paths, sha256 of the 33 containers |
| `validation/final_det` (generated texts, `metrics.json`) | 3 | 2 | `metrics.json`: four BERTScore values |
| figure PNGs / SVGs / `*_text.json` | 9 / 9 / 8 | 9 / 0 / 0 | SVG save date and element ids; `cache_tag`; Fig. 4 BERTScore (Welch t 8.4613 vs 8.4608) |

`metrics.json`: 2,483 of 2,491 values equal; the other 8 are four BERTScore F1 values and their copies in `comparisons`:

| metric | run c | run d | delta |
|---|---:|---:|---:|
| stage 13 `bertscore_f1_matched` | 0.5770110488 | 0.5770104527 | -6.0e-7 |
| stage 13 `bertscore_f1_null` | 0.5691432953 | 0.5691429377 | -3.6e-7 |
| stage 14 `bertscore_f1_matched` | 0.5524411201 | 0.5524407625 | -3.6e-7 |
| stage 14 `bertscore_f1_null` | 0.5474722982 | 0.5474724770 | +1.8e-7 |

The `~/.cache/neurovlm` listing was identical before and after each run (10,450 entries, sha256 prefix
`ec931ea2bf381411`). Stage-14 sha256 in runs c and d: `neuro_qformer_canonical/model.safetensors`
`411e7b2c965cfdc98190ee93391924a3a1cb399b2c2ea3c1a2e677679ab908cc`.

### Comparison with the previous final

| stage | byte-identical to the previous final | the rest |
|---|---:|---|
| 08 autoencoder, 12 anatomical head, 20 PubMed CV (five folds) | 4/4, 2/2, 40/40 | |
| 09 projection heads | 7/8 | `proj_head_text_bce.pt`: same tensors, other container |
| 11 anatomical atlases | 7/7 data files | AAL pickle (torch storage keys) and lineage (paths) |
| 13 stage-1 projector | tokens cache only | retrained; the previous final was not deterministic |
| 14 stage-2 projector | banks only | retrained; the seed selection now picks seed 3 (before: 4) |

| stage 13 / 14 metrics | recall AUC | contrastive sim | BERTScore F1 | semantic sim |
|---|---:|---:|---:|---:|
| 13, previous final | 0.8969 | 0.2357 | 0.5752 | 0.4246 |
| 13, runs c and d | 0.8957 | 0.2339 | 0.5770 | 0.4310 |
| 14, previous final | 0.9487 | 0.2752 | 0.5529 | 0.4653 |
| 14, runs c and d | 0.9486 | 0.2744 | 0.5524 | 0.4654 |

The other six components (08, 09 heads, networks, text to brain, 12, 20 over five folds) are exact (|delta| <= 1e-4).
With the previous recipe, stage 14 packaged seed 3 (final validation loss 1.3183). With identical weights, the previous
final's 08 and 20 summaries differ in voxel AUROC by 1-2e-6 (08: 0.97405010, now 0.97404897): they were evaluated on
2026-10-04/05, before deterministic mode. Between c and d these summaries match exactly.

## Stage 14 recipe change (2026-10-10)

After runs c and d, stage 14 in `runs/final` was retrained with a new recipe, on run c's inputs. Runs e and f and runs g
and h (above) reproduce the result end to end.

**Why.** With the previous recipe the Fig. 2 Salience title depended on the seed. The Du SAL/PMN map was titled
"Salience Network" in 2 of 5 seeds; the packaged seed 3 printed "Frontoparietal Network". The 14 Salience training rows
are synthetic renderings of near-identical texts (Dice 0.87). Training projected their image semantics onto the all
bank, where "Salience Network" got 0.076 of the weight, while `[NETWORK]` inference uses the network bank. The Du map's
autoencoder latent is slightly closer to the Frontoparietal prototype than to the Salience one (cosine 0.896 vs 0.877).
In a ten-seed test of the new recipe, the Du map was titled "Salience Network" in 10 of 10 seeds.

**What changed** (`scripts/models/14_qformer_anat.py`; the new behaviour is the default, `--train-bank all --no-map-aug
--title-weight 1` gives the previous recipe):

| flag | default | effect |
|---|---|---|
| `--train-bank level\|all` | `level` | training semantics (image and text) projected onto the row's own level bank, as at inference; before: the all bank |
| `--map-aug / --no-map-aug` | on | each training map re-binarised per batch at a percentile drawn from U(90, 97); half the maps lose 10 % of their voxels; draws only from `torch.Generator(device).manual_seed(seed + 202)`; validation not augmented |
| `--title-weight` | `5.0` | title tokens weigh 5x in the training LM loss; the validation loss is unweighted |

Unchanged: five seeds, the lowest final validation loss packaged, 30 epochs, the split seed, the packaged files,
determinism and HF pinning. Command: `train_pipeline.py --only 14 --run-name final --seeds-14 1,2,3,4,5 --force
--skip-data`; 1,502 s for five seeds (296-306 s each; before 251-271 s). Seed 3 was selected again:

| seed | 1 | 2 | **3** | 4 | 5 |
|---|---:|---:|---:|---:|---:|
| val loss, epoch 30 | 1.3144 | 1.3225 | **1.3005** | 1.3144 | 1.3186 |
| title accuracy, 119 network maps | 0.739 | 0.765 | **0.748** | 0.765 | 0.798 |
| Du SAL/PMN title | Salience | Salience | **Salience** | Salience | Salience |

Five-seed mean title accuracy 0.763 (previous recipe 0.743). Validation losses are not comparable with the previous
recipe's, because the training inputs differ.

**Determinism.** After the change, seed 3 was trained three times: in the sweep (`runs/final_s3`, packaged into
`runs/final`) and twice more with the pipeline's per-seed command (`runs/det14_a`, `runs/det14_b`, deleted afterwards).
These files are byte-identical across the three, and runs e, f, g and h gave the same bytes from scratch; the histories differ
only in `epoch_seconds`, `config.json` only in the absolute `qformer_source` path.

| file | sha256 |
|---|---|
| `neuro_qformer_canonical/model.safetensors` | `eaf5ee494e96e40f9c75f2b9bd4aed0edcbe4589b8fafe9d0365402a5515a5a9` |
| `qformer_canonical_bottleneck_epoch-29.pt` | `f9307abcc041b50b216b0890dc5e4c0eab281ddf10df46ff690cc5ba0eab377a` |
| `canonical_semantic_banks.pt` (unchanged from runs c and d) | `f28293a0595afab0a137ba8cc96b8023b5562e7b464f7071a3d5230da7f20771` |
| `generations_refs.jsonl` | identical (`cmp`) |

**Validation** (`validation/final`, 8/8 PASS vs released; `verify_reproduction.py` 8/8 exact against the rebuilt
`reference_metrics.json`):

| stage 14 metric | runs c and d (previous recipe) | `runs/final` (new recipe) | delta |
|---|---:|---:|---:|
| recall AUC | 0.9486 | 0.9544 | +0.0058 |
| contrastive sim | 0.2744 | 0.2823 | +0.0078 |
| BERTScore F1 | 0.5524 | 0.5520 | -0.0005 |
| semantic sim (MiniLM) | 0.4654 | 0.4622 | -0.0031 |

The other components are unchanged. Stage 13's captions are byte-identical and its BERTScore F1 moved by -6.6e-7
([Hash seed](#hash-seed): run c had a random hash seed; `validation/final` then equalled runs e and f bitwise).
`reference_metrics.json` held the new stage-14 values. Both were later regenerated with the token suppression
([runs e and f](#previous-recipe-record-runs-e-and-f-2026-10-11)); the table above shows the captions before it.

## Stage-level determinism checks

| check (single stages before the full runs, previous final's inputs, `runs/det*`) | result |
|---|---|
| stage 13, seed 17: det13_1 vs det13_2 | `model.safetensors`, all 12 checkpoints, `probes.jsonl` identical; history identical except epoch_seconds |
| stage 14, seed 4: det14_1 vs det14_2 | epoch-29 checkpoint, `model.safetensors`, banks, `generations_refs.jsonl` identical; history as above |
| stages 08, 09, 12, 20 fold 0 (one run each) vs the previous final | weights identical (09: `proj_head_text_bce.pt` same tensors) |
| run c stage 13 vs det13_1 | 20/20 files identical |

| cost | non-deterministic | deterministic | change |
|---|---:|---:|---:|
| stage 13, time per epoch / median step | 120 / 0.154 s | 135 / 0.165 s | +12 / +7 % |
| stage 14 seed 4, training | 189 s | 205.5 / 210.2 s | +9-11 % |

Deterministic stage 13 stops early after 7 epochs instead of 8, so its training time stays near 1,000 s.

## Before determinism

Stage 13 trained repeatedly with the same seed (17) and the same inputs on this machine, before deterministic mode:

| training | recall AUC | best val loss | training time |
|---|---:|---:|---:|
| previous final (2026-10-08) | 0.8969 | 1.9450 | 2,149 s |
| repeat 1 | 0.8993 | 1.9466 | 1,001 s |
| repeat 2 | 0.9020 | 1.9449 | 1,013 s |
| fresh-environment run B (2026-10-09, one CV fold) | 0.8797 | 1.9450 | 1,043 s |

Recall AUC spread over 0.022 (sample std 0.010, n = 4); the validation loss moved by less than 0.002 and the epoch-0
checkpoints already differed by up to 0.03. Re-scoring a checkpoint gave the same AUC and generations, so the spread was
in training. Source (SDPA probe, frozen NeuroQwen3): flash attention's backward on batches without padding, 5 distinct
gradients in 5 repeats; with deterministic algorithms, 1. The verifier's stage-13 allowance of 0.025 AUC was removed.

## Data provenance

Training reads only the pinned Hugging Face files ([Data](README.md#data)); `--data scratch` reruns stages 01-07 as
provenance. The released maps were built with nimare <= 0.15, whose `mm2vox` truncates MNI coordinates to voxel
indices; nimare 0.22 rounds them. On the 4 mm mask 66 % of the foci then move by one voxel (ALE rows r ~0.63), 48
instead of 42 studies get an empty map, and in the 2026-10-09 run A the stage-08 split changed.
`01_coordinate.py --ijk-mode trunc` (default) sets the voxel indices itself: its ALE maps equal the released ones bit
for bit (1,846 rows checked) and their DiFuMo projection matches the released tensor to 1.3e-4 (CPU vs GPU).

## Known limits

- Shown on one RTX 5090 with driver 591.44 and the locked libraries. PyTorch guarantees bitwise identity only on the
  same GPU architecture, driver and library versions; a different GPU may give different bits. Not tested.
- nilearn's AAL server fails TLS verification (2026-10), so the AAL SPM12 atlas came from `--external-dir`.
- The proof runs drew Fig. S3 without the third-party baselines (NiCLIP / NeuroConText), which need an external
  checkout (`--external-root`); the committed Fig. S3 is drawn with them.
- `neurovault_collections.parquet` was published with e-mail addresses redacted, so an author's unredacted local copy
  is a `fetch_hf.py` mismatch (use `--overwrite` or an empty `--home`). The proof runs did not hit this.
- The HF cache was warm, so the first download (~23 GB) is not in the wall times.

## How to reproduce

```bash
bash scripts/reproduce.sh --home "$(mktemp -d)" --external-dir DIR    # DIR/nilearn_data/aal_SPM12/; run name: repro
(cd scripts/outputs/runs/repro && sha256sum autoencoder.safetensors neuro_qformer_*/model.safetensors autoencoder_kfold-4.safetensors)
cat scripts/outputs/reproduce/repro/verify.log                       # metrics vs reference_metrics.json
```

| file | sha256 in runs g and h (and e, f) and `runs/final` (expected on the same GPU model, driver and libraries) |
|---|---|
| `autoencoder.safetensors` (08) | `151d62eebc805ea79007415355c8b26069af7d9c8ee545b7fddb6a5922731f55` |
| `neuro_qformer_pubmed/model.safetensors` (13) | `4fc17243c5514021ca8d8543112826107da3c87899531c279698531bbeb5cb37` |
| `neuro_qformer_canonical/model.safetensors` (14) | `eaf5ee494e96e40f9c75f2b9bd4aed0edcbe4589b8fafe9d0365402a5515a5a9` |
| `autoencoder_kfold-4.safetensors` (20) | `c07c28c515fdd3d29466323bfdcb8dcb219ad5277527f67219e65392b9ca36e1` |
