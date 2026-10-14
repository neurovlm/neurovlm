# `scripts/models/`: training stages

One script per training stage, ported from the `docs/03_models` notebooks (00 from efficient_ai, 20 from
`docs/figures/20`). `scripts/pipeline/train_pipeline.py` runs them in order (see [`../README.md`](../README.md)). Shared
code is in `_common_models.py` (model loading, projector, caption decoding, map calibration),
`_anatomical_atlases_lib.py` (stage 11) and `_qformer_anat_data.py` (stage 14). Each stage writes into
`scripts/outputs/runs/<run>/` and takes `--smoke`, `--force`, `--seed` and `--no-deterministic`.

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
| 14 | 03_models/14_qformer_anat | neuro_qformer_canonical/, canonical_semantic_banks.pt, qformer_canonical_bottleneck_epoch-29.pt, generations_refs.jsonl | stage 2 projector (canonical); synthesized maps max-scaled and binarized; training semantics on the row's own level bank, per-batch map augmentation, title tokens weighted 5; see [Salience](#salience-fig-2) |

Stages 13 and 14 keep the `qformer` file names (scripts, checkpoints, packages, validator components `qformer_pubmed`
and `qformer`) whichever projector they hold; the kind is in the package's `config.json["projector"]`.

Differences from the notebooks:
* 11 uses training-split papers only (`--splits all` = notebook) and writes `synth_text.parquet` instead of overwriting
  its input `synth.parquet`.
* 12, 13 and 14 take the heads, adapter and grounded projector from the run directory when present (12 `--base
  released`, 13 `--heads released`, 14 `--grounded-qformer`/`--adapter <file>` restore the notebook wiring). With run
  heads, 13 also reads the run's `latent_neuro.pt` (`--latents auto`); the notebook's HF latents only match the released heads.
* `synth.parquet` (11) and `text_synth_less.parquet` (14) were generated before this tree (an LLM prompt,
  `scripts/data/08_synthetic_texts.py` repeats it as a proof of concept); they are published as `neurovlm/synthetic_pairs`.

## Stage 00: NeuroQwen

`00_neuroqwen_dapt.py` retrains the text decoder, NeuroQwen3-0.6B: continued pretraining of `Qwen/Qwen3-0.6B` on
PubMed titles and abstracts, ported from efficient_ai `project/notebooks/01_dapt.ipynb` (the recipe on the model card).
The corpus is `pubmed_neuroimaging.parquet`, 1,231,613 PubMed records from the neuroimaging query of efficient_ai
`project/scrape.py`. It is not on the Hugging Face Hub, so `../config/hf_manifest.json` pins it by sha256 under `not_on_hub` and
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
plus deterministic kernels, fp32 master weights and the 23 validation passes. `../pipeline/smoke_pipeline.sh` runs the smoke.
Stages 13 and 14 still load the published model from the Hub.

## Text-to-brain head v2

The notebook's text head (`ProjHead(768, 512, 384)`, BCE against the raw map through the frozen decoder) lost to a
ridge regression from SPECTER2 to the autoencoder latent (validation Dice 0.489 vs 0.524): trained only through the
decoder, it lands off the latent manifold and decodes to near-average maps. Head v2 (same module): loss = latent MSE + 1.0 (1 - r of decoder logits) + 0.3 (1 - r of sigmoid
maps) against the decoded true latent; input noise `l2n(x + 0.5 N(0, I)/sqrt(768))`; output bias initialized at the mean
training latent; AdamW (lr 1e-3, wd 0.01), batch 512, up to 80 epochs, the epoch with the best validation Dice kept.

| model (validation, n = 2,961) | Dice | top-10 % Dice | Pearson |
|---|---|---|---|
| ridge, alpha 0.3 | 0.5242 | 0.2746 | 0.3067 |
| notebook head | 0.4891 | 0.2553 | 0.3166 |
| **head v2 (stage 09 in `runs/final`)** | **0.5322** | **0.2808** | **0.3173** |

Test (once, n = 2,971; `figures/review/core/a1_text_to_brain_baselines`): Dice / top-10 % Dice / Pearson 0.531 / 0.281
/ 0.322 for head v2, 0.526 / 0.275 / 0.311 for ridge, 0.492 / 0.256 / 0.321 for the notebook head. Stage 09 still trains
the notebook head (`proj_head_text_bce`) to initialise the InfoNCE text head; the CV models (20) keep the notebook recipe.

## Stage 11 split leak (fixed)

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
synthetic pairs with this port's. These numbers predate head v2 and calibration; the final numbers are in [`../README.md`](../README.md#final-run).

## Anatomical head v2 (stage 12, anchored)

With the notebook recipe, the anatomical head started from head v2 drifts away from PubMed while fitting the synthetic
pairs (PubMed-validation recall AUC 0.911 at epoch 1, 0.783 at epoch 2,000; release 0.792). The anchored recipe
(`--recipe anchored`, default) keeps the notebook's pairs, model, optimizer, schedule and 2,000 epochs, adds head v2's
loss on 512 random PubMed training pairs per step (`--anchor-weight 1.0`, `--anchor-batch 512`) and keeps the epoch with
the best PubMed-validation recall AUC (maps divided by their maximum, every 20 epochs). `runs/final`: best epoch 200,
validation recall AUC 0.948. `--recipe notebook` restores the notebook.

## Projector choice

Stages 13 and 14 use a residual MLP projector (`--projector mlp`, default; `--projector qformer` = the notebook's
Q-Former): `LN(raw) ++ LN(semantic) -> Linear(768, W) -> D pre-LN residual blocks -> Linear(W, 32 x 1024) + per-query
embedding -> LayerNorm(1024)` (`ResidualMLPProjector` in `_common_models.py`), with the Q-Former's `forward(raw,
semantic)`, so `NeuroQFormer` and `generate_text` use it unchanged. `--mlp-width 704 --mlp-depth 2`: 28.34 M parameters
(Q-Former 28.46 M). fp32 master weights, best-validation epoch packaged (`--package-checkpoint auto`), early stopping
(`--early-stop-patience 3 --min-epochs 5`; the LR schedule still spans 30 epochs). Packages without
`config.json["projector"]` are Q-Formers (`_common_models.load_projector` builds either).

On the proof run's inputs (PubMed test n = 2,971) the Q-Former stays below the release at stage 2 and the MLP passes
it (reviewer analysis: [`figures/review/core/a5_mlp_projector/`](../figures/review/core/a5_mlp_projector/README.md)):

| stage | model | recall AUC | contrastive cos | BERTScore F1 | MiniLM cos |
|---|---|---|---|---|---|
| stage 1 (`qformer_pubmed`) | released | 0.819 | 0.187 | 0.573 | 0.399 |
|  | Q-Former (proof) | 0.860 | 0.213 | 0.574 | 0.417 |
|  | MLP (proof_mlp) | 0.907 | 0.244 | 0.575 | 0.431 |
| stage 2 (`qformer`) | released | 0.940 | 0.266 | 0.557 | 0.468 |
|  | Q-Former (proof; five seeds 0.889-0.916) | 0.916 | 0.242 | 0.556 | 0.473 |
|  | MLP (proof_mlp) | 0.945 | 0.273 | 0.556 | 0.471 |

## Salience (Fig. 2)

Stage 14 trains on the anatomical head's dense synthesized maps (mean 0.20) but names binary atlas maps.
`--synth-binarize-pct 95` (default) divides each synthesized map by its maximum and binarizes it at its 95th percentile
before the encoder, as NeuroVLM does for any continuous brain image: 1,106 of 1,338 canonical rows then pass the
self-consistency filter (858 without), all 16 Salience rows among them.

With that alone, the Salience title still depended on the seed: the Du SAL/PMN map was "Salience Network" in 2 of 5
seeds. The 14 Salience training rows are renderings of near-identical texts (Dice 0.87). Training projected their
image semantics onto the all bank, where "Salience Network" got 0.076 of the weight, while `[NETWORK]` inference uses
the network bank; and the Du map's autoencoder latent is slightly closer to the Frontoparietal prototype than to the
Salience one (cosine 0.896 vs 0.877). The current defaults train against the row's own level bank (`--train-bank
level`), re-binarize each training map per batch at a percentile drawn from 90-97 with 10 % voxel dropout in half the
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

Flags of the current recipe (`14_qformer_anat.py`):

| flag | default | effect |
|---|---|---|
| `--train-bank level\|all` | `level` | training semantics (image and text) projected onto the row's own level bank, as at inference; before: the all bank |
| `--map-aug / --no-map-aug` | on | each training map re-binarized per batch at a percentile drawn from U(90, 97); half the maps lose 10 % of their voxels; draws only from `torch.Generator(device).manual_seed(seed + 202)`; validation not augmented |
| `--title-weight` | `5.0` | title tokens weigh 5x in the training LM loss; the validation loss is unweighted |

Stage 14 trains five seeds and packages the one with the lowest final validation loss (seed 3 in `runs/final`).
Validation split fixed by `--split-seed 123`; titles scored after the selection, never used for it (beam 5, maps
rendered as in Fig. 2); deterministic run:

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

## Generation

Captions come from the frozen, multilingual NeuroQwen3-0.6B. Reading the projector's soft prefix tokens instead of text,
it used to drift at low-margin steps into Chinese, Korean, Cyrillic or Thai tokens and byte fragments (165 of 2,971
stage-2 test captions held CJK characters, 1 at stage 1), although no training text holds such characters. Every script
that captions therefore sets `lm.generation_config.suppress_tokens = list(latin_script_suppress_tokens(tok))`
(`_common_models.py`; applied in `validate_models.Bundle.nvlm`, `figures/main/_shared.make_nvlm` and
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
after a [NETWORK] / [REGION] steering token to those names (`_common_models.py`): `title_vocabulary(run_dir)` matches
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

## Map calibration

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
`runs/final` was trained with); stage 14 divides each synthesized map by its maximum before binarizing it at a
percentile, where clipping would tie voxels at 1; the CV component keeps figures/20 cell 9's per-map maximum (the folds
have no validation split).
