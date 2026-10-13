# Paper figures (`scripts/figures/main/`)

One script per figure of the main text and supplement, drawn by default from the final retrained model
(`scripts/outputs/runs/final`).

## Run

```bash
cd neurovlm
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
.venv/bin/python scripts/figures/main/make_all.py                     # all eight -> here
.venv/bin/python scripts/figures/main/make_all.py --only fig2,figS4   # subset
.venv/bin/python scripts/figures/main/make_all.py --released          # released weights -> released/
.venv/bin/python scripts/figures/main/make_all.py --max-samples 64    # preview -> preview_n64/ (own caches)
.venv/bin/python scripts/figures/main/make_all.py --paper             # also copy the PNGs into paper/figures/
.venv/bin/python scripts/figures/main/fig4_generative_metrics.py --rows b    # one script, one row
```

`--model-dir scripts/outputs/runs/<run>` draws another run (into `<run>/`), `--force` recomputes the cached panel data.
Flags after `--` go to every script, e.g. `make_all.py --only fig4 -- --no-spin`; each script's `--help` lists its
flags.  A cold render of all eight takes ~21 min on one GPU (Fig. 4 ~13 min).

## Outputs

Each script writes `<stem>.png` (300 dpi), `<stem>.svg` and `<stem>_text.json` (every number and text behind the
panels, plus `model_sources`: the checkpoint file of each component) next to itself; panel caches go to `cache/`.

| paper figure | script | files | model components (from `runs/final`) |
|---|---|---|---|
| Fig. 2 networks: text-to-brain and brain-to-text | `fig2_network_examples.py` | `fig2_network_examples.*` (the figure, one page); `fig2_texts_table.{tex,md}` (supplementary table: complete input and generated texts; `--text-pages` also draws them as figure pages `_p2`-`_p4`) | autoencoder, text-to-brain head v2, stage 2 projector (MLP) |
| Fig. 3 NeuroVault examples | `fig3_newgen.py` | `fig3_newgen.*`, `fig3_newgen_p2.*` (second row of each group) | text-to-brain head v2, autoencoder, InfoNCE heads, stage 2 projector |
| Fig. 4 generative metrics (PubMed test) | `fig4_generative_metrics.py` | `fig4_generative_metrics.*` | row a: autoencoder + latents, head v2; row b: stage 2 projector, InfoNCE heads; deberta-xlarge-mnli / MiniLM scorers |
| Fig. 5 contrastive / autoencoder / networks | `fig5_perf_quant.py` | `fig5_perf_quant.*` | CV folds, autoencoder, InfoNCE heads; b: read from the a3 bounds analysis (same final components, checked) |
| Fig. 6 autoencoder reconstructions | `fig6_autoencoder_qual.py` | `fig6_autoencoder_qual.*` | autoencoder, split |
| Fig. S2 ALE vs DiFuMo | `figS2_difumo_v_ale.py` | `figS2_difumo_v_ale.*` | none (data only) |
| Fig. S3 model comparison | `figS3_recall_comparison.py` | `figS3_recall_comparison.*` | autoencoder, InfoNCE heads; NiCLIP / NeuroConText from their own checkpoints |
| Fig. S4 ICA components (HCP, UK Biobank) | `figS4_clip_ica.py` | `figS4_clip_ica.*` (UK Biobank), `figS4_clip_ica_p2.*` (HCP) | autoencoder, InfoNCE heads, stage 2 projector |

**Fig. 3 layout and labels.** Two pages in the layout of the original figure, 5.15 x 4.55 and 5.15 x 4.16 in at 7 pt
(the original, 6.7 x 3.9 in, used ~3.5 pt text): page 1 holds the first row of each group (papers 189, 101, 92), page 2
the second (149, 110, 74); each page's density strips mark its own three rows.  Maps: sagittal above axial views.  Quality = the row's two cosines (Image: paper text vs generated map, Text: generated
text vs true map, shared space); the thin strips on the left are the densities of both cosines over all 305 NeuroVault
papers (dashed: thirds) with each row marked in its group colour, and High / Medium / Low means both cosines lie in the
top / middle / bottom third.  True (text-to-brain) shows the complete title and the abstract's first lines; the
generated text is complete.  The pool is generated in batches of six, like the rows, because the bf16 generation
depends on the batch size (~2 min on a GPU).  Its captions use basis "all": the model writes the level tag itself,
and a [NETWORK] / [REGION] title after a written tag is held to the canonical names, as after a prompt tag
(`_common_models.install_title_constraint`).  Hand-picked rows 124, 152, 147 and 167 (P98 / P26, P61 / P88, P8 / P47,
P14 / P72) broke the rule and are replaced by the paper of their third closest to its centre (101, 110, 92, 74); the
Fig. 3 key numbers below describe the six rows drawn.  The generated and the true map of a row share their three cuts: per view the
slice that maximises the product of both maps' drawn activation summed over the slice (each map scaled to max 1), so
every row shows activation in both maps (JSON `rows[].cut_coords`).

`_shared.py` holds the shared code (CLI, caches, model loading, brain maps, text layout); `make_all.py` runs all eight.
SPECTER2 and NeuroQwen3-0.6B are frozen and always come from the release.

The PNGs here are the `runs/final` renders, redrawn on 2026-10-12 at the template widths (two CPU renders from the
caches were byte-identical to them, `scripts/REPRODUCTION.md`); Figs. 2, 4, 5, 6, S3 and S4 were redrawn from the same
caches on 2026-10-13 (CPU) for the recall-axis label, the axial views of Figs. 6 / S4 and the Fig. 2 text table.  Later on 2026-10-13, Figs. 2, 3, 4 and S4 were regenerated with canonical [NETWORK] / [REGION] titles and calibrated maps, and Figs. 2 and 3 draw every map at 50 % of its own maximum.  `cache/` holds the panel data of Figs. 2, 3, 4, S2 and S4; Figs. 5, 6
and S3 recompute theirs once (~3 min on a GPU).  Fig. 5b reads `../review/core/a3_neurovault_bounds/outputs/`
(`recall_curves.csv`, `bounds_metrics.csv`; `--bounds-dir`), the a3 run on the published map tensor (variant
`released`) with the final autoencoder, InfoNCE heads and exclusions; another model needs its own a3 run.

## Retrained weights (`--model-dir`)

`--model-dir` (default `scripts/outputs/runs/final`, the layout of `scripts/train_pipeline.py`) replaces every trained
component a figure uses; a missing file falls back to the released checkpoint with a warning (`_shared.RUN_FILES`; `.pt`
= whole module or state dict, `.safetensors` = state dict loaded into the released architecture):

| component | run-dir file | used by |
|---|---|---|
| autoencoder | `autoencoder.pt` / `.safetensors` | all except S2 |
| text-to-brain head (head v2 in `runs/final`) | `proj_head_text_mse.pt` | Fig. 2, Fig. 3 image column, Fig. 4 top row |
| InfoNCE heads | `proj_head_text_infonce.pt`, `proj_head_image_infonce.pt` | Fig. 3 (quality cosines, pairing), 4 (retrieval / contrastive), 5b/5d, S3, S4 |
| anatomical projection head (NeuroAdapter) | `adapter_anat_full.pt` | Fig. 3 image column with `--t2b-model adapter` |
| stage 2 projector (canonical; MLP in `runs/final`, `qformer` file names) | `neuro_qformer_canonical/` + `canonical_semantic_banks.pt` | Fig. 2, 3, 4 bottom row, S4 |
| stage 1 projector (grounded, PubMed) | `neuro_qformer_pubmed/` | none of the eight figures |
| 5-fold CV models | `{autoencoder,proj_head_text_infonce,proj_head_image_infonce}_kfold-{0..4}.pt` | Fig. 5a (else `data_dir`'s folds, with a warning) |
| PubMed split | `pmids_split.pt` | test split of Fig. 4, 5c, 6, S3 |
| true latents | `latent_neuro.pt` | Fig. 4 top-row target `sigmoid(decoder(latent))` |

The projector is loaded as in `scripts/validate_models.py` (`scripts/models/_common_models.neuro_projector_from_state`):
`config.json["projector"]` picks the Q-Former (absent or `qformer`) or the residual MLP (`mlp`).  Caches of a run carry
`_m-<run>-<digest>`, where the digest hashes which run file each component comes from (name, size, mtime);
`cache/model_sources_<tag>.json` decodes it.  Every generated map is multiplied by its model's global calibration factor
and clipped at 1 before it is drawn, encoded or scored (`--map-calibration global`, the default; `--rescale-max` divides
each map by its own maximum instead, see scripts/README.md "Map calibration"); map-panel caches carry `_cal-<rule>`.

## Evaluation exclusions

Every figure that evaluates or samples PubMed-test or NeuroVault items drops the leakage-audit items of
`scripts/eval_exclusions/` (`--no-exclusions` keeps them; caches of the default carry `_excl`):

| figure | what is dropped |
|---|---|
| Fig. 3 | the 7 NeuroVault papers that are PubMed training papers and their 29 maps (none of the six examples) |
| Fig. 4 | the 16 leakage-audit test-set papers from both rows (n = 2,971) |
| Fig. 5 | a: the 42 publications from the CV folds' held-out parts (all in fold 0); b: the 7 NeuroVault papers / 29 maps; c: the 16 test papers and the 29 maps |
| Fig. 6 | excluded draws are redrawn from the same RNG stream (none of the 24 examples is affected) |
| Fig. S3 | the 16 test-set papers (n = 368 -> 366); `--baseline-csv` curves cannot be re-filtered and are flagged |

Figs. 2, S2 and S4 use no held-out items.

## Key numbers: final vs released

"released" = the released checkpoints (Figs. 3, 5, S3: on the same exclusion-aware item sets; Fig. 4: the released render
of n = 2,987, before the 16 leakage-audit papers were dropped, which move its metrics by <= 0.004).

| figure / panel | released | **final** | note |
|---|---|---|---|
| Fig. 2 network titles, exact network name | 7/8 | **7/8** | 8/8 name the right system (released and final); not exact: released Somatomotor "Somatosensory Network", final Visual "Object Perception Network"; table below |
| Fig. 3 brain-to-text quality (text cosine), mean of 6 rows | 0.248 | **0.258** | final rows 101, 110, 92, 74 replace 124, 152, 147, 167 |
| Fig. 3 text-to-brain quality (image cosine), mean of 6 rows | 0.213 | **0.347** | calibrated head v2 maps |
| Fig. 3 generated vs true map (shared-space cosine), mean of 6 rows | - | **0.522** | JSON `image_cos_same` |
| Fig. 4a generated-image retrieval AUC (n) | 0.850 (2,987) | **0.953** (2,971) | uncalibrated 0.887 |
| Fig. 4a contrastive / Dice top 30 % / Pearson, match vs null | 0.297/0.088, 0.495/0.408, 0.321/0.184 | **0.345/-0.004, 0.531/0.408, 0.319/0.123** | |
| Fig. 4a spin test, fraction p < 0.05 (cortical maps) | 0.61 (1,337) | 0.58 (1,332) | |
| Fig. 4b generated-text retrieval AUC | 0.940 | **0.953** | validator 0.9527 |
| Fig. 4b contrastive match / null | 0.266 / 0.018 | **0.281 / 0.009** | |
| Fig. 4b BERTScore F1 match / null (p) | 0.5569 / 0.5519 (4e-17) | 0.5530 / 0.5478 (9e-19) | |
| Fig. 4b MiniLM cosine match / null (p) | 0.469 / 0.401 (< 1e-100) | 0.463 / 0.397 (1e-109) | |
| Fig. 4b contrastive null test, fraction p < 0.05 | 0.61 | **0.70** | 5 % expected by chance |
| Fig. 5a PubMed 5-fold CV AUC t->i / i->t | 0.825 / 0.820 | 0.826 / 0.821 | |
| Fig. 5b NeuroVault upper bound AUC t->i / i->t (n = 305) | 0.832 / 0.774 | **0.845 / 0.785** | a3, mean over k, p-maps inverted (raw maps 0.844 / 0.785, trapezoid 0.843 / 0.783) |
| Fig. 5b multi-target t->i / i->t | 0.859 / - | **0.864 / 0.622** | |
| Fig. 5b lower bound t->i / i->t | - | **0.396 / 0.399** | worst map per paper |
| Fig. 5c voxel AUROC PubMed / NeuroVault | 0.975 / 0.939 | 0.974 / 0.939 | |
| Fig. 5c median bits-per-pixel gain PubMed / NeuroVault | 30.1 % / 35.7 % | 29.7 % / 35.5 % | |
| Fig. 5d network accuracy (n = 119) | 0.82 (97) | 0.82 (98) | |
| Fig. 6 reconstructions | same 24 examples | equivalent | |
| Fig. S2 | - | identical | no model |
| Fig. S3 NeuroVLM AUC t->i / i->t (n = 366) | 0.835 / 0.835 | 0.834 / 0.835 | NiCLIP 0.823 / 0.807, NeuroConText 0.801 / 0.813 |
| Fig. S4 component titles identical to released | 10/10 | **9/10** | 10/10 name the right system; the one difference is HCP frontoparietal "Frontoparietal Control Network" (released "Frontoparietal Network") |

**Fig. 2 network titles** (`[NETWORK]` caption; `[REGION]` / `[FUNCTION]` in the JSON):

| reference map | released | **final** | correct? |
|---|---|---|---|
| Visual (YeoLab VisualA) | Visual Network | Object Perception Network | right system (ventral visual), not the exact name |
| Language (Glasser) | Language Network | Language Network | yes |
| Auditory (Shirer) | Auditory Network | Auditory Network | yes |
| Somatomotor (WashU LateralSM) | Somatosensory Network | Somatomotor Network | yes |
| Interoception (Laird) | Interoceptive Network | Interoceptive Network | yes (region / function "Anterior Insula / Interoceptive Processing") |
| Default mode (Shen) | Default Mode Network | Default Mode Network | yes |
| Frontoparietal control (Du FPN-A) | Frontoparietal Control Network | Frontoparietal Control Network | yes |
| Salience (Du SAL/PMN) | Salience Network | Salience Network | yes (region / function "Anterior Cingulate Cortex / Executive Control") |

The eight hand-picked titles are less stable than the aggregate metrics. With the previous stage-14 recipe the packaged
seed 3 titled the Salience map "Frontoparietal Network" (2 of 5 seeds named it "Salience Network"). With the current
recipe (level-bank projection, map augmentation, title-token weight 5; `scripts/REPRODUCTION.md`, "Stage 14 recipe
change") all five seeds do, and seed 3 (lowest final-epoch validation loss) is packaged again. Per-seed title accuracy
over the 119 labelled network maps: 0.739, 0.765, 0.748 (seed 3, selected), 0.765, 0.798 (previous recipe, same scorer:
0.731, 0.739, 0.756, 0.723, 0.765).

**Fig. 3 rows** (quality = cosine in the shared space, image then text; released -> final for rows in both; final
titles with the level tag the model wrote): 189 (High) image 0.35 -> 0.47, text 0.34 -> 0.41, "Sensorimotor Network"
-> [FUNCTION] "Hand Sensorimotor Control"; 149 (High) 0.28 -> 0.43, 0.37 -> 0.33, "Speech Production" -> [FUNCTION]
"Phonological Processing"; 101 (Medium, replaces 124) 0.34, 0.23, [NETWORK] "Salience Network"; 110 (Medium, replaces
152) 0.36, 0.24, [FUNCTION] "Contextual Perception"; 92 (Low, replaces 147) 0.25, 0.17, [NETWORK] "Speech Perception
Network"; 74 (Low, replaces 167) 0.24, 0.17, [FUNCTION] "Autobiographical Memory".
Row 189 pairs a different same-DOI NeuroVault image than the release (the pairing uses the InfoNCE heads).

**Fig. S4 titles** (final): UK Biobank Visual Network, Language Network, Sensorimotor Network, Default Mode Network,
Frontoparietal Control Network (all as released); HCP Visual Network, Language Network, Sensorimotor Network (component ICA23 instead of
ICA9), Default Mode Network, Frontoparietal Control Network (released Frontoparietal Network).

## Caption corrections

**Fig. 4** (`\label{fig:gen_perf}`).  "$^*p < 1e-100$ was found for all comparisons" is wrong for BERTScore.  Welch
t-test of matched vs deranged-pair null per panel (n = 2,971; -log10 p panels: exact binomial test against the
uniform-p null; `p_values_per_panel` in `fig4_generative_metrics_text.json`):

| panel | p | Cohen's d | < 1e-100? |
|---|---|---|---|
| a Contrastive | 4e-1116 | 2.48 | yes |
| a Dice (top 30 %) | 2e-246 | 0.91 | yes |
| a Pearson | 6e-290 | 1.00 | yes |
| a Spin test (1,332 cortical maps) | 3e-620 | 1.53 | yes |
| b Contrastive | 7e-1158 | 2.48 | yes |
| b BERTScore | **9e-19** | **0.23** | **no** |
| b Semantic (MiniLM) | 1e-109 | 0.59 | yes |
| b Contrastive null test | 6e-1924 | 2.29 | yes |

Suggested text: "Welch t-tests of matched vs shuffled pairs; $^*p < 10^{-100}$ for all panels except BERTScore
($p = 9\times10^{-19}$, Cohen's $d = 0.23$; match 0.553 vs null 0.548)."  The last panel is not a near-chance result: it
is the per-map permutation p-value of the true map among all 2,971 test maps; under the null p is uniform (median 0.5,
5 % below 0.05), drawn as the grey box.  Suggested: "contrastive null test: per-map permutation p-value of the true map
among all 2,971 test maps ($-\log_{10} p$; grey = exact null, p uniform); 70 % of maps reach p < 0.05 vs 5 % expected by
chance", and for the spin panel "58 % of 1,332 cortical maps beat their spatial nulls at p < 0.05 (100 rotations,
smallest attainable p = 1/101)".  The text-to-brain numbers in the text should read Dice 0.53 vs null 0.41 and Pearson
0.32 vs 0.12 (PubMed test).

**Fig. 5b** (`\label{fig:perf_quant}`, "Contrastive recall (AUC=0.78) curve (AUC=0.85)").  The paper recipe (one image
per paper, chosen with the model) gives text->image **0.85** and image->text **0.78** with p-maps inverted (0.844 on the raw maps).  The panel
now draws, per direction (blue text->image, orange image->text), the upper bound (paper recipe), the multi-target curve
(every image of the paper is a target) and the lower bound (the image the model scores lowest), read from the a3
analysis.  Suggested: "b) NeuroVault contrastive recall with three choices of target image per paper: upper bound (the
image the model scores highest for the paper's text; text->image AUC = 0.85, image->text 0.78), multi-target (every
image of the paper is a target; 0.86 / 0.62) and lower bound (the image it scores lowest; 0.40 / 0.40). Dashed:
chance of each curve, the same rule with every abstract given to another paper (mean of 20 derangements; AUC 0.74 / 0.70,
0.76 / 0.50 and 0.30 / 0.31)."  Chance is not k/N here: the upper and lower bounds choose the target with the same
similarities they rank (best / worst of 10.3 maps per paper on average) and multi-target text->image counts a hit for any
of the paper's maps; only multi-target image->text (one abstract per map among 305) is the diagonal.  Independent random
similarities would give 0.82 (upper), 0.18 (lower) and 0.82 (multi text->image), because they ignore that a paper's maps
resemble each other; the derangement keeps the model's map geometry (a3's null, also drawn in its own figure).  **Why
text->image > image->text:** both rank the same 305 x 305 similarity matrix (one image per paper, same exclusions), so
the pools are matched and the gap is not an artefact; a few generic abstracts are close to most maps (one is the nearest
abstract for 39 of the 305 maps), which pushes the true abstract down when a map ranks the abstracts but is constant
within an abstract's ranking of the maps (subtracting each abstract's mean similarity to all maps lifts image->text from
0.785 to 0.841, text->image stays 0.844); the multi-target gap (0.86 / 0.62) is by construction (several hits among
3,154 maps vs one hit among 305 abstracts; chance 0.76 / 0.50).  Panel a's
"AUC=0.81" matches neither direction (5-fold CV 0.83 / 0.82); panel c's AUROC is 0.974 on the PubMed test split (the
text's 0.975) and 0.939 on NeuroVault (the caption's 0.94).

## Deviations from the paper panels

* Fig. 2 "Interoception" reference is `Laird/Emo/Interoception4` (`--laird-map` to change).  The text-to-brain input
  descriptions are transcribed from the paper figure (they are not stored in the repository).
* Fig. 3 quality numbers are cosines in the shared contrastive space (`image` = cos(text, generated map), `text` =
  cos(generated text, true map); `--quality same` for same-modality cosines).  The paper PNG appears to print percentile
  ranks; the caption says cosine.
* Fig. 4 top row: Dice at the percentile with the highest mean Dice over 70-95 (notebook rule); NCT-style spin test on
  fsaverage-3k with 100 BrainSpace rotations, cortical targets only (>= 80 % of the top-10 % activation in the cortex).
* Fig. 5b also draws the lower bound and the multi-target curves of both directions (a3 analysis), each with its own
  dashed chance curve in its colour (a3's derangement null, see Caption corrections; legend: AUC (chance AUC)).  Fig. 5d:
  119 labelled non-ICA maps.  Fig. 5 is 5.15 x 4.58 in: a and b (square recall panels, b's legend column right of it) on
  top; c (bits-per-pixel histogram and square voxel ROC, one legend below both with the ROC AUCs) and d below; d names
  the classes in full on both axes (columns at 60 degrees), one percentage of the true class per cell (no colour bar).
* Fig. S4 "input image" is the thresholded, smoothed map fed to the model, not the raw two-sided ICA component.

## Layout conventions

Drawn at the journal template's text width, `FIG_FULL_W` = 5.15 in (`scripts/common.py`; S3 at half width, 2.57 in),
at most `FIG_MAX_H` = 4.59 in tall per page, every text >= 7 pt at print size; the figures in the response letter are
printed at 6.5 in.  Generated and input texts are printed in full, wrapped by measured width, except Fig. 3's true
abstracts (first lines only, as in the original); Fig. 2's complete input and generated texts are a table
(`fig2_texts_table.{tex,md}`).  Pages after the first (`_p2`, ...): Fig. 3's second rows, Fig. S4's HCP cohort.  Brain maps: one colormap (`Reds`), each
map scaled to its own maximum, views per figure (Fig. 2: sagittal / coronal / axial; Fig. 3 and S2: sagittal and
axial; Fig. 6 and S4: axial only, Fig. 6 at the slice that maximises the product of the true and the reconstructed
map's activation, S4 at the slice with the most activation).  Figs. 2 and 3 draw every map (generated, atlas and true
alike) at the voxels above 50 % of its own maximum, no cluster threshold (`_shared.display_vector`; the rule and each
map's threshold and drawn voxels are in the `_text.json`, `map_display`).  The earlier 99th-percentile cut drew nothing
on calibrated maps, which are clipped at 1 (up to 13 % of a map's voxels at exactly 1).  Recall@k panels are square with matplotlib's default margins, x label "k/n (fraction of all
candidates)"; null distributions grey, matched
blue; p-values printed with their mantissa, `*` = p < 1e-100.

## External inputs for Fig. S3

NiCLIP / NeuroConText need files outside this repository, resolved from `--external-root` (default
`$NEUROVLM_REPOSITORY_ROOT`, else `~/neurovlm`):

```
<root>/baseline/niclip_train_pmids.npy
<root>/pubmed/model-clip_section-body_embedding-BrainGPT-7B-v0.2_best.pth   ($NEUROVLM_NICLIP_CHECKPOINT)
<root>/NeuroConText/best_val.pt                                              ($NEUROVLM_NEUROCONTEXT_CHECKPOINT)
<root>/NeuroConText/data/data_NeuroConText/{train_pmids.pkl,text_preprocessing_pipeline.pkl}
<root>/docs/03_evaluation/{img_emb_512.pt,braingpt_embeddings.pt,mistral_embeddings.pt}   (--external-cache)
```

The 7B text-embedding passes (BrainGPT-7B for NiCLIP, Mistral-7B for NeuroConText) are not re-run; DiFuMo-512 features
are recomputed if missing.  A baseline that cannot be evaluated is read from `--baseline-csv`
(`model,direction,normalized_k,recall`) or omitted.

## Provenance

`runs/final`: head v2 (stage 09, best epoch 30 of 80), stage 1 projector MLP (stage 13, best epoch 4 of 7 run), stage 11
from training-split publications only, anatomical head v2 (stage 12, anchored recipe, best epoch 200), stage 2 projector
MLP (stage 14, seed 3 of 1-5 by validation loss, synthesised maps calibrated and binarised at the 95th percentile);
autoencoder, InfoNCE heads, split, latents and CV folds from the seeded proof run.  Validation:
`scripts/outputs/validation/final/report.md`, 8 of 8 components pass.  Commands and timings: `scripts/README.md`.
