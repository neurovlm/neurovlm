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
| Fig. 2 networks: text-to-brain and brain-to-text | `fig2_network_examples.py` | `fig2_network_examples.*` (the figure, one page, one row per network under one set of column headers: the network's name above its true and generated maps in sagittal, coronal and axial views at one scale, then the Network, Anatomy and Cognition texts generated from the true map, complete with bold run-in titles for Auditory and Default mode, the three titles for the other six; 5.147 x 6.63 in); `fig2_texts_table.{tex,md}` (supplementary table: complete input and generated texts; `--text-pages` also draws them as figure pages `_p2`-`_p4`) | autoencoder, text-to-brain head v2, stage 2 projector (MLP) |
| Fig. 3 NeuroVault examples | `fig3_newgen.py` | `fig3_newgen.*` (one page, six rows) | text-to-brain head v2, autoencoder, InfoNCE heads, stage 2 projector |
| Fig. 4 generative metrics (PubMed test) | `fig4_generative_metrics.py` | `fig4_generative_metrics.*` | row a: autoencoder + latents, head v2; row b: stage 2 projector, InfoNCE heads; deberta-xlarge-mnli / MiniLM scorers |
| Fig. 5 contrastive / autoencoder / networks | `fig5_perf_quant.py` | `fig5_perf_quant.*` | CV folds, autoencoder, InfoNCE heads; b: read from the a3 bounds analysis (same final components, checked) |
| Fig. 6 autoencoder reconstructions | `fig6_autoencoder_qual.py` | `fig6_autoencoder_qual.*` | autoencoder, split |
| Fig. S2 Gaussian kernel vs DiFuMo | `figS2_difumo_v_ale.py` | `figS2_difumo_v_ale.*` | none (data only) |
| Fig. S3 model comparison | `figS3_recall_comparison.py` | `figS3_recall_comparison.*` | autoencoder, InfoNCE heads; NiCLIP / NeuroConText from their own checkpoints |
| Fig. S4 ICA components (HCP, UK Biobank) | `figS4_clip_ica.*` (one page: UK Biobank above, HCP below; input maps with sagittal, coronal and axial slices, 6.54 in tall; `--views z` gives the one-slice version) | autoencoder, InfoNCE heads, stage 2 projector |

**Fig. 3 layout and labels.** One page, 5.15 x 6.82 in at 7 pt (the original, 6.7 x 3.9 in, used ~3.5 pt text).  The
six complete generated texts need about 5.7 in at this width, so Fig. 3 is the one figure above `FIG_MAX_H` (4.59 in);
it fits the 7.67 in text height with its caption.  Columns (check7): the densities of both quality cosines over all 305
NeuroVault papers side by side on one axis labelled "Cosine similarity" (Image: paper text vs generated map, Text:
generated text vs true map, shared space; dashed: thirds; each row one marker in its group colour, all markers on one
vertical line, the cosines themselves are in the JSON), one bar per group, the paper's title (no abstract), the
generated and the true map (check8: sagittal, coronal and axial stacked at one mm-to-inch scale, 0.34 in wide, the
width at which the page and its caption fit the text height), and the complete generated text.  High / Medium / Low means both
cosines lie in the top / middle / bottom third.  Rows (`select_rows`): per third, nearest the third's centre in
percentile space first, among the papers whose generated text is clean (`text_check`: a title of the stage-14 bank of
its level, no code-like character, no word used fewer than twice in the 30,826 PubMed titles and abstracts and in no
bank title, no anatomical name with a word pair PubMed never uses, no content word used more than twice; it rejects
e.g. "[ROOT]" titles, "Placebinding", "angular orbitofrontal cortex", "facial x5"); High and Medium rows also need a
title that names exactly one domain of a fixed word list (`DOMAINS`) and a clean true map (as drawn: >= 2 clusters of
>= 20 voxels, half of the drawn mass in <= 3 pieces, <= 10 % of it in specks), the second row of another domain.
`low_check` says why a Low row scores low (poor input map = true map not clean; surprising pair = the paper's text and
its own map in the pool's bottom third by cosine; else model miss); it goes in the caption, not the figure.  Rows: 30,
150 (High), 52, 117 (Medium), 92 (poor input map: 12 % of its drawn mass in specks), 53 (model miss: clean true map,
true-pair cosine at P57); clean texts per third 46 / 24 / 25 of 53 / 34 / 42, and Medium has exactly two papers that
pass its rule.  Every paper of each third with its checks and text issues is in the JSON `selection.candidates`.  The
pool is generated in batches of six, like the
rows, because the bf16 generation depends on the batch size (~3 min on a GPU).  Its captions use basis "all": the model
writes the level tag itself, and a [NETWORK] / [REGION] title after a written tag is held to the canonical names, as
after a prompt tag (`_common_models.install_title_constraint`).  The generated and the true map of a row share their
three cuts: per view the slice that maximises the product of both maps' drawn activation summed over the slice (each
map scaled to max 1), so every row shows activation in both maps (JSON `rows[].cut_coords`).

`_shared.py` holds the shared code (CLI, caches, model loading, brain maps, text layout); `make_all.py` runs all eight.
SPECTER2 and NeuroQwen3-0.6B are frozen and always come from the release.

The PNGs here are the `runs/final` renders, redrawn on 2026-10-12 at the template widths (two CPU renders from the
caches were byte-identical to them, `scripts/REPRODUCTION.md`); Figs. 2, 4, 5, 6, S3 and S4 were redrawn from the same
caches on 2026-10-13 (CPU) for the recall-axis label, the axial views of Figs. 6 / S4 and the Fig. 2 text table.  Later on 2026-10-13, Figs. 2, 3, 4 and S4 were regenerated with canonical [NETWORK] / [REGION] titles and calibrated maps, and Figs. 2 and 3 drew every map at 50 % of its own maximum (since check6: its top 5 % of voxels).  Figs. 2, 5, S2 and S4 were redrawn
from the caches on 2026-10-13 (CPU) for the author's check6 items: the Fig. 2 layout, Fig. 5c's key and 5d's title,
Fig. S2's "Gaussian kernel" label and Fig. S4's bars.  For check6, Fig. 3 was regenerated on one page with rule-picked rows
(GPU, ~3 min) and Fig. 6 redrawn with the submitted figure's order, slices and NeuroVault maps (CPU).  For check7, Fig. 3
was redrawn from its cache (CPU) with titles only, wider densities on one shared axis, aligned markers, no cosine column
and no Low labels, and rows that also need a clean generated text.  For check8, Fig. 3 was redrawn from its cache
(CPU) with a coronal view added to every map (same rows and cuts), and Fig. 4 without the paired-difference boxes
(3.18 in tall instead of 4.06 in; the legend moved into the last row's band).  Also for check7, Figs. 4, S3 and S4 were redrawn from their caches (CPU): Fig. 4 with a paired-difference box under each box panel and no n in the titles, S3 without n in the title, S4 with both cohorts on one page (smaller maps, generated titles run in) instead of a second `_p2` page; the Fig. 4 text reference pairs (`cache/fig4_text_reference*.pt`) took 15 min on CPU.  `cache/` holds the panel data of Figs. 2, 3, 4, S2 and S4; Figs. 5, 6
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
| Fig. 3 brain-to-text quality (text cosine), mean of 6 rows | 0.243 | **0.246** | released: check6 rows 30, 150, 52, 117, 92, 74; final: check7 rows 30, 150, 52, 117, 92, 53 (check6 rows 0.246) |
| Fig. 3 text-to-brain quality (image cosine), mean of 6 rows | 0.256 | **0.339** | calibrated head v2 maps; check6 rows 0.336 |
| Fig. 3 generated vs true map (shared-space cosine), mean of 6 rows | 0.600 | 0.528 | JSON `image_cos_same`; check6 rows 0.523; the release pairs row 52 with another same-DOI map |
| Fig. 4a generated-image retrieval AUC (n) | 0.850 (2,987) | **0.953** (2,971) | uncalibrated 0.887 |
| Fig. 4a contrastive / Dice top 30 % / Pearson, match vs null | 0.297/0.088, 0.495/0.408, 0.321/0.184 | **0.345/-0.004, 0.531/0.408, 0.319/0.123** | |
| Fig. 4a spin test, fraction p < 0.05 (cortical maps) | 0.61 (1,337) | 0.58 (1,332) | |
| Fig. 4b generated-text retrieval AUC | 0.940 | **0.953** | validator 0.9527 |
| Fig. 4b contrastive match / null | 0.266 / 0.018 | **0.281 / 0.009** | |
| Fig. 4b BERTScore F1 match / null (p) | 0.5569 / 0.5519 (4e-17) | 0.5530 / 0.5478 (9e-19) | submitted figure's null 0.520 was stale (Caption corrections) |
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

**Fig. 3 rows** (quality = cosine in the shared space, image then text, released -> final; titles with the level tag
the model wrote, released -> final): 30 (High) 0.37 -> 0.44, 0.38 -> 0.31, [NETWORK] "Frontoparietal Control Network"
-> [FUNCTION] "Working Memory"; 150 (High) 0.26 -> 0.42, 0.31 -> 0.34, [NETWORK] "Language Network" (both); 52 (Medium)
0.29 -> 0.35, 0.19 -> 0.26, [REGION] "Parahippocampal Gyrus" -> [FUNCTION] "Contextual Scene Memory"; 117 (Medium) 0.27
-> 0.31, 0.18 -> 0.22, [REGION] "Fusiform Gyrus" (both); 92 (Low) 0.23 -> 0.25, 0.20 -> 0.17, [REGION] "Superior
Temporal Gyrus" -> [NETWORK] "Speech Perception Network"; 74 (Low) 0.12 -> 0.24, 0.20 -> 0.17, [REGION] "Posterior
Cingulate Cortex" -> [FUNCTION] "Autobiographical Memory".  check7 replaces 74 (its caption names an "angular
orbitofrontal cortex") with 53 (Low, final 0.27, 0.17, [FUNCTION] "Salience-Guided Reorienting").  Rows from the final model's rule (the released pool's own
rule would pick 175, 136, 100, 117, 103, 127).  Before check6 the rows were 189, 149 (hand-picked) and 101, 110, 92, 74;
the rule drops 149 (its title names two domains, spoken and manual) and 110 (no domain).

**Fig. S4 titles** (final): UK Biobank Visual Network, Language Network, Sensorimotor Network, Default Mode Network,
Frontoparietal Control Network (all as released); HCP Visual Network, Language Network, Sensorimotor Network (component ICA23 instead of
ICA9), Default Mode Network, Frontoparietal Control Network (released Frontoparietal Network).

## Caption corrections

**Fig. 4** (`\label{fig:gen_perf}`).  Stars mark p < 0.05 (*), p < 0.01 (**) and p < 0.001 (***), no star = not
significant; every panel is ***.  Welch t-test of matched vs deranged-pair null per box panel (n = 2,971); -log10 p
panels: one-sided exact binomial test of #(p < 0.05) against the uniform-p null (`p_values_per_panel` in
`fig4_generative_metrics_text.json`):

| panel | p | Cohen's d | stars |
|---|---|---|---|
| a Contrastive | 4e-1116 | 2.48 | *** |
| a Dice (top 30 %) | 2e-246 | 0.91 | *** |
| a Pearson | 6e-290 | 1.00 | *** |
| a Spin test (1,332 cortical maps) | 3e-620 | 1.53 | *** |
| b Contrastive | 7e-1158 | 2.48 | *** |
| b BERTScore | 9e-19 | 0.23 | *** |
| b Semantic (MiniLM) | 1e-109 | 0.59 | *** |
| b Contrastive null test | 6e-1924 | 2.29 | *** |

The panels no longer print the spin-test n, the Dice threshold or the k/n definition; the caption carries them.
Suggested: "Recall@$k$ curves show retrieval of the true pair, with $k/n$ the fraction of all $n$ candidates searched.
Boxes compare matched pairs (blue) with shuffled pairs (gray). Dice uses each map's top 30\% of voxels. The spin test
(1,332 cortical maps) and the contrastive null test show each map's $-\log_{10} p$, with $p = 0.05$ dashed. Stars mark
Welch $t$-tests of matched against shuffled pairs in the box panels and one-sided binomial tests of the share of maps
below $p = 0.05$ in the two $p$ panels. One, two and three stars mark $p < 0.05$, $0.01$ and $0.001$, and no star marks
a non-significant difference."

Paired differences (check7, `paired_differences` in the JSON; Δ = match − null of the same map or caption, n = 2,971;
median with bootstrap 95 % CI, d_z = mean / SD of Δ, paired t and sign test):

| panel | Δ > 0 | median Δ [95 % CI] | d_z | paired t p | sign test p |
|---|---|---|---|---|---|
| a Contrastive | 95 % | 0.346 [0.334, 0.355] | 1.60 | 3e-819 | 2e-638 |
| a Dice | 74 % | 0.118 [0.108, 0.128] | 0.63 | 1e-220 | 2e-158 |
| a Pearson | 76 % | 0.193 [0.179, 0.207] | 0.71 | 1e-265 | 4e-187 |
| b Contrastive | 96 % | 0.269 [0.262, 0.278] | 1.75 | 6e-904 | 5e-670 |
| b BERTScore | 57 % | 0.0048 [0.0034, 0.0063] | 0.18 | 3e-22 | 3e-13 |
| b Semantic (MiniLM) | 66 % | 0.062 [0.055, 0.069] | 0.42 | 6e-108 | 1e-68 |

Why the BERTScore gap is narrow (`text_score_reference`, 200 test papers, seed 13, deberta-xlarge-mnli rescored on
CPU; 20 cached matched pairs reproduce to 4e-4).  Unrescaled F1 has a high floor: two random test papers' texts (title
+ LLM summary) score 0.579, a paper and its nearest test paper by MiniLM 0.623, a paper's summary and its own abstract
0.684.  The captions describe the map in another register (network, region, function), so they sit below even random
paper pairs: 0.553 matched, 0.548 shuffled.  bert_score's baseline for this model and layer (random Common Crawl pairs,
F1 0.515; `rescale_with_baseline`) maps these to 0.132, 0.222, 0.349, 0.078 and 0.068; rescaling is linear, so Δ
doubles (0.0107) while d and every p stay the same.  MiniLM: random 0.37, nearest paper 0.73 (selected by MiniLM),
same paper 0.84, captions 0.46 matched and 0.40 shuffled.  The small effect is real: Δ > 0 for 57 % of captions and
the mean Δ is 9.8 standard errors from 0, which n = 2,971 resolves (Welch p = 9e-19, paired p = 3e-22).

BERTScore gap.  The submitted figure drew null 0.520 against match 0.557, but notebook 24 reused a cached null
(`null_bert_f1_seed0.pt`, 26 May 17:12, accepted on length alone) from an earlier generation run after the captions
were regenerated (`df_gen.parquet` 20:03, `bert_f1.pt` 20:05).  Rescored on CPU, the submitted captions against their
own shuffled references give match 0.5567 / null 0.5538 (400 random items; the stale null on them 0.520); the released render (0.5569 / 0.5519) and the final model (0.5530 /
0.5478) agree.  Scorer, references, null rule and axis range are unchanged; the narrow gap is the real one.

The last panel is not a near-chance result: it
is the per-map permutation p-value of the true map among all 2,971 test maps; under the null p is uniform (median 0.5,
5 % below 0.05), drawn as the grey box.  The text-to-brain numbers in the text should read Dice 0.53 vs null 0.41 and
Pearson 0.32 vs 0.12 (PubMed test).

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
  descriptions are transcribed from the paper figure (they are not stored in the repository).  Page 1 (5.15 x 6.45 in,
  redrawn from the cache for check7) follows the author's sketch.  The two networks with the cleanest generations get a
  row each: name, true map ("True") and the map generated from the description ("Generated (text-to-brain)") on the
  left, the complete generated texts in Network | Anatomy | Cognition columns under one header ("Generated
  (brain-to-text)", a thin rule under each column name; column widths chosen for the fewest lines, `column_widths`).
  The other six sit in two rows of three cells: name, the two maps side by side with the same labels, and the three
  generated titles, labelled in a left gutter.  Every map shows sagittal, coronal and axial views; figure names are in
  sentence case (`NAMES`; the table keeps the titles).  The two are chosen by rule (`pick_full`): most of the three generated titles named verbatim in the input
  description, ties broken by the share of the description's word pairs that the three generated texts repeat
  (Default Mode Network 3 titles, Auditory 2 titles and overlap 0.42 over Frontoparietal and Salience; JSON
  `full_examples`).  The atlas map names and all complete texts are in `fig2_texts_table`.
* Fig. 3 quality numbers are cosines in the shared contrastive space (`image` = cos(text, generated map), `text` =
  cos(generated text, true map); `--quality same` for same-modality cosines).  The paper PNG appears to print percentile
  ranks; the caption says cosine.
* Fig. 4 top row: Dice at the percentile with the highest mean Dice over 70-95 (notebook rule); NCT-style spin test on
  fsaverage-3k with 100 BrainSpace rotations, cortical targets only (>= 80 % of the top-10 % activation in the cortex).
* Fig. 5b also draws the lower bound and the multi-target curves of both directions (a3 analysis), each with its own
  dashed chance curve in its colour (a3's derangement null, see Caption corrections; legend: AUC (chance AUC)).  Fig. 5d:
  119 labelled non-ICA maps.  Fig. 5 is 5.15 x 4.58 in: a and b (square recall panels, b's legend column right of it) on
  top; c (bits-per-pixel histogram and square voxel ROC, one key for both in the ROC's empty lower right, the AUROCs in
  the caption) and d below; d names the classes in full on both axes (columns at 60 degrees), one percentage of the true
  class per cell (no colour bar), titled "% of true class" (n and accuracy in the caption).
* Fig. S2 top maps are single-study maps of nimare's ALE kernel (each focus convolved with a 9 mm FWHM Gaussian, maximum
  over the foci, `scripts/data/01_coordinate.py`).  With one study no ALE is taken across studies, so the figure calls
  them Gaussian-kernel maps.
* Fig. S4 "input image" is the thresholded, smoothed map fed to the model, not the raw two-sided ICA component.  Bars
  are matplotlib "C0" at alpha 0.35, each label on its bar's midline (`va="center_baseline"`) from the bar's left end.
  Both cohorts share one 5.15 x 6.13 in page (check7; the page also holds the section heading and the caption, a
  test build of the manuscript has no overfull page): UK Biobank above, HCP below, each named on the left; maps 0.44 in
  wide; the complete generated text at 7 pt with 1.05 line spacing and its title run in bold.

## Layout conventions

Drawn at the journal template's text width, `FIG_FULL_W` = 5.15 in (`scripts/common.py`; S3 at half width, 2.57 in),
at most `FIG_MAX_H` = 4.59 in tall per page, every text >= 7 pt at print size; the figures in the response letter are
printed at 6.5 in.  Generated and input texts are printed in full, wrapped by measured width, except Fig. 3's papers
(title only since check7; the abstracts are in its JSON) and Fig. 2's six short examples (the three generated titles); Fig.
2's complete input and generated texts are a table (`fig2_texts_table.{tex,md}`).  Brain maps: one colormap (`Reds`), each
map scaled to its own maximum, views per figure (Figs. 2 and 3: sagittal, coronal and axial; Fig. S2: sagittal and axial; Fig. 6 and S4: axial only, Fig. 6 at the submitted figure's slice (z of nilearn's cut for the thresholded
reconstruction), S4 at the slice with the most activation).  Figs. 2 and 3 draw every map (generated, atlas and true
alike) at its top 5 % of in-mask voxels, ties at the cut drawn, no cluster threshold (`_shared.display_vector`; the rule
and each map's threshold and drawn voxels are in the `_text.json`, `map_display`), about the extent of the submitted Fig.
3.  Half the maximum (the rule before check6) kept up to a sixth of the brain on calibrated maps, whose global factor (x19
for the final head) clips the broad base of a peaked map at 1: Fig. 3's text 149 drew 4,678 voxels (2,091 at exactly 1),
the submitted figure 1,727 (released NeuroAdapter, raw > 0.5).  Fig. 3 draws its generated maps unclipped (each divided
by its maximum) and colours every map from its cut to its maximum (`_shared.display_vmin`).  Recall@k panels are square with matplotlib's default margins, x label "k/n (fraction of all
candidates)" (Fig. 4: "k/n", defined in its caption, so five columns fit); null distributions grey, matched
blue.  Fig. 4 marks each pair with stars (* p < 0.05, ** p < 0.01, *** p < 0.001, no star = not significant;
the p-values are in `fig4_generative_metrics_text.json`), writes Match / Null under every pair and the legend in
the last row's band (under the Match / Null labels, right of the k/n label), and fits each box panel's y range to the
whiskers (outliers are not drawn); its columns are spaced from the measured y-axis labels (`place_columns`).  The
paired-difference boxes of check7 were removed in check8 (they left a band of white space); the per-item paired
statistics stay in the JSON (`paired_differences`, `text_score_reference`) and the small BERTScore gap is defended
in the caption and the response; the retrieval titles no longer print n (caption).

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
