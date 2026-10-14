# Reviewer 1

Each comment, our response (copied from [`../RESPONSE.md`](../RESPONSE.md)) and the figure / table files used, copied here and named by comment.  Rebuilt by `../build_reviewer_dirs.py`; edit RESPONSE.md, not this file.

<a id="1.0"></a>
## 1.0: General remarks

**Comment:** an ambitious and timely framework with a large corpus and public code, but the evaluation and data construction need clarification before the stronger claims are supported.

**Response:** We added the requested analyses and clarifications ([1.1](#1.1) to [1.10](#1.10)).

<a id="1.1"></a>
## 1.1: More independent evaluation of the generative outputs

**Comment:** outputs are scored in NeuroVLM's own space. Add independent representations, expert ratings, hard negatives, nearest training examples, simple baselines, recall@k, MRR and median rank.

**Response:** We agree. Fig. R1 and Table R1 score text-to-brain maps in voxel space, against each test paper's reconstructed map. NeuroVLM leads, narrowly ahead of linear regression and ahead of Text2Brain, mean and domain maps, atlas labeling, Neurosynth, NeuroQuery, and the nearest training paper (Dice 0.53 against at most 0.49). Neurosynth, NeuroQuery, and Text2Brain are used as published, although their corpora each hold nearly half the test papers.

**Independent evaluation.** Both directions are also scored outside NeuroVLM's representation space. Maps are compared in voxel space, and NeuroVLM also leads against the raw coordinate maps. MiniLM cosine and BERTScore, whose encoders were not used in training, compare generated captions with the paper's text ([3.3](../reviewer_03/README.md#3.3)).

**Matched and shuffled pairs.** The submission's BERTScore null came from an earlier generation run. With the null computed from the same captions, BERTScore separates matched from shuffled pairs only weakly (0.553 against 0.548). BERTScore mostly rewards the style of the reference, an LLM summary. Even random training texts in that style score higher (Table R11).

**Modest differences.** The narrow gaps are real. Captions describe the map rather than the study, so they share little content with the paper's text. Unrescaled BERTScore stays near 0.58 even for two unrelated papers. The matched caption scores higher for 57% of maps by BERTScore, and more often by MiniLM. The paired test over all test maps resolves this small shift (Fig. R2).

**Expert ratings.** We did not run blinded expert ratings, because they are time consuming and hard to scale. Instead, we used existing expert labels and automatic checks. For anatomical and semantic accuracy, region titles are checked against anatomical atlases and network titles against expert atlas labels ([3.6](../reviewer_03/README.md#3.6)). Hard negatives test specificity. Unsupported claims are counted as study details, such as a sample size, that a map cannot support ([3.3](../reviewer_03/README.md#3.3)).

**Hard negatives.** Table R2 scores retrieval among all test papers and among hard negatives, the test papers of the query's own cognitive domain. Hard negatives are only slightly harder than random papers of the same number (recall@10 0.24 against 0.28).

**Nearest training examples.** Generated maps match each test paper's map better than the map of the nearest training paper does (Table R1). By MiniLM cosine, generated captions are as close to the paper's text as the nearest training caption. However, 57% of these retrieved captions report details of another study, against 2% of generated captions ([3.3](../reviewer_03/README.md#3.3)).

**Simple baselines.** Fig. R1 and Table R1 include each suggested baseline. The linear regression maps the SPECTER2 text embedding to the autoencoder latent, the spatial coefficients of each map. We fit it as ridge regression, which is linear regression with a small penalty against overfitting, chosen on the validation set. NeuroVLM's final head learns the same mapping through a hidden layer, with a loss that also scores the decoded map.

**Retrieval measures.** Table R2 reports recall@1, recall@5, recall@10, mean reciprocal rank, and median rank in both directions. Fig. R1 and Table R2 are new in the Supplementary Information.

![Fig. R1](1.1_text_to_brain_baselines.png)
*Fig. R1.* Text-to-brain baselines. Text-to-brain maps from NeuroVLM and other methods, scored against each test paper's reconstructed map (n = 2,971). Panels show (a) Dice and (b) voxelwise Pearson r, with means and 95% bootstrap intervals. Dice uses each map's top 30% of voxels. Neurosynth, NeuroQuery, and Text2Brain were used as published. Their corpora include 1,386 (Neurosynth) or 1,295 (NeuroQuery and Text2Brain) test papers, which can only favor them. Text2Brain's training split is unreleased, so instead of retraining we scored the 1,157 papers in neither corpus, and NeuroVLM still beats all three.

Files: [`1.1_text_to_brain_baselines.png`](1.1_text_to_brain_baselines.png), [`1.1_text_to_brain_baselines.svg`](1.1_text_to_brain_baselines.svg), [`1.1_text_to_brain_baselines_table.tex`](1.1_text_to_brain_baselines_table.tex), [`1.1_retrieval_table.tex`](1.1_retrieval_table.tex), [`1.1_brain_to_text_baselines_table.md`](1.1_brain_to_text_baselines_table.md).
Source: `core/a1_text_to_brain_baselines/outputs/text_to_brain_baselines.{png,svg}`, `core/a1_text_to_brain_baselines/outputs/text_to_brain_baselines_table.tex`, `core/a1_retrieval_metrics_table/outputs/retrieval_metrics_table.tex`, `core/a1_brain_to_text_baselines/outputs/brain_to_text_baselines_table.md`.

<a id="1.2"></a>
## 1.2: Publication-level supervision and dataset separation

**Comment:** explain how coordinates are combined per paper, give final counts, confirm that held-out papers were kept out of the language-model corpus and the synthetic data, and add a data-flow diagram.

**Response:** **1. Experiments and contrasts.** Yes, each paper's map combines the coordinates of all its experiments and contrasts. The released coordinates carry no experiment or contrast labels. Each map matches the paper-level text it is paired with.

**2. Positive and negative effects.** Positive and negative effects are pooled, because the released coordinates carry no sign labels. Each map therefore shows where effects lie, not their sign. The Methods state this.

**3. Multiple coordinate tables.** All coordinate tables of a paper are pooled into one map. Among open-access papers with a coordinate table, 42% report two or more.

**4. Final counts.** Table R3 gives the final counts, 30,826 publications with 857,298 peaks. Coordinate tables can be counted only in papers with open-access full text, and the table gives that count. Experiments and contrasts are unlabeled, so they cannot be counted.

**5. Contrast-level evaluation.** A contrast-level evaluation is possible only on NeuroVault, whose maps are released per contrast ([1.3](#1.3)). With every map as its own target, the model beats chance in both directions (AUC 0.62 and 0.63).

**Separation of datasets.** Fig. R3 is the requested data-flow diagram. No validation or test paper trains the autoencoder, projection heads, or projectors. The language model's training script removes the validation, test, and NeuroVault evaluation papers from its corpus. Generated captions share only short phrases with their abstracts (median two words).

**Synthetic data and prototype bank.** The synthetic maps use training papers only ([2.4](../reviewer_02/README.md#2.4)). The prototype bank holds network, region, and function texts, not test documents. It names the networks that label the evaluation atlas maps, but no atlas network map trains any component.

| Quantity | Released data |
|---|---|
| Publications / peaks | 30,826 / 857,298 (median 17 peaks per paper) |
| Coordinate tables | 20,541 in the 11,943 open-access papers with a coordinate table (42% have two or more) |
| Experiments, contrasts | not labeled (only paper identifiers and coordinates are released) |
| Train / validation / test papers | 23,894 / 2,987 / 2,987, and 958 in no split (2,971 test papers after the leakage audit) |

![Fig. R3](1.2_dataflow.png)
*Fig. R3.* Data flow of the final model. Boxes 1–5 are the trained components in training order, with their training and validation sizes. Gray arrows carry training data or weights, and orange dashed arrows mark sets used only for evaluation.

Files: [`1.2_dataflow.png`](1.2_dataflow.png), [`1.2_dataflow.svg`](1.2_dataflow.svg), [`1.2_counts_table.tex`](1.2_counts_table.tex), [`1.2_stage_splits.md`](1.2_stage_splits.md).
Source: `core/a4_dataflow_diagram/outputs/a4_dataflow.{png,svg}`, `core/a4_counts_table/outputs/counts_table.tex`, `core/a4_counts_table/outputs/stage_splits.md`.

<a id="1.3"></a>
## 1.3: NeuroVault evaluation

**Comment:** keep the model-selected image as an upper bound, add a target-independent evaluation, and treat and report map types separately.

**Response:** We keep the model-selected map as an upper bound and add a lower bound, each paper's worst-matching map (Fig. R4). We add two target-independent scores. Multi-target retrieval counts any of a paper's maps as a hit, and the per-map score makes each map its own target. The model beats chance under every rule. Per-map text-to-brain and brain-to-text retrieval reach an area under the recall curve (AUC) of 0.62 and 0.63.

**Map types.** The submission thresholded every map alike, after removing masks, parcellations, anatomical maps, and variance maps. P-maps are inverted to 1 − p and signed maps keep their positive tail, as before. Using both tails, or adding negative tails as extra maps, reduces evaluation performance. Table R4 reports each map type separately, and Fig. 5b uses the same handling.

**Maps labeled Other.** Other is a type that NeuroVault uploaders choose, not a missing one. We typed these maps from their file names, descriptions, and values. Most are t, z, or F maps, effect estimates, or components (Table R4). Mean or variance images and maps we could not type are excluded, like the variance maps above. This moves every AUC by at most 0.004.

| Map type | Maps | Papers | Handling | AUC, text → brain | AUC, brain → text |
|---|---|---|---|---|---|
| T | 1,761 | 189 | positive tail | 0.64 | 0.64 |
| Z | 811 | 75 | positive tail | 0.57 | 0.58 |
| beta, univariate | 43 | 5 | positive tail | 0.61 | 0.77 |
| F | 40 | 17 | positive tail | 0.64 | 0.55 |
| beta, multivariate | 14 | 5 | positive tail | 0.84 | 0.83 |
| p | 3 | 2 | inverted (1 − p) | 0.64 | 0.60 |
| 1 − p | 2 | 1 | used as is | 0.54 | 0.44 |
| Other, t, z, or F | 204 | 23 | positive tail | 0.63 | 0.70 |
| Other, effect estimate | 103 | 15 | positive tail | 0.69 | 0.62 |
| Other, component or gradient | 93 | 8 | positive tail | 0.49 | 0.51 |

![Fig. R4](1.3_neurovault_bounds.png)
*Fig. R4.* NeuroVault retrieval. Recall for (a) text to brain and (b) brain to text, with each paper's best, worst, or every map as target. Dashed lines show chance from shuffled texts. The best map, or any of a paper's maps counting as a hit in (a), lifts chance above the diagonal, and the worst map lowers it.

Files: [`1.3_neurovault_bounds.png`](1.3_neurovault_bounds.png), [`1.3_neurovault_bounds.svg`](1.3_neurovault_bounds.svg), [`1.3_bounds_table.tex`](1.3_bounds_table.tex), [`1.3_map_type_table.tex`](1.3_map_type_table.tex).
Source: `core/a3_neurovault_bounds/outputs/neurovault_bounds.{png,svg}`, `core/a3_neurovault_bounds/outputs/bounds_table.tex`, `core/a3_neurovault_bounds/outputs/map_type_table.tex`.

<a id="1.4"></a>
## 1.4: Scope of the generative claims

**Comment:** describe outputs as coarse spatial priors, and add failure cases, out-of-distribution inputs and a confidence or abstention signal.

**Response:** We agree and adopt this framing. The Abstract, Results, and Discussion call text-to-brain outputs coarse spatial priors, without effect size, sign, or uncertainty. They call brain-to-text outputs broad network-level labels, not interpretations of a specific study or mechanism. Failure cases are not rare. In 16% of test-set captions, the similarity to the paper's text is negative ([2.12](../reviewer_02/README.md#2.12)).

**Out-of-distribution inputs.** Fig. R5 flags a map when it correlates poorly with its autoencoder reconstruction or its latent lies far from the training latents. It flags a text that lies far from every training abstract. Thresholds set on the validation split flag about 5% of held-out test inputs. They flag every shuffled, white-noise, and rescaled map, every off-topic text, and 61 to 80% of rotated or spun maps.

**Abstention and confidence.** Flagged inputs can be withheld, but the scores are warnings, not calibrated probabilities. Within the training distribution, the peak of a calibrated text-to-brain map works as a confidence score, since it tracks the map's retrieval rank (Spearman 0.63). It does not detect off-topic text, whose maps mostly peak within the range of test abstracts.

![Fig. R5](1.4_ood.png)
*Fig. R5.* Out-of-distribution warnings. Panels show (a) each input's similarity to the training data, (b) the distance of a map's latent from the training latents, and (c) the share of inputs flagged. Dashed lines mark thresholds set on the validation split. Blue marks held-out test inputs.

Files: [`1.4_ood.png`](1.4_ood.png), [`1.4_ood.svg`](1.4_ood.svg), [`1.4_ood_table.tex`](1.4_ood_table.tex).
Source: `core/a7_ood_confidence/outputs/a7_ood.{png,svg}`, `core/a7_ood_confidence/outputs/a7_ood_table.tex`.

<a id="1.5"></a>
## 1.5: Inconsistent dataset and metric values

**Comment:** 3,000 or 4,553 NeuroVault maps, 100 or 300 network images, and AUROC 0.94 or 0.975.

**Response:** We recomputed every count from the released data and corrected the text and captions (Table R5). The supplement's 4,553 maps predate image quality control and the removal of papers already in PubMed. Neither network-image count can be reproduced, so the released counts replace both. Both AUROCs are correct but refer to different datasets, which the Fig. 5 caption names.

| In the paper | Correct value |
|---|---|
| 3,000 or 4,553 NeuroVault maps | 3,183 released maps from 312 papers, 3,154 maps from 305 papers after the leakage audit, and 3,074 maps from 298 papers for retrieval |
| 100 or 300 network images | 185 network maps, made of 145 atlas maps (119 with a network label) and 40 ICA components |
| AUROC 0.94 and 0.975 | both correct, 0.94 on NeuroVault and 0.975 on the PubMed test set (0.974 with the final model) |

Files: [`1.5_counts_table.tex`](1.5_counts_table.tex).
Source: `core/a4_counts_table/outputs/counts_table.tex`.

<a id="1.6"></a>
## 1.6: SPECTER and SPECTER2

**Comment:** SPECTER and SPECTER2 are used inconsistently.

**Response:** Every text-encoding step uses SPECTER2, the allenai/specter2_aug2023refresh model with its ad hoc query adapter. The original SPECTER is never used. We corrected every mention to SPECTER2.

<a id="1.7"></a>
## 1.7: Two methodological analogies

**Comment:** transformers also use a fixed tokenizer vocabulary, and a projection head is not LoRA.

**Response:** We agree with both points. We deleted the LoRA analogy and rewrote the tokenizer sentence. It states that contextual encoders also use a fixed vocabulary and that their advantage is composing words in context. SPECTER2 beats a context-free word2vec encoder ([3.4](../reviewer_03/README.md#3.4)). Swapping one word moves the generated map as expected, but negation has no effect ([3.5](../reviewer_03/README.md#3.5)).

<a id="1.8"></a>
## 1.8: Edge-device claim

**Comment:** report memory, latency, hardware, precision and quantization for the full pipeline.

**Response:** We timed both full pipelines, projector and language model included, on a consumer GPU and a desktop CPU at several precisions (Table R6). We used the released weights, whose projector and text-to-brain head match the final ones in size. On the CPU, int8 quantization runs brain-to-text fastest, with the same memory as bf16. No phone or embedded device was tested, so "edge devices" becomes "a consumer CPU" in the Methods.

| Pipeline, one query | Parameters | GPU, bf16 | CPU, fp32 | CPU, bf16 | CPU, int8 |
|---|---|---|---|---|---|
| text-to-brain | 171 M | 7 ms, 0.4 GB | 135 ms, 0.6 GB | 53 ms, 0.4 GB | 120 ms, 0.5 GB |
| brain-to-text, 128 tokens | 685 M | 3.2 s, 1.3 GB | 13.1 s, 2.7 GB | 6.3 s, 1.4 GB | 3.5 s, 1.4 GB |

Files: [`1.8_edge_table.md`](1.8_edge_table.md).
Source: `core/a8_edge_benchmark/outputs/a8_summary_table.md`.

<a id="1.9"></a>
## 1.9: Figure text size, full descriptions, affiliations

**Comment:** text in Figs. 2, 3 and S4 is too small. Give the full generated descriptions and fix the "Organization" placeholders.

**Response:** We redrew Figs. 2, 3, and S4 at the journal's text width with larger text. Figs. R6, R7, and R8 show the new Figs. 2, 3, and S4. Supplementary Table S5 gives the complete texts of Fig. 2, and Figs. 3 and S4 print each generated text in full beside its map. We also corrected the affiliations.

![Fig. R6](1.9_fig2_network_examples.png)
*Fig. R6.* New Fig. 2. Auditory and default mode networks show the complete generated texts, and the other six their three generated titles. True is the atlas map. Maps show their top 5% of voxels in sagittal, coronal, and axial views. Supplementary Table S5 gives every text.

![Fig. R7](1.9_fig3_newgen.png)
*Fig. R7.* New Fig. 3. NeuroVault papers from the high, medium, and low thirds of the shared-space cosines, chosen by a fixed rule (3.9). Low rows show a scattered true map (top) and a model miss on a clean map (bottom). Maps show their top 5% of voxels.

![Fig. R8](1.9_figS4_clip_ica.png)
*Fig. R8.* New Fig. S4. Labeling of resting-state components from UK Biobank (top) and the Human Connectome Project (bottom). Bars give the five most similar labels from the contrastive models. The generated column is the [NETWORK] caption of the stage 2 MLP projector, its title in bold.

Files: [`1.9_fig2_network_examples.png`](1.9_fig2_network_examples.png), [`1.9_fig2_network_examples.svg`](1.9_fig2_network_examples.svg), [`1.9_fig3_newgen.png`](1.9_fig3_newgen.png), [`1.9_fig3_newgen.svg`](1.9_fig3_newgen.svg), [`1.9_figS4_clip_ica.png`](1.9_figS4_clip_ica.png), [`1.9_figS4_clip_ica.svg`](1.9_figS4_clip_ica.svg).
Source: `../main/fig2_network_examples.{png,svg}`, `../main/fig3_newgen.{png,svg}`, `../main/figS4_clip_ica.{png,svg}`.

<a id="1.10"></a>
## 1.10: Overall assessment

**Comment:** a creative and useful framework that needs independent validation of the generated outputs, clearer dataset separation, a revised NeuroVault evaluation, and more precise framing of the generated maps.

**Response:** Comments [1.1](#1.1), [1.2](#1.2), [1.3](#1.3), and [1.4](#1.4) answer the four revisions named here, in the same order.

<a id="1.11"></a>
## 1.11: Code availability

**Comment:** fix the merge conflicts, remove manual steps and machine-specific paths, pin dependencies and Hugging Face artifacts, and add a figure-by-figure guide. Coverage was 47%.

**Response:** We resolved the merge conflicts in both notebooks and added a figure-by-figure guide that names the script behind every paper figure. One command builds the pinned environment and fetches the Hugging Face inputs at pinned, checksummed revisions. It then retrains the model, redraws the figures, and checks each metric against the paper. Two runs from scratch gave byte-identical weights on the same GPU model and library versions.

**Tests and coverage.** Coverage is measured on the 57 files needed to train the models and draw the figures. The CNN and other experiments outside the paper are left out. Training takes hours, so continuous integration trains every model for one batch on CPU, which checks the whole pipeline end to end.

**Remaining limits.** Two inputs still come from outside the pinned set. They are the Automated Anatomical Labeling (AAL) atlas, whose download server fails, and the NiCLIP and NeuroConText outputs compared in Fig. S3.
