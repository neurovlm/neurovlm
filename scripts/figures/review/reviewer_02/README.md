# Reviewer 2

Each comment, our response (copied from [`../RESPONSE.md`](../RESPONSE.md)) and the figure / table files used, copied here and named by comment.  Rebuilt by `../build_reviewer_dirs.py`; edit RESPONSE.md, not this file.

<a id="2.0"></a>
## 2.0: General remarks

**Comment:** a meaningful topic with a potentially valuable dataset, but not suitable for publication in its current form.

**Response:** We addressed each issue ([2.1](#2.1) to [2.12](#2.12)).

<a id="2.1"></a>
## 2.1: Motivation of the Qformer

**Comment:** the motivation for the Qformer is unclear. Compare it with simpler alternatives, such as an MLP projector or cross-attention.

**Response:** We agree. Our original Qformer attended over only two input tokens, the map latent and its semantic vector, so its attention had little to select. A multilayer perceptron (MLP) projector of the same size therefore replaces the Qformer in both stages. It raises stage 1 retrieval AUC from 0.82 to 0.90 (Fig. R9). A cross-attention module would also attend over only two tokens, so we did not add one.

![Fig. R9](2.1_mlp_projector.png)
*Fig. R9.* Qformer and MLP projector. Gray bars show the original model with the Qformer and blue bars the final model with the MLP projector, on PubMed test maps. Panels show (a) retrieval AUC, (b) caption-to-map cosine, and (c) BERTScore F1 against the paper's text.

Files: [`2.1_mlp_projector.png`](2.1_mlp_projector.png), [`2.1_mlp_projector.svg`](2.1_mlp_projector.svg), [`2.1_mlp_projector_table.tex`](2.1_mlp_projector_table.tex).
Source: `core/a5_mlp_projector/outputs/a5_mlp_projector.{png,svg}`, `core/a5_mlp_projector/outputs/a5_mlp_projector_table.tex`.

<a id="2.2"></a>
## 2.2: Datasets and sample sizes per model

**Comment:** give the datasets and sample sizes of each model, and say whether image-text pairs are reused across training stages.

**Response:** **1. Datasets and sample sizes.** Fig. R3 and Table R7 give the data and sample size of each training stage ([1.2](../reviewer_01/README.md#1.2)).

**2. Reuse across stages.** Yes, the training papers are reused across stages. The autoencoder, the projection heads, and the stage 1 projector all learn from these papers. The anatomically centered head and the stage 2 projector learn from short concept texts, paired with maps built from training papers only ([2.6](#2.6)). No validation or test paper trains these components. The language model's training script also removes them from its corpus ([2.11](#2.11)).

![](2.2_dataflow.png)

Files: [`2.2_dataflow.png`](2.2_dataflow.png), [`2.2_dataflow.svg`](2.2_dataflow.svg), [`2.2_stage_splits.md`](2.2_stage_splits.md).
Source: `core/a4_dataflow_diagram/outputs/a4_dataflow.{png,svg}`, `core/a4_counts_table/outputs/stage_splits.md`.

<a id="2.3"></a>
## 2.3: NeuroVault selection bias

**Comment:** when a paper has several NeuroVault images, the model picks the most similar one, which may inflate the results.

**Response:** We agree that choosing the most similar map inflates the score. That map is reported as an upper bound ([1.3](../reviewer_01/README.md#1.3)). A lower bound keeps each paper's worst-matching map, and a multi-target score counts any of the paper's maps (Fig. R4). Each rule has its own chance curve, from texts shuffled across papers. Under every rule, the model's AUC stays about 0.1 above chance.

![](2.3_neurovault_bounds.png)

Files: [`2.3_neurovault_bounds.png`](2.3_neurovault_bounds.png), [`2.3_neurovault_bounds.svg`](2.3_neurovault_bounds.svg), [`2.3_bounds_table.tex`](2.3_bounds_table.tex).
Source: `core/a3_neurovault_bounds/outputs/neurovault_bounds.{png,svg}`, `core/a3_neurovault_bounds/outputs/bounds_table.tex`.

<a id="2.4"></a>
## 2.4: Data leakage

**Comment:** DOI filtering may miss duplicate records, preprint and journal versions, and repeated coordinate tables.

**Response:** We audited leakage beyond DOI filtering. No DOI or PubMed identifier is shared across splits. The three checks below flag 16 test and 26 validation papers (Table R8). Every evaluation leaves them out. We also removed seven NeuroVault papers that duplicate a training paper.

**1. Duplicate records.** Each held-out paper was compared with the training and NeuroVault papers by fuzzy title matching and by the similarity of its title and abstract. Among the papers they flag is one paper stored as two records.

**2. Preprint and journal versions.** The same checks flag a preprint and its journal version, which carry different DOIs.

**3. Repeated coordinate tables.** The coordinate tables from PubMed and Neurosynth Compose were compared across all splits. Identical or near-identical tables mark a paper that repeats another paper's peaks.

**Other training data.** The synthetic maps of the anatomically centered head had pooled held-out papers, so we rebuilt them from training papers only. This fix and the removals each change AUC, Dice, and correlation by less than 0.01. The language model's training script also removes all evaluation papers from its corpus ([2.11](#2.11)).

| Check against other papers | Test set (2,987) | Validation set (2,987) |
|---|---|---|
| Near-identical title | 4 | 6 |
| Near-identical title and abstract | 3 | 6 |
| Identical coordinate table | 6 | 12 |
| Near-identical coordinate table | 12 | 21 |
| Removed from every evaluation | 16 | 26 |

Files: [`2.4_leakage_table.md`](2.4_leakage_table.md).
Source: `core/a4_leakage_audit/outputs/leakage_table.md`.

<a id="2.5"></a>
## 2.5: Competitive baselines

**Comment:** the generative tasks lack competitive baselines. Compare with existing methods and with simpler projection or retrieval baselines.

**Response:** **Text to brain.** Fig. R1 and Table R1 compare NeuroVLM's maps with existing approaches and simple baselines ([1.1](../reviewer_01/README.md#1.1)). NeuroVLM leads Neurosynth, NeuroQuery, mean and domain maps, and atlas labeling. It narrowly leads linear regression, the simple projection baseline, and leads the nearest training paper's map, the retrieval baseline. Text2Brain, a published model, trails NeuroVLM (Dice 0.49 against 0.53), most on papers outside its training corpus.

**Brain to text.** We know of no published model that writes text from group-level brain maps. Existing decoders NiCLIP and NeuroConText rank fixed texts, and NeuroVLM is slightly ahead of both (Fig. S3). A simpler MLP projector beats the Qformer and replaces it ([2.1](#2.1)). As a retrieval baseline, the nearest training caption is no closer to the paper's text than generated captions, and it carries details of other studies ([3.3](../reviewer_03/README.md#3.3)).

![](2.5_text_to_brain_baselines.png)

Files: [`2.5_text_to_brain_baselines.png`](2.5_text_to_brain_baselines.png), [`2.5_text_to_brain_baselines.svg`](2.5_text_to_brain_baselines.svg), [`2.5_text_to_brain_baselines_table.tex`](2.5_text_to_brain_baselines_table.tex).
Source: `core/a1_text_to_brain_baselines/outputs/text_to_brain_baselines.{png,svg}`, `core/a1_text_to_brain_baselines/outputs/text_to_brain_baselines_table.tex`.

<a id="2.6"></a>
## 2.6: Summarized and synthetic text

**Comment:** describe how the summarized and synthetic texts were generated, and give representative examples.

**Response:** One prompt generated the paper summaries and another generated the synthetic and curated texts. Both prompts are given below, and Table R9 gives a complete example of each kind of text.

**Summaries.** Llama-3.3-70B-Instruct summarized each paper with the first prompt as its system message, removing low-level study details. The user message was the paper's title and abstract, joined by a separator token. Decoding sampled with temperature 0.3 and top-p 0.9, so a rerun words each summary differently. Summaries re-created for test-set papers cover the same topics as the released ones. The stage 1 projector learns from these summaries.

**Synthetic texts.** Qwen3.5-27B wrote short entries with the second prompt. The seeds were region names from standard brain atlases, Cognitive Atlas concepts, and the network names of our network atlases. The user message was the seed alone, such as "Insula". Decoding used temperature 0.1 and top-p 0.9, with thinking mode off. Sampling each seed several times gave re-wordings of one concept. The anatomically centered head learns from these entries.

**Curated texts.** The stage 2 projector learns from 1,338 curated texts about networks, regions, and functions. Qwen3.5-27B wrote them in the same way. The original generation code was not kept. A script in the code repository repeats the process, and its texts match the released texts of the same seeds in content (MiniLM cosine 0.88 on average).

```
Model: meta-llama/Llama-3.3-70B-Instruct
Decoding: sampling, temperature 0.3, top-p 0.9, top-k 50, at most 256 new tokens
User message: title[SEP]abstract
System message:

You are writing a short, topic-focused neuroscience mini-review.

INPUT:
You will receive:
    title + " [SEP] " + abstract from an fMRI publication.

GOAL:
Write a single coherent paragraph (4–6 sentences) that:
- Describes the scientific topic implied by the paper (e.g., meditation and the default mode
  network, exercise and working memory), not the paper’s methods.
- Explains the key cognitive processes (e.g., working memory, self-referential thought,
  emotion regulation) and the main brain systems or networks involved.
- May mention typical effects or relationships in general terms (e.g., “has been associated
  with changes in connectivity” or “can influence executive control”), but should stay high-level.

ALLOWED:
- General statements about effects or relationships (e.g., “exercise has been associated
  with improved cognition”, “meditation can alter DMN connectivity”).
- Generic references to “studies” or “research” in the field.
- Qualitative descriptions of brain activity or connectivity (e.g., “increased activation
  in prefrontal regions”, “changes in functional connectivity”).

AVOID:
- Exact numerical details (sample sizes, p-values, coordinates, time points).
- Detailed task or protocol descriptions from the specific study.
- Directly quoting or closely paraphrasing long sentences from the abstract.
- Overly narrow focus on this single experiment; write as if summarizing the topic area.

STYLE:
- Neutral, scientific tone, like a short review or encyclopedia entry.
- One paragraph only. No bullet points or headings.
- Focus on giving a clear sense of “what this topic is about” in the context of the field.
```

```
Write a short educational neuroscience entry for the provided fMRI concept.

Requirements:
- Explain the concept in general neuroscience terms.
- Always include relevant anatomical regions.
- Connect the anatomy to the cognitive or neural function.
- Use cautious, broad language; avoid overclaiming.
- Do not mention studies, patients, diseases, disorders, treatments, age groups, tasks, results, methods, or statistics.
- Do not use: critical for, responsible for, causal, proven, this study, activation, altered, reduced, increased.
- Do not invent highly specific subregions unless they are commonly associated with the concept.
- If the concept is broad, name the main commonly associated regions or networks.
- If the concept is anatomical, describe its broader network/function associations.
- Write 3-5 complete sentences.

Return valid JSON only:
{
  "title": "1-5 words",
  "description": "3-5 sentences"
}
```

| Text (used by) | Complete example |
|---|---|
| Summary of a training paper (stage 1 projector) | The human parietal and opercular regions are involved in processing somatosensory information. Somatotopy refers to the organization of these regions according to the location of sensory inputs on the body. The parietal cortex, including areas such as the postcentral gyrus and intraparietal sulcus, contains multiple somatosensory areas with distinct somatotopic representations. Attentional modulation can enhance activity in these areas, particularly in higher-order regions. |
| Synthetic text "Insula" (anatomically centered head) | The insula is a deep cortical region situated within the lateral sulcus of the cerebral hemisphere. This area maintains connections with structures such as the anterior cingulate cortex and the amygdala. Processes occurring in this region are often linked to perceptions of internal bodily states and emotional awareness. It contributes to the integration of sensory information with subjective experience across various brain networks. |
| Curated [NETWORK] text "Salience Network" (stage 2 projector) | The salience network commonly includes anterior insula and dorsal anterior cingulate cortex. It is associated with detecting and prioritizing behaviorally relevant internal and external information. Its connections with limbic, autonomic, and frontoparietal systems support flexible shifts between internal monitoring and externally oriented control. |

<a id="2.7"></a>
## 2.7: Dataset sizes

**Comment:** 3,000 or 4,553 NeuroVault maps, and 100 or 300 network images.

**Response:** As for comment [1.5](../reviewer_01/README.md#1.5), we recomputed every count from the released data and corrected the text and captions (Table R5). The supplement's 4,553 maps predate image quality control and the removal of papers already in PubMed. Neither network-image count can be reproduced, so the released counts replace both.

Files: [`2.7_counts_table.tex`](2.7_counts_table.tex).
Source: `core/a4_counts_table/outputs/counts_table.tex`.

<a id="2.8"></a>
## 2.8: Fig. 5 caption

**Comment:** the Fig. 5 caption gives both 0.78 and 0.85 for the NeuroVault AUC.

**Response:** The corrected caption names the curve behind each value. With each paper's best-matching map, the AUC is 0.85 from text to brain and 0.79 from brain to text (Fig. R10b). P-maps are inverted, so that high values mark strong effects, as for comment [1.3](../reviewer_01/README.md#1.3). Each curve has its own dashed chance curve, from texts shuffled across papers.

**Lower bound.** The lower bound and its chance curve lie below 0.50 by construction. The lower bound keeps the map that the model scores lowest for each paper. That map ranks below a typical candidate even when the texts are shuffled. The model still beats this chance level.

![Fig. R10](2.8_fig5_perf_quant.png)
*Fig. R10.* New Fig. 5. Contrastive recall on (a) PubMed and (b) NeuroVault, (c) autoencoder reconstruction, and (d) network classification. In (b), each paper's target is its best map (upper bound), every map (multi-target), or its worst map (lower bound), and dashed lines show chance.

Files: [`2.8_fig5_perf_quant.png`](2.8_fig5_perf_quant.png), [`2.8_fig5_perf_quant.svg`](2.8_fig5_perf_quant.svg), [`2.8_neurovault_bounds.png`](2.8_neurovault_bounds.png), [`2.8_neurovault_bounds.svg`](2.8_neurovault_bounds.svg).
Source: `../main/fig5_perf_quant.{png,svg}`, `core/a3_neurovault_bounds/outputs/neurovault_bounds.{png,svg}`.

<a id="2.9"></a>
## 2.9: SPECTER naming

**Comment:** SPECTER, SPECTER2 and Specter are used inconsistently. Name the exact model of each component.

**Response:** Every text-encoding step uses SPECTER2, the allenai/specter2_aug2023refresh model with its ad hoc query adapter ([1.6](../reviewer_01/README.md#1.6)). The original SPECTER is never used. We corrected every mention to SPECTER2. The language model is NeuroQwen3-0.6B, which is Qwen3-0.6B adapted to PubMed abstracts ([2.11](#2.11)). Llama-3.3-70B-Instruct wrote the paper summaries, and Qwen3.5-27B wrote the synthetic and curated texts ([2.6](#2.6)).

<a id="2.10"></a>
## 2.10: LoRA analogy

**Comment:** an MLP projection head is not a LoRA adapter.

**Response:** We agree and deleted the LoRA analogy ([1.7](../reviewer_01/README.md#1.7)). The head is a small network trained on frozen SPECTER2 embeddings. The revision calls it an anatomically centered projection head, not an adapter.

<a id="2.11"></a>
## 2.11: Training details

**Comment:** report parameters, GPU, training time, seeds, the Qwen fine-tuning strategy, and hyperparameters.

**Response:** Table R10, a new Supplementary Table, gives the parameters, training data, epochs, optimizer, learning rate, batch size, seed, and training time of every component. All components except the language model train on one RTX 5090 GPU, with fixed seeds and deterministic kernels. The whole pipeline, five-fold cross-validation included, trains from scratch in about 1.8 hours.

**Qwen fine-tuning strategy.** NeuroQwen3-0.6B is Qwen3-0.6B with all weights fine-tuned, without adapters, on PubMed titles and abstracts (Table R10). It trained for one epoch on one H100 GPU, and its training time was not recorded. Its training script removes the validation, test, and NeuroVault evaluation papers from this corpus. Comment [1.2](../reviewer_01/README.md#1.2) reports a memorization check.

| Component | Parameters | Training data (train / validation) | Epochs (selection) | Optimizer, learning rate, batch | Seed | Time |
|---|---|---|---|---|---|---|
| Autoencoder | 59.9 M | PubMed maps, 23,894 / 2,987 | 101 (lowest validation loss) | AdamW, 5e-5, 256 | 0 | 10 min |
| Contrastive heads, text and image | 0.59 M and 0.30 M | text and map latents, 23,894 / 2,987 | 301 (last) | AdamW, 1e-5, 2,048 | 123 | 6 s |
| Text-to-brain head | 0.59 M | text and map latents, 23,894 / 2,961 | 30 of at most 80 (best validation Dice) | AdamW, 1e-3, 512 | 0 | 22 s |
| Anatomically centered head | 3.8 M | synthetic pairs, 6,309 / 64, and PubMed training pairs | 200 of 2,000 (best PubMed validation recall) | AdamW, 2e-4, 2,048 | 123 | 103 s |
| Stage 1 projector (MLP) | 28.3 M | maps and LLM summaries, 22,547 / 2,837 | 4 of at most 30 (lowest validation loss, early stopping) | AdamW, 3e-4 with cosine decay, 32 | 17 | 17 min |
| Stage 2 projector (MLP) | 28.3 M | concept texts and maps, 972 / 134 | 30 (last) | AdamW, 1e-4 with cosine decay, 18 | 3 (best of 1–5) | 5 min per seed |
| NeuroQwen3-0.6B | 596 M | PubMed titles and abstracts, 1.22 M / 12,317 (385 M tokens) | 1 | AdamW, 5e-6 with cosine decay, 16 sequences of 1,024 tokens, bf16 | 12345 | not recorded (one H100) |

<a id="2.12"></a>
## 2.12: Random results and failure cases

**Comment:** Figs. 2 and 3 show selected examples. Add randomly selected results and representative failure cases.

**Response:** We added random results and failure cases for every evaluation set. They include random test papers, the worst test-set cases in both directions, the weakest atlas networks, and NeuroVault papers that fail on their best-matching map. Failures are not rare, since 16% of test-set captions have a negative similarity to the paper's text. Fig. R11 shows the six worst brain-to-text cases. The other five sets are in the code repository.

**Invented names.** With unsteered title decoding, 23% of test-set [NETWORK] titles and 9% of [REGION] titles matched no canonical or atlas name, such as "Cerebro-Cervico-Brain". The language model still generates every title, but it is steered toward the canonical names. At the title position, tokens that cannot continue a canonical network or region name are masked, and the model's own probabilities choose the name. Nothing is classified or looked up.

**Body text.** The body text and [FUNCTION] titles are generated without this mask. They can still contain invented words or stray fragments, such as "useary" in Fig. R11, a limitation the Discussion states.

**Non-Latin text.** A few captions contained non-Latin characters from the language model's multilingual vocabulary. Decoding uses Latin-script tokens only, which removes them.

![Fig. R11](2.12_pubmed_b2t_worst.png)
*Fig. R11.* Worst brain-to-text cases. The six PubMed test maps whose caption is least similar to the paper's text, two per page. Each shows the map in sagittal, coronal, and axial views, the paper's title and summary beside the complete generated caption, and the caption's similarity with its percentile.

Files: [`2.12_pubmed_b2t_worst.png`](2.12_pubmed_b2t_worst.png), [`2.12_pubmed_b2t_worst.svg`](2.12_pubmed_b2t_worst.svg), [`2.12_pubmed_b2t_worst_p2.png`](2.12_pubmed_b2t_worst_p2.png), [`2.12_pubmed_b2t_worst_p2.svg`](2.12_pubmed_b2t_worst_p2.svg), [`2.12_pubmed_b2t_worst_p3.png`](2.12_pubmed_b2t_worst_p3.png), [`2.12_pubmed_b2t_worst_p3.svg`](2.12_pubmed_b2t_worst_p3.svg), [`2.12_pubmed_t2b_worst.png`](2.12_pubmed_t2b_worst.png), [`2.12_pubmed_t2b_worst.svg`](2.12_pubmed_t2b_worst.svg), [`2.12_pubmed_t2b_worst_p2.png`](2.12_pubmed_t2b_worst_p2.png), [`2.12_pubmed_t2b_worst_p2.svg`](2.12_pubmed_t2b_worst_p2.svg), [`2.12_pubmed_t2b_worst_p3.png`](2.12_pubmed_t2b_worst_p3.png), [`2.12_pubmed_t2b_worst_p3.svg`](2.12_pubmed_t2b_worst_p3.svg), [`2.12_pubmed_random.png`](2.12_pubmed_random.png), [`2.12_pubmed_random.svg`](2.12_pubmed_random.svg), [`2.12_pubmed_random_p2.png`](2.12_pubmed_random_p2.png), [`2.12_pubmed_random_p2.svg`](2.12_pubmed_random_p2.svg), [`2.12_pubmed_random_p3.png`](2.12_pubmed_random_p3.png), [`2.12_pubmed_random_p3.svg`](2.12_pubmed_random_p3.svg), [`2.12_networks_bottom.png`](2.12_networks_bottom.png), [`2.12_networks_bottom.svg`](2.12_networks_bottom.svg), [`2.12_networks_bottom_p2.png`](2.12_networks_bottom_p2.png), [`2.12_networks_bottom_p2.svg`](2.12_networks_bottom_p2.svg), [`2.12_networks_bottom_p3.png`](2.12_networks_bottom_p3.png), [`2.12_networks_bottom_p3.svg`](2.12_networks_bottom_p3.svg), [`2.12_neurovault_b2t_best.png`](2.12_neurovault_b2t_best.png), [`2.12_neurovault_b2t_best.svg`](2.12_neurovault_b2t_best.svg), [`2.12_neurovault_b2t_best_p2.png`](2.12_neurovault_b2t_best_p2.png), [`2.12_neurovault_b2t_best_p2.svg`](2.12_neurovault_b2t_best_p2.svg), [`2.12_neurovault_b2t_best_p3.png`](2.12_neurovault_b2t_best_p3.png), [`2.12_neurovault_b2t_best_p3.svg`](2.12_neurovault_b2t_best_p3.svg), [`2.12_neurovault_t2b_best.png`](2.12_neurovault_t2b_best.png), [`2.12_neurovault_t2b_best.svg`](2.12_neurovault_t2b_best.svg), [`2.12_neurovault_t2b_best_p2.png`](2.12_neurovault_t2b_best_p2.png), [`2.12_neurovault_t2b_best_p2.svg`](2.12_neurovault_t2b_best_p2.svg), [`2.12_neurovault_t2b_best_p3.png`](2.12_neurovault_t2b_best_p3.png), [`2.12_neurovault_t2b_best_p3.svg`](2.12_neurovault_t2b_best_p3.svg).
Source: `core/a7_failure_cases_percentiles/outputs/a7_pubmed_b2t_worst.{png,svg}`, `core/a7_failure_cases_percentiles/outputs/a7_pubmed_b2t_worst_p2.{png,svg}`, `core/a7_failure_cases_percentiles/outputs/a7_pubmed_b2t_worst_p3.{png,svg}`, `core/a7_failure_cases_percentiles/outputs/a7_pubmed_t2b_worst.{png,svg}`, `core/a7_failure_cases_percentiles/outputs/a7_pubmed_t2b_worst_p2.{png,svg}`, `core/a7_failure_cases_percentiles/outputs/a7_pubmed_t2b_worst_p3.{png,svg}`, `core/a7_failure_cases_percentiles/outputs/a7_pubmed_random.{png,svg}`, `core/a7_failure_cases_percentiles/outputs/a7_pubmed_random_p2.{png,svg}`, `core/a7_failure_cases_percentiles/outputs/a7_pubmed_random_p3.{png,svg}`, `core/a7_failure_cases_percentiles/outputs/a7_networks_bottom.{png,svg}`, `core/a7_failure_cases_percentiles/outputs/a7_networks_bottom_p2.{png,svg}`, `core/a7_failure_cases_percentiles/outputs/a7_networks_bottom_p3.{png,svg}`, `core/a7_failure_cases_percentiles/outputs/a7_neurovault_b2t_best.{png,svg}`, `core/a7_failure_cases_percentiles/outputs/a7_neurovault_b2t_best_p2.{png,svg}`, `core/a7_failure_cases_percentiles/outputs/a7_neurovault_b2t_best_p3.{png,svg}`, `core/a7_failure_cases_percentiles/outputs/a7_neurovault_t2b_best.{png,svg}`, `core/a7_failure_cases_percentiles/outputs/a7_neurovault_t2b_best_p2.{png,svg}`, `core/a7_failure_cases_percentiles/outputs/a7_neurovault_t2b_best_p3.{png,svg}`.
