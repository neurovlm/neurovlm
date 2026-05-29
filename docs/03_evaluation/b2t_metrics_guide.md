# Brain-to-Text Evaluation Metrics

This document describes every metric used in `21_brain_to_text_metrics.ipynb` for
the Brain → Text direction. Each section covers what the metric measures, how it is
computed, its numerical scale, and how to interpret scores in the context of this task.

Brain-to-text metric primitives such as BERTScore, sentence semantic similarity,
and NeuroVLM latent similarity live with the brain-to-text workflows in
`src/neurovlm/brain_to_text_metrics.py`. Shared retrieval-curve helpers live in
`src/neurovlm/retrieval_metrics.py`, and `src/neurovlm/metrics.py` remains only as
a backward-compatible import index for older code.

---

## Overview

| Metric                                 | Type                         | Scale                | What it asks                                                                                   |
| -------------------------------------- | ---------------------------- | -------------------- | ---------------------------------------------------------------------------------------------- |
| BERTScore F1                           | Semantic token overlap       | 0 – 1                | Do the tokens mean the same things?                                                            |
| BERTScore P / R                        | Precision / Recall breakdown | 0 – 1                | Is the generated text precise / complete?                                                      |
| Semantic Cosine Sim                    | Sentence-level meaning       | −1 – 1               | Do the two sentences convey the same idea?                                                     |
| NeuroVLM Latent Sim                    | Domain-specific alignment    | ~0 – 0.5 in practice | Does the generated text map to the same brain region in NeuroVLM's shared space?               |
| Generated-text normalized Recall@k AUC | Retrieval rank curve         | 0 – 1                | Across the full ranked list, how early does each generated text retrieve its source brain map? |

All metrics are **higher = better**.

---

## 1. BERTScore

### What it measures

BERTScore computes token-level similarity between the generated text and the reference
using contextual embeddings from a pretrained language model. Unlike BLEU/ROUGE it
does not require exact word overlap — synonyms and paraphrases score highly if they
map to nearby vectors.

### How it is computed

1. Both strings are tokenised and encoded by the model (`microsoft/deberta-xlarge-mnli`,
   well-calibrated for scientific text).
2. Each token in the generated text is matched to its closest token in the reference
   (and vice versa) using cosine similarity in the contextual embedding space.
3. Three values are produced:
   - **Precision (P)** — average best-match similarity for tokens in the _generated_ text against the reference. High P means the model did not hallucinate irrelevant content.
   - **Recall (R)** — average best-match similarity for tokens in the _reference_ against the generated text. High R means important content from the reference was covered.
   - **F1** — harmonic mean of P and R. This is the primary summary metric.

### Scale

Raw BERTScore values are bounded away from 0 by construction — even completely
unrelated text scores ~0.3 just from shared punctuation and common words. The
meaningful range for DeBERTa-XL on English text is roughly 0.45–0.95:

| Range       | Interpretation                                             |
| ----------- | ---------------------------------------------------------- |
| F1 > 0.90   | Near-identical or very high-quality paraphrase             |
| 0.80 – 0.90 | Good semantic overlap, minor gaps                          |
| 0.70 – 0.80 | Partial overlap, correct domain, some missing concepts     |
| < 0.70      | Weak overlap; possibly wrong domain or major hallucination |

Short mode (one sentence vs one sentence) scores higher than long mode (paragraph
vs full abstract) because there is less content to cover and fewer ways to diverge.

### Observed scores

| Dataset    | Mode  | F1    | P     | R     |
| ---------- | ----- | ----- | ----- | ----- |
| Networks   | short | 0.722 | 0.708 | 0.738 |
| Networks   | long  | 0.520 | 0.459 | 0.601 |
| NeuroVault | short | 0.576 | 0.591 | 0.565 |
| NeuroVault | long  | 0.509 | 0.490 | 0.530 |
| PubMed     | short | 0.585 | 0.579 | 0.595 |
| PubMed     | long  | 0.488 | 0.472 | 0.506 |

### Caveats

- DeBERTa is a general-purpose model. It may not recognise neuroscience-specific
  synonyms (e.g. "DMN" vs "default mode network") unless seen in pretraining.
- Very short generated strings (< 5 tokens) can produce unstable scores.

---

## 2. Semantic Cosine Similarity

### What it measures

The sentence is encoded as a single fixed-length vector by a sentence transformer
(`all-MiniLM-L6-v2`). The cosine similarity between the generated and reference
vectors measures overall meaning alignment at the sentence level rather than
token by token.

### How it is computed

```python
emb_gen = model.encode(generated, convert_to_tensor=True)
emb_gt  = model.encode(gt_text,   convert_to_tensor=True)
score   = cosine_similarity(emb_gen, emb_gt)
```

### Scale

| Range       | Interpretation                                   |
| ----------- | ------------------------------------------------ |
| > 0.85      | Sentences express essentially the same idea      |
| 0.70 – 0.85 | Same topic, largely overlapping meaning          |
| 0.50 – 0.70 | Related topic, meaningful differences in content |
| 0.30 – 0.50 | Loosely related; important conceptual gaps       |
| < 0.30      | Likely wrong topic or major semantic mismatch    |

### Observed scores

| Dataset    | Mode  | Sem Sim |
| ---------- | ----- | ------- |
| Networks   | long  | 0.676   |
| Networks   | short | 0.604   |
| PubMed     | long  | 0.412   |
| PubMed     | short | 0.353   |
| NeuroVault | long  | 0.371   |
| NeuroVault | short | 0.286   |

Networks scores highest because the GT descriptions are structured and distinctive.
NeuroVault/PubMed short scores are lower because the LLM generates a generic
neuroscience sentence that captures the domain but not the specific paper topic.

### Caveats

- `all-MiniLM-L6-v2` is a small general model and may conflate unrelated
  neuroscience terms that appear in similar contexts.
- Fine-grained factual correctness (lateralisation, anatomy) is not captured.

---

## 3. NeuroVLM Latent Similarity

### What it measures

This is the most domain-specific metric. It asks: **if you re-encode the generated
text through NeuroVLM's own text encoder, does it land near the original brain image
in the shared latent space the model was trained in?**

A generation that correctly identifies the brain network will re-encode to a point
close to the brain image embedding. A hallucinated or off-topic generation will land
far away, even if it scores well on general embedding metrics.

### How it is computed

```
brain latent (384-d)
    → proj_head_image  (InfoNCE projection head)
    → L2-normalise
    → z_brain  (shared space)

generated text (string)
    → SPECTER encoder  (768-d)
    → proj_head_text_infonce  (InfoNCE projection head)
    → L2-normalise
    → z_text  (shared space)

nvlm_sim = cosine_similarity(z_brain, z_text)
```

Both projection heads are trained jointly with InfoNCE loss to pull matching
brain–text pairs together and push non-matching pairs apart in the same space.

### Scale

The InfoNCE shared space has inherently lower absolute cosine values than generic
embedding spaces — this is by design. The model learns to discriminate, not to
saturate values toward 1.0. As a reference, the top retrieved context terms at
inference time sit at **0.25–0.35**, which is the same range as the observed
`nvlm_sim` scores (~0.30–0.36). A generated text landing in that range is entering
the same concept region the model uses for retrieval — that is a good result.

### Observed scores

| Dataset    | Mode  | nvlm_sim |
| ---------- | ----- | -------- |
| Networks   | long  | 0.357    |
| Networks   | short | 0.333    |
| NeuroVault | long  | 0.323    |
| NeuroVault | short | 0.324    |
| PubMed     | long  | 0.298    |
| PubMed     | short | 0.310    |

The consistency across datasets and modes (~0.30–0.36) indicates the model is
reliably anchoring to the right brain concept regardless of output length.

### Optional scale check for interpretability

If you care about making `nvlm_sim` easy to interpret for readers, it is worth
adding the NeuroVLM scale-check plot as a contextual diagnostic. This is not a
separate success criterion; it is a sanity check that helps answer: **is a cosine
value like 0.32 actually meaningful in this embedding space?**

The notebook does this by comparing two distributions within each dataset/mode
group:

1. **Matched pairs** — each generated text is compared with the brain map that
   originally produced it.
2. **Random/off-diagonal pairs** — each generated text is compared with all other
   brain maps in the same group.

In code, `generated_text_pair_baseline(...)` re-embeds the generated texts with
NeuroVLM's text encoder and projection head, projects the corresponding brain
latents into the same shared space, then builds the full text-by-brain cosine
similarity matrix. The diagonal of that matrix is the matched distribution. The
off-diagonal entries are the random-pair baseline.

The resulting histogram is useful because absolute cosine values in an InfoNCE
space are not meant to be read like ordinary sentence-transformer scores. A value
around 0.30 can be strong if matched pairs are clearly shifted above random pairs.
If the matched and random distributions overlap heavily, then `nvlm_sim` is less
interpretable on its own, even if the mean value looks reasonable.

Use this plot when presenting results to make the scale intuitive:

```python
baseline_df = generated_text_pair_baseline(
    nvlm=nvlm,
    b2t_all=b2t_all,
    networks_data=networks_data,
    pubmed_eval=pubmed_eval,
    neurovault_eval=neurovault_eval,
)
```

The notebook saves this diagnostic as:

```
b2t_nvlm_sim_scale.png
```

---

## 4. Generated-text normalized Recall@k AUC

### What it measures in plain terms

Suppose you run the evaluation on 30 PubMed samples. You now have 30 brain images
and 30 generated texts. **The 30 generated texts are the exact same outputs already
produced during the B2T evaluation run** — one text per brain image, nothing new is
generated. You do not look at any ground truth — you just ask:
**how early can each generated text find its own brain image in a lineup?**

You compute NeuroVLM shared-space similarity between every generated text and every
brain map in the same dataset/mode group, giving a 30×30 similarity matrix. For
each generated text, the notebook ranks all 30 brain maps and records the rank of
the source brain map.

```
Recall@k = fraction of generated texts whose source brain map appears in the top k
normalized k = k / N
generated_text_normalized_k_recall_curve_auc = area under Recall@k over normalized k
```

The notebook uses the full curve and reports the normalized AUC. This is less
brittle than a single top-rank decision and is easier to compare across datasets
with different numbers of examples, because the x-axis is always `k / N`.

Chance follows the diagonal: at normalized `k = 0.10`, random ranking should recover
about 10% of source brain maps; at normalized `k = 0.50`, random ranking should
recover about 50%. A good model rises above that diagonal early.

### How another generated text can rank higher — a concrete failure case

This is the key idea. You are measuring NeuroVLM similarity between each generated
text and _every_ candidate brain map, not just its own. The question is: when a
generated text looks at all 30 brain maps floating in the shared embedding space,
does its source brain land near the top?

Consider two closely related networks:

```
Brain A  (auditory network)  → generated: "auditory and speech processing"
Brain B  (language network)  → generated: "speech and language comprehension"
```

When you compute similarity between `text_A` and all candidate brains, Brain B might
rank above Brain A because "auditory and speech processing" overlaps heavily
with auditory concepts in NeuroVLM's shared space. Brain A's own generated text
still may be relevant, but it did not rank its source brain first.

```
Shared embedding space:

  brain_B (language) ── text_A ("auditory/speech") ──── brain_A (auditory)
        ↑                              ↑
  ranks above brain_A            generated from brain_A,
  for this text                  but not closest to it
```

Under normalized Recall@k AUC, it is penalized according to how far down the ranked
list the source brain falls. Ranking the source brain second is much better than
ranking it twentieth.

Brain A's generated text may be accurate but not _specific_ enough to beat a
neighbouring concept. BERTScore would not catch this; it only compares text_A
against the ground truth, never against the other brain images. The normalized
Recall@k AUC catches this specificity problem.

### What this means for the evaluation

| Compared against                    | Metric                                 | What it tests                                                    |
| ----------------------------------- | -------------------------------------- | ---------------------------------------------------------------- |
| Ground truth text                   | BERTScore, Semantic Sim                | Is the generated text accurate?                                  |
| All other brain images in the group | Generated-text normalized Recall@k AUC | Is the generated text specific across the ranked retrieval list? |

A model that always outputs "brain network involved in cognition" would score
reasonably on BERTScore for many samples but close to the random diagonal on the
normalized Recall@k curve, because that generic text matches every brain equally.
Generated-text normalized Recall@k AUC is the metric here that penalizes generic
correct-sounding outputs in retrieval space.

### A concrete example

Say N=4 with these brains and generated texts:

```
Brain A  → generated: "Motor cortex for movement execution"
Brain B  → generated: "Default mode network for self-referential thought"
Brain C  → generated: "Visual cortex for object recognition"
Brain D  → generated: "Auditory cortex for speech encoding"
```

You build the 4×4 matrix of NeuroVLM similarity between generated texts and brain maps:

```
             brain_A  brain_B  brain_C  brain_D
text_A   [   0.35     0.12     0.09     0.11  ]   ← brain_A ranks #1  ✓
text_B   [   0.10     0.38     0.11     0.08  ]   ← brain_B ranks #1  ✓
text_C   [   0.08     0.11     0.33     0.14  ]   ← brain_C ranks #1  ✓
text_D   [   0.14     0.09     0.15     0.30  ]   ← brain_D ranks #1  ✓
```

At `k = 1`, Recall@k = 4/4 = 1.0. The curve reaches perfect recall immediately, so
the normalized AUC is high. All four generated texts uniquely identified their brain.

If instead text_C ranked its source brain second, the curve would be lower at
`k = 1` but would recover by `k = 2`. That is worse than perfect top-1 retrieval,
but much better than ranking the source brain last.

### How it is computed

```python
# z_texts:  (N, D) — generated text embeddings in shared space
# z_brains: (N, D) — brain embeddings in shared space

scores = z_texts @ z_brains.T
order = scores.argsort(dim=1, descending=True)
target = torch.arange(N).view(-1, 1)

# rank position of the source brain map for each generated text
first_hits = order.eq(target).int().argmax(dim=1)

# recall_curve[k - 1] = fraction of source brains found within top k
hit_counts = torch.bincount(first_hits, minlength=N).float()
recall_curve = torch.cumsum(hit_counts, dim=0) / float(N)
normalized_k = torch.arange(1, N + 1) / float(N)

generated_text_normalized_k_recall_curve_auc = normalized_recall_curve_auc(recall_curve)
```

### Output files

The notebook saves the generated-text retrieval results to:

- `b2t_generated_text_recall_auc.csv/json`: one AUC row per dataset/mode group.
- `b2t_generated_text_recall_curve.csv/json`: full Recall@k curve rows.
- `b2t_generated_text_normalized_recall_curve.png`: visual comparison of the curves.

### Why this is the most important metric

Generated-text normalized Recall@k AUC tests **specificity** — whether the model
generated something unique to that brain image or something generic enough to match
any brain. BERTScore and semantic similarity only measure whether the text resembles
the ground truth; they cannot detect a model that always outputs the same
correct-sounding sentence. The recall curve exposes that immediately: such a model
would follow the random diagonal instead of rising above it early.

---

## Summary: Which metric to lead with

| Situation                                         | Primary metric                             | Supporting metrics  |
| ------------------------------------------------- | ------------------------------------------ | ------------------- |
| Reporting overall generation quality              | **BERTScore F1**                           | Semantic Sim        |
| Assessing domain-specific concept alignment       | **NeuroVLM Latent Sim**                    | —                   |
| Testing whether outputs are specific, not generic | **Generated-text normalized Recall@k AUC** | NeuroVLM Latent Sim |
| Diagnosing precision vs coverage tradeoff         | **BERTScore P vs R**                       | —                   |

For a single headline number, **generated-text normalized Recall@k AUC** is the most
meaningful retrieval-style metric. It has a known chance baseline, requires no
reference text at evaluation time, works across different group sizes, and tests the
property that actually matters: can the model generate text specific enough to
identify the brain it came from early in the ranked list?
