# Brain-to-Text Evaluation Metrics

This document describes every metric used in `new_neuropvlm_metrics_v2.ipynb` for
the Brain → Text direction. Each section covers what the metric measures, how it is
computed, its numerical scale, and how to interpret scores in the context of this task.

---

## Overview

| Metric | Type | Scale | What it asks |
|---|---|---|---|
| BERTScore F1 | Semantic token overlap | 0 – 1 | Do the tokens mean the same things? |
| BERTScore P / R | Precision / Recall breakdown | 0 – 1 | Is the generated text precise / complete? |
| Semantic Cosine Sim | Sentence-level meaning | −1 – 1 | Do the two sentences convey the same idea? |
| NeuroVLM Latent Sim | Domain-specific alignment | ~0 – 0.5 in practice | Does the generated text map to the same brain region in NeuroVLM's shared space? |
| Recall@1 | Retrieval rank | 0 – 1 | Given all generated texts in a group, how often does the correct brain rank #1? |

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
   - **Precision (P)** — average best-match similarity for tokens in the *generated* text against the reference. High P means the model did not hallucinate irrelevant content.
   - **Recall (R)** — average best-match similarity for tokens in the *reference* against the generated text. High R means important content from the reference was covered.
   - **F1** — harmonic mean of P and R. This is the primary summary metric.

### Scale
Raw BERTScore values are bounded away from 0 by construction — even completely
unrelated text scores ~0.3 just from shared punctuation and common words. The
meaningful range for DeBERTa-XL on English text is roughly 0.45–0.95:

| Range | Interpretation |
|---|---|
| F1 > 0.90 | Near-identical or very high-quality paraphrase |
| 0.80 – 0.90 | Good semantic overlap, minor gaps |
| 0.70 – 0.80 | Partial overlap, correct domain, some missing concepts |
| < 0.70 | Weak overlap; possibly wrong domain or major hallucination |

Short mode (one sentence vs one sentence) scores higher than long mode (paragraph
vs full abstract) because there is less content to cover and fewer ways to diverge.

### Observed scores
| Dataset | Mode | F1 | P | R |
|---|---|---|---|---|
| Networks | short | 0.722 | 0.708 | 0.738 |
| Networks | long | 0.520 | 0.459 | 0.601 |
| NeuroVault | short | 0.576 | 0.591 | 0.565 |
| NeuroVault | long | 0.509 | 0.490 | 0.530 |
| PubMed | short | 0.585 | 0.579 | 0.595 |
| PubMed | long | 0.488 | 0.472 | 0.506 |

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

| Range | Interpretation |
|---|---|
| > 0.85 | Sentences express essentially the same idea |
| 0.70 – 0.85 | Same topic, largely overlapping meaning |
| 0.50 – 0.70 | Related topic, meaningful differences in content |
| 0.30 – 0.50 | Loosely related; important conceptual gaps |
| < 0.30 | Likely wrong topic or major semantic mismatch |

### Observed scores
| Dataset | Mode | Sem Sim |
|---|---|---|
| Networks | long | 0.676 |
| Networks | short | 0.604 |
| PubMed | long | 0.412 |
| PubMed | short | 0.353 |
| NeuroVault | long | 0.371 |
| NeuroVault | short | 0.286 |

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
| Dataset | Mode | nvlm_sim |
|---|---|---|
| Networks | long | 0.357 |
| Networks | short | 0.333 |
| NeuroVault | long | 0.323 |
| NeuroVault | short | 0.324 |
| PubMed | long | 0.298 |
| PubMed | short | 0.310 |

The consistency across datasets and modes (~0.30–0.36) indicates the model is
reliably anchoring to the right brain concept regardless of output length.

---

## 4. Recall@1

### What it measures in plain terms

Suppose you run the evaluation on 30 PubMed samples. You now have 30 brain images
and 30 generated texts. **The 30 generated texts are the exact same outputs already
produced during the B2T evaluation run** — one text per brain image, nothing new is
generated. You do not look at any ground truth — you just ask:
**can each generated text find its own brain image in a lineup?**

You compute `nvlm_sim` between every possible brain–text pair, giving a 30×30
similarity matrix. For each brain image, you check whether its own generated text
scored higher than all 29 other texts. If yes, that is a **hit**. Recall@1 is the
fraction of brains that got a hit.

```
Recall@1 = (number of brains whose generated text ranked #1) / N
```

Chance level is `1/N` — with random texts, each of the 30 texts would equally
likely be the top match, so you'd expect 1/30 ≈ 3.3% hits by luck alone.

### How another generated text can rank higher — a concrete failure case

This is the key idea. You are measuring `nvlm_sim` between a brain image and
*every* generated text, not just its own. The question is: when brain_i looks at
all 30 texts floating in the shared embedding space, does its own text land closest?

Consider two closely related networks:

```
Brain A  (auditory network)  → generated: "auditory and speech processing"
Brain B  (language network)  → generated: "speech and language comprehension"
```

When you compute `nvlm_sim(brain_A, text_B)`, it might score **higher** than
`nvlm_sim(brain_A, text_A)` — because "speech and language comprehension" overlaps
heavily with auditory concepts in NeuroVLM's shared space. Brain A's own text
did not land closest to it. That is a Recall@1 miss.

```
Shared embedding space:

  text_B ("speech/language") ── brain_A (auditory) ──── text_A ("auditory/speech")
        ↑                              ↑
  closer to brain_A              its own text,
  than text_A is                 but farther away
```

Brain A's generated text was accurate — but not *specific* enough to beat a
neighbouring concept. BERTScore would not catch this; it only compares text_A
against the ground truth, never against the other brain images. Recall@1 catches it.

### What this means for the evaluation

| Compared against | Metric | What it tests |
|---|---|---|
| Ground truth text | BERTScore, Semantic Sim | Is the generated text accurate? |
| All other brain images in the group | Recall@1 | Is the generated text specific? |

A model that always outputs "brain network involved in cognition" would score
reasonably on BERTScore for many samples but near chance on Recall@1 — because
that generic text matches every brain equally. Recall@1 is the only metric here
that penalises generic correct-sounding outputs.

### A concrete example

Say N=4 with these brains and generated texts:

```
Brain A  → generated: "Motor cortex for movement execution"
Brain B  → generated: "Default mode network for self-referential thought"
Brain C  → generated: "Visual cortex for object recognition"
Brain D  → generated: "Auditory cortex for speech encoding"
```

You build the 4×4 matrix of `nvlm_sim(brain_i, text_j)`:

```
             text_A   text_B   text_C   text_D
brain_A  [  0.35     0.12     0.09     0.11  ]   ← text_A ranks #1  ✓
brain_B  [  0.10     0.38     0.11     0.08  ]   ← text_B ranks #1  ✓
brain_C  [  0.08     0.11     0.33     0.14  ]   ← text_C ranks #1  ✓
brain_D  [  0.14     0.09     0.15     0.30  ]   ← text_D ranks #1  ✓
```

Recall@1 = 4/4 = 1.0. All four generated texts uniquely identified their brain.

If instead brain_C's highest match was text_A (wrong), Recall@1 = 3/4 = 0.75.

### How it is computed

```python
# z_brains: (N, D) — brain embeddings in shared space
# z_texts:  (N, D) — generated text embeddings in shared space

sim_matrix = z_brains @ z_texts.T                               # (N, N)
# diagonal = similarity of each brain to its own generated text
# off-diagonal = similarity to other texts
ranks = (sim_matrix > sim_matrix.diag().unsqueeze(1)).sum(dim=1)  # how many texts scored higher
recall_at_1 = (ranks == 0).float().mean()                       # fraction where nothing scored higher
```

### Observed scores

| Dataset | Mode | N | Chance (1/N) | Recall@1 | Above chance |
|---|---|---|---|---|---|
| Networks | long | 8 | 0.125 | **1.000** | 8× |
| Networks | short | 8 | 0.125 | **1.000** | 8× |
| NeuroVault | long | 30 | 0.033 | **0.800** | 24× |
| NeuroVault | short | 30 | 0.033 | **0.733** | 22× |
| PubMed | long | 30 | 0.033 | **0.700** | 21× |
| PubMed | short | 30 | 0.033 | **0.600** | 18× |

### Why this is the most important metric

Recall@1 tests **specificity** — whether the model generated something unique to
that brain image or something generic enough to match any brain. BERTScore and
semantic similarity only measure whether the text resembles the ground truth;
they cannot detect a model that always outputs the same correct-sounding sentence.
Recall@1 will expose that immediately — such a model would score near chance.

The results above are strong: even on the hardest setting (PubMed short, 30 samples),
the model is 18× above chance. Networks at 100% means every generated text is
discriminative enough to uniquely identify its source brain among all 8 networks.

---

## Summary: Which metric to lead with

| Situation | Primary metric | Supporting metrics |
|---|---|---|
| Reporting overall generation quality | **BERTScore F1** | Semantic Sim |
| Assessing domain-specific concept alignment | **NeuroVLM Latent Sim** | — |
| Testing whether outputs are specific, not generic | **Recall@1** | NeuroVLM Latent Sim |
| Diagnosing precision vs coverage tradeoff | **BERTScore P vs R** | — |

For a single headline number, **Recall@1** is the most meaningful and directly
interpretable — it has a known chance baseline, requires no reference text at
evaluation time, and tests the property that actually matters: can the model
generate text specific enough to identify the brain it came from?
