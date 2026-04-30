# Why BLEU and ROUGE Are the Wrong Metrics for Brain → Text

## The core problem: n-gram overlap ≠ meaning

BLEU and ROUGE measure **exact n-gram overlap** between a generated string and a reference string.
BLEU-4 specifically requires matching sequences of **4 consecutive words**. A paraphrase — even a
perfect one — scores near zero.

Consider a concrete example from the Networks evaluation:

| | Text |
|---|---|
| **Generated** | "Default mode network links self-referential thoughts and memory." |
| **Ground truth** | "Default mode network for self-referential thought and memory retrieval." |

These sentences convey identical meaning. BLEU-4 finds only one 4-gram match (`default mode network
for` is not present; the longest exact overlap is 3 words) and scores it close to **0.000**. ROUGE-1
counts unigram overlap so it does slightly better, but still penalises "thoughts" vs "thought" and
"retrieval" vs its absence.

This is not a model failure — it is a metric failure.

---

## Why it gets worse at longer outputs

For the **long mode** (free-form paragraph vs technical abstract), the problem compounds:

- The model generates a well-organised description of a brain network.
- The reference is a PubMed-style abstract written in domain jargon with specific boilerplate phrases.
- Even a *human expert paraphrase* of the abstract would score ~0.005 BLEU-4 because the wording
  convention is completely different.
- ROUGE-L (longest common subsequence) partially rescues recall, but precision is still hurt by any
  sentence the model generates that the abstract does not literally contain.

The low BLEU/ROUGE numbers observed in the evaluation (BLEU-4 ≈ 0.002–0.008, ROUGE-1 ≈ 0.05–0.28)
**do not mean the model is generating garbage**. They mean the model is paraphrasing, which is
exactly what a good language model should do.

---

## Why the short / PubMed title case is also misleading

For PubMed short mode the LLM is asked to generate a paper *title*. Titles are highly
formulaic — word order and exact phrasing are important for BLEU — but a generated title like
*"Cortical activation patterns in acoustic neuroma patients"* compared against the reference
*"Auditory cortex responses to pure-tone stimuli in acoustic neuroma"* would score near zero despite
being semantically on-target. The model correctly identified the domain, anatomy, and task; it just
used different words.

---

## What goes wrong statistically

| Metric | What it assumes | Why that breaks here |
|---|---|---|
| BLEU-n | Surface n-grams are meaningful units of quality | Scientific paraphrases share meaning but not exact phrases |
| ROUGE-1/2 | Word/bigram overlap correlates with quality | Domain synonyms and morphological variants break overlap |
| ROUGE-L | Longest common subsequence captures structure | Sentence re-ordering (common in paraphrase) breaks LCS |
| Token-F1 | Token overlap | Same problem as ROUGE-1; no synonym awareness |

---

## Better metrics: why they work

### BERTScore
Uses contextual embeddings (e.g. DeBERTa) to match tokens by meaning rather than identity.
"thoughts" and "thought", "retrieval" and "recall" map to nearby vectors — they score high even
without exact overlap. For scientific text `microsoft/deberta-xlarge-mnli` is well-calibrated.

### Sentence embedding cosine similarity
Models like `all-MiniLM-L6-v2` compress full sentences into a single semantic vector. Cosine
similarity between the generated and reference vectors measures **overall meaning alignment**,
entirely bypassing word-level matching.

### METEOR
Adds synonym matching (via WordNet) and stemming on top of unigram overlap. This catches
"thought" ↔ "thoughts", "linked" ↔ "linking", "retrieval" ↔ "retrieve". It is a direct
improvement over ROUGE-1 for paraphrase-heavy outputs.

---

## Summary

> BLEU and ROUGE were designed for machine translation in the early 2000s, where the output space
> is constrained and near-literal correspondence with a reference translation is a reasonable proxy
> for quality. Brain-to-text generation is an *open-ended semantic decoding task* — the model must
> infer meaning from an activation map and express it in natural language. No single phrasing is
> correct; semantic fidelity is what matters. The right evaluation tools are those that measure
> **meaning overlap**, not **surface form overlap**.
