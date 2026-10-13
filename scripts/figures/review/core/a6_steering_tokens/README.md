# 3.6: are the [NETWORK] / [REGION] / [FUNCTION] caption titles correct? (final model)

Comment 3.6: "How does quantitative performance compare between these outputs?" Reviewer letter: one figure, one table.

```bash
.venv/bin/python scripts/figures/review/core/a6_steering_tokens/run_correctness.py --model-dir scripts/outputs/runs/final --device cuda  # GPU ~2 min; captions cached in outputs/cache
.venv/bin/python scripts/figures/review/core/a6_steering_tokens/plot_correctness.py  # figure at the letter width (6.5 in), CPU
.venv/bin/python scripts/figures/review/core/a6_steering_tokens/run_correctness.py --model-dir scripts/outputs/runs/final --strict-containment --out <dir>  # previous [REGION] rule
```

Inputs: the final run (`neuro_qformer_canonical/`, `canonical_semantic_banks.pt`, `autoencoder.pt`, `pmids_split.pt`), the pinned HF network atlases, PubMed maps and Cognitive Atlas, Neurosynth v7 (`--neurosynth-dir`; z-map cache copied from `a1_text_to_brain_baselines`, which `run_all.sh` runs first). 145 network-atlas maps, 40 ICA components and 150 PubMed test maps (seeded, leakage exclusions) are captioned once per token and only the title is scored: [REGION] names a region holding ≥ 5 % of the map's top-10 % voxels or its peak, or a region of which at least half lies in those voxels (containment rule, default; `--strict-containment` drops it); [NETWORK] names the curated (else best-overlapping Yeo-7) network; [FUNCTION] shares a synonym group with a top-5 Neurosynth-decoded term. Chance: titles shuffled across maps (5,000 permutations).

Outputs: `outputs/steering_correctness.{png,svg}`, `outputs/correctness_table.{csv,tex}` (final model; every token beats chance on every set, p < 0.001; [REGION] 93 / 85 / 63 % on network atlases / ICA / PubMed vs chance 24 / 33 / 25 %, with `--strict-containment` 88 / 78 / 59 % vs 23 / 30 / 23 %; [NETWORK] on PubMed 29 % vs 9 %), `outputs/correctness_per_caption.csv`, `outputs/correctness_results.json`, `outputs/examples.md`.

`correctness_results.json` also scores the captions of the removed released-model analysis (`outputs/cache/generations_9774092757ec_excl.parquet`, kept as an input; the comparison is skipped if it is absent).

**Containment rule ([REGION]).** The top 10 % of a continuous map is 2,854 voxels, so 5 % of it is 143 voxels: a small structure lying inside the map (amygdala ~90 voxels, caudate ~124, hypothalamus 26) could never count, however well the title names it. A named region therefore also counts when at least half of its own voxels lie in the map's top voxels (`MIN_INSIDE = 0.5`); this adds 7 / 3 / 7 hits on networks / ICA / PubMed and raises chance by 1-3 points. Of the PubMed misses that remain, most name a neighbouring region (within one 4 mm voxel) or a region chosen by the study topic; restricting [REGION] decoding to the region prototypes whose predicted maps overlap the input was tested (about 95 / 90 / 73 % with k = 50) but needs a library option and is left to the authors.

**Title constraint (comment 2.m6).** `--title-constraint vocab` lets the [NETWORK] / [REGION] title take only a title of the
final run's stage-14 bank of that level (48 network, 83 region names); `--title-constraint shortlist` also limits the [REGION] title
to the titles of the `--shortlist-k 50` region bank rows whose NeuroAdapter map (top 5 %) lies most inside the map's top voxels. A trie of the
names' token ids masks the logits until the newline that ends the title; the body is never constrained, [FUNCTION] is left free. The
implementation is the shared one of `scripts/models/_common_models.py` (`title_vocabulary`, `TitleConstraint`, `install_title_constraint`;
copied into `_hf_pins.py`), and the vocabulary is the shared `scripts/outputs/runs/final/title_vocabulary.json`. `vocab` has since become the
default decoding of every other caption (validation, paper figures, `pin_neuroqwen` of the review analyses; `--free-titles` turns it off);
this analysis keeps `none` (free titles) as its default, the baseline of the table below. Results go to `outputs/title_<mode>/`; captions
are cached next to the default ones.
```bash
.venv/bin/python scripts/figures/review/core/a6_steering_tokens/build_title_vocabulary.py   # once: runs/final/title_vocabulary.json (if missing; copy in outputs/), outputs/cache/bank_maps.npz (~1 min)
.venv/bin/python scripts/figures/review/core/a6_steering_tokens/run_correctness.py --model-dir scripts/outputs/runs/final --device cuda --title-constraint vocab      # ~2 min
.venv/bin/python scripts/figures/review/core/a6_steering_tokens/run_correctness.py --model-dir scripts/outputs/runs/final --device cuda --title-constraint shortlist  # ~2 min
.venv/bin/python scripts/figures/review/core/a6_steering_tokens/worst_case_titles.py   # the six worst test captions (Fig. R11), stage-2 recipe, ~1 min
```
| Titles (networks / ICA / PubMed) | [REGION] correct | [NETWORK] correct | bank titles among [REGION] / [NETWORK] titles |
|---|---|---|---|
| free (default) | 93 / 85 / 63 % | 70 / 50 / 29 % | 95 / 93 / 91 %, 85 / 75 / 73 % |
| `vocab` | 93 / 85 / 63 % | 70 / 53 / 33 % | 100 %, 100 % |
| `shortlist` (k = 50) | 93 / 90 / 73 % | 70 / 53 / 33 % | 100 %, 100 % |

Generation time is unchanged (within run-to-run noise). The shortlist ranks regions by the input map's own top voxels, the voxels the [REGION]
score also reads, so part of its gain is that alignment. In the stage-2 recipe the six worst test captions of Fig. R11 have 9 bank titles among
their 12 [NETWORK] / [REGION] titles, 12 with either option ("Cerebro-Cervico-Brain" -> Cerebro-Cerebellar Network, "Medial Personal Relevance
Cortex" -> Medial Prefrontal Cortex, or Ventrolateral Prefrontal Cortex with the shortlist; `outputs/worst_case_titles.csv`; the free run
reproduces 11 of the 12 released titles).

