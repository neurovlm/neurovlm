# a4_memorization_check: do stage-2 captions copy the paper's abstract? (comment 2.m5)

NeuroQwen3-0.6B was further pretrained on PubMed titles and abstracts without excluding the NeuroVLM test split: 2,906 of the
2,971 evaluated test papers (and about 98 % of the training papers) are in its corpus. If that pretraining leaked abstract
text into the captions, a test caption would share more verbatim text with its own abstract than a training caption does.

```bash
.venv/bin/python scripts/figures/review/core/a4_memorization_check/run_analysis.py --device cuda \
    --dapt-corpus "$HOME/efficient_ai/project/pubmed/pubmed_neuroimaging.parquet"   # GPU ~5 min (training captions, cached), CPU ~1 min
```

Needs `scripts/outputs/validation/final/df_gen_canonical.parquet` (scripts/pipeline/validate_models.py). The 2,971 stage-2 test captions are
read from it; 2,971 training papers (seed 0, among those with a map, an abstract and a PubMed summary) are captioned here with the
validation path and recipe (it imports `scripts/pipeline/validate_models.py`: greedy, 128 new tokens, projection temperature 0.035,
repetition penalty 1.25, batches of 128 in PMID order, seed 123 + batch start, canonical [NETWORK] / [REGION] titles). A caption is
its [NETWORK], [REGION] and [FUNCTION] captions together. References: the paper's title + abstract (pinned
`neurovlm/neuro_image_papers` publications.parquet, the text NeuroQwen3 was pretrained on), its PubMed summary (the stage-13
training target), and the same texts of another paper of the group (derangement, seed 0) as a null. Words: rouge_score's
tokenizer (lower-case alphanumeric runs). Per caption: the longest run of consecutive words shared with the reference, the share
of the caption's word 8-grams found in it, and ROUGE-L F1. Test vs training: medians, 95th percentiles, two-sided Mann-Whitney U.
`--dapt-corpus` (optional) flags which papers are in the pretraining corpus.

Outputs: `outputs/memorization_table.{md,csv}` (the table), `outputs/per_caption.parquet`, `outputs/summary.json`,
`outputs/cache/train_captions_*.parquet` (+ `.json` with the model sources).

Result (final model, canonical titles; captions have a median of 153 words in both groups; 2,906 test and 2,906 training papers
of the samples are in the pretraining corpus; 3 test papers have no abstract): the captions share no verbatim text with their own
abstract beyond short phrases. Longest shared run: median 2 words, 95th percentile 4, maximum 8 (test) / 7 (training), p = 0.21;
share of caption 8-grams found in the own abstract: median and 95th percentile 0 in both groups (one test caption holds one
8-gram, "the anterior insula dorsal anterior cingulate cortex and"; no training caption), p = 0.32; ROUGE-L F1 0.104 vs 0.105
(95th percentile 0.134 vs 0.133), p = 0.39, against 0.099 for another paper's abstract. Against the PubMed summaries: runs 2 / 2
words (95th percentile 4 / 4), ROUGE-L 0.129 / 0.129 (p = 0.92). Test captions do not copy their abstracts more than training
captions do, so the pretraining overlap did not carry abstract text into the captions.
