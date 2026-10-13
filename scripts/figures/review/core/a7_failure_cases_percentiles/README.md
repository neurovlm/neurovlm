# a7_failure_cases_percentiles: failure cases (2.12) and example percentiles (3.9)

Reviewer 2.12: add randomly selected results and representative failure cases. Reviewer 3.9: where the examples fall in the distribution.
From the repository root, with `A7=scripts/figures/review/core/a7_failure_cases_percentiles`:
- `.venv/bin/python $A7/run_analysis.py --model-dir scripts/outputs/runs/final --device cuda --gen-batch 64` (GPU, ~8 min; generations are
  cached in `outputs/generated_*.csv` under their recipe and a fingerprint of the model files, `--force` regenerates them).
- `.venv/bin/python $A7/make_figure.py` (CPU, ~1 min).

Inputs: the final model, the paper figures' sidecars (`scripts/figures/main/*_text.json`) and the validator's stage-2 PubMed captions
(`scripts/outputs/validation/final`); leakage-audit papers excluded; generated maps divided by their maximum, as in the paper figures.
Outputs: pages `a7_{pubmed_b2t_worst,pubmed_t2b_worst,pubmed_random,networks_bottom,neurovault_b2t_best,neurovault_t2b_best}.{png,svg}`, then
`_p2`, `_p3`, ... (6 cases per set: maps, true text, complete generated text, score and percentile; a NeuroVault paper is scored by its best map).
Each page is one response-letter page: 6.5 in wide, at most 7.56 in tall (`FIG_LETTER_PAGE_H`, the letter's figure-page height), text 8 pt, brain
slices >= 0.9 in tall; two cases share a page where both fit with their complete texts, else a case has its own page. Pages per set: worst
brain-to-text 6 (every case needs ~4.5 in), worst text-to-brain 5, random 6, weakest networks 3, NeuroVault brain-to-text 4, NeuroVault
text-to-brain 4. The letter shows only the worst brain-to-text set (2.12); the other sets are copied to `reviewer_02/` for the reviewers.
`captions.tex` and `a7_example_percentiles.{md,csv}` (percentile = 100 x (#below + 0.5 #ties) / n), per-item scores in `*_items.parquet`.
Result (final model): 16 % of test-set captions (482 of 2,971) have a negative cosine with the true text, 7 % of text-to-brain maps (222) a
negative Pearson r; the rule-selected Fig. 3 rows (check6: High 30, 150, Medium 52, 117, Low 92, 74, `KEEP_TEXT_IDXS`) lie at
P86 / 77 / 52 / 36 / 16 / 15 (text to brain) and P79 / 87 / 62 / 43 / 18 / 20 (brain to text) of 305 papers.
Comment 2.m6 (`audit_names.py`, CPU, ~20 s; needs `../a6_steering_tokens/outputs/title_vocabulary.json`): titles outside the stage-14 bank of their
level, and caption bodies with an anatomical-looking phrase (ending in cortex / network / gyrus / area / lobe / system) that is no Harvard-Oxford,
AAL, Yeo, Du, Pauli, MNI-lobe or bank name. Test set (2,971): 827 [NETWORK] titles (28 %) and 300 [REGION] titles (10 %) are no bank title, 678 (23 %)
and 259 (9 %) no atlas name either; 41 % of the 8,913 captions have a body phrase that is no such name (mostly standard terms: limbic system,
frontal cortex, inferior frontal cortex) and 16 % one with a word found in no name (premotricity, parahippocampestral). Outputs:
`name_audit_{titles,bodies}.csv`, `name_audit.md` (10 seeded examples each). Both quoted titles are stage-2 test titles: "Cerebro-Cervico-Brain"
is the [NETWORK] title of PMID 32595302 and "Medial Personal Relevance Cortex" the [REGION] title of PMID 21715612 (worst cases 2 and 3; the
complete caption joins the three captions).

