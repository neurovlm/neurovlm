# Generation vs retrieval, study-specific details, collapse (reviewer 3.3; numbers cited in 3.1)
**Comment.** 3.3: are generated captions more useful than retrieved ones, does the canonical projection make outputs retrieval-like, how granular are they.
```bash
.venv/bin/python scripts/figures/review/core/a2_granularity_tradeoff/run_analysis.py --model-dir scripts/outputs/runs/final --device cuda   # GPU, ~14 min
.venv/bin/python scripts/figures/review/core/a2_granularity_tradeoff/make_figure.py                                                        # CPU, 2 s
```
**Inputs:** the final run (`scripts/outputs/runs/final`: autoencoder, heads, stage-1/2 MLP projectors, banks, latents, split), the validator's PubMed
stage-1/2 captions (`scripts/outputs/validation/final/df_gen_{pubmed,canonical}.parquet`, batch 512: run `scripts/pipeline/validate_models.py` first), pinned HF data
and models (`_hf_pins.py`), leakage exclusions. Stage 2 without projection and NeuroVault (305 papers) are generated here (batch 128). Without `--force`,
cached captions (`generated_*.csv`) are reused; `outputs/cache/` holds the training bank and the stage-2 texts (rebuilt from `text_synth_less.parquet` if absent).
**Outputs** (`outputs/`): `a2_granularity_tradeoff.{png,svg}`, `metrics.json` (all numbers), `table.csv`, `per_text_*.parquet`, `collapse_*.parquet`,
`lexicon_summary.csv` and `lexicon_top_terms.json` (`lexicon.py` detects study-specific details), `gen_vs_retrieval.csv` (10 compositional text-to-brain queries).
**Result** (PubMed test set, n = 2,971): captions naming a study detail 35.5 % (stage 1) -> 1.4 % (stage 2; 2.4 % without the projection, batch 128); MiniLM
cosine to the true text 0.465 (stage 2) vs 0.428 (nearest training caption), BERTScore 0.552 vs 0.588; 56.5 % of retrieved captions carry another study's details;
stage-2 texts are closer to the training texts (median 0.88, real 0.79) but 99.4 % distinct. Compositional queries: mean min r 0.29 vs 0.17, generation ahead in 7/10.
