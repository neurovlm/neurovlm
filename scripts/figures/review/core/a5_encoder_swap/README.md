# Context-free baseline (reviewer 3.4; figure also cited in 3.7)

SPECTER2 against the mean word2vec vector (Google News, 300-d) of each title + abstract; for both only the projection heads
are retrained, on the final model's latents with its recipe (stage A head -> InfoNCE heads, text-to-brain head v2, seed 123).
PubMed test set (n = 2,971): retrieval recall-curve AUC both ways, Dice (70th percentile) and voxel Pearson r.
```bash
.venv/bin/python scripts/figures/review/core/a5_encoder_swap/run_analysis.py --model-dir scripts/outputs/runs/final --device cuda --force  # GPU, ~2 min
.venv/bin/python scripts/figures/review/core/a5_encoder_swap/make_figure.py  # figure + table, CPU, 1 s
```
Inputs: `scripts/outputs/runs/final`, pinned HF data (`_hf_pins.py`), gensim's `word2vec-google-news-300` in `~/gensim-data`
(md5 a5e5354d40acb95f9ec66d5977d140ef; `gensim.downloader.load("word2vec-google-news-300")` fetches it).
Outputs: `outputs/a5_encoder_swap.{png,svg}`, `outputs/a5_encoder_swap_table.{csv,tex}`, `outputs/metrics.json` (row `reference`:
the final heads scored directly, 0.836 / 0.833 / 0.531 / 0.322), heads in `outputs/checkpoints/final/`.
Result (AUC t→b / b→t, Dice, r): SPECTER2 0.835 / 0.833 / 0.530 / 0.327, word2vec 0.809 / 0.808 / 0.516 / 0.307.
