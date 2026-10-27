# Retrieval metrics and hard negatives (comment 1.1)
**Comment 1.1:** report recall@1/5/10, MRR and median rank besides the recall-curve AUC, and use hard negatives from the same cognitive domain. **Goes to** the supplement (one table).
```bash
.venv/bin/python scripts/figures/review/core/a1_retrieval_metrics_table/run_analysis.py --model-dir scripts/outputs/runs/final --device cuda --force   # 1 min (CPU 1 min)
.venv/bin/python scripts/figures/review/core/a1_retrieval_metrics_table/make_figure.py                                                                # tables, 2 s
```
**Inputs:** the final run's autoencoder latents and InfoNCE heads (`_final_model.py`), pinned HF texts, SPECTER2 latents, Cognitive Atlas and NeuroVault data (`_hf_pins.py`),
leakage exclusions (`scripts/eval_exclusions/`).
**Method:** contrastive space, both directions, PubMed test set (2,971) and NeuroVault (305 papers). Rank = 1 + candidates with a larger cosine. PubMed queries are also
ranked only against papers of their own cognitive domain (hard negatives; nearest Cognitive Atlas concept, 10 k-means domains) and against a random pool of the same size (random negatives, 5 draws).
**Outputs** (`outputs/`): `retrieval_metrics_table.{md,csv,tex}` (Table R2: PubMed, one block per candidate set, all / hard negatives / random
negatives, with a chance row each), `retrieval_metrics.csv` (every dataset and metric, with bootstrap CIs; NeuroVault and R@x % are here), `per_query_ranks.parquet`, `config.json`;
`retrieval_recall_curves.{png,svg}` is for reference only.
**Result:** PubMed text -> brain R@1/5/10 0.018 / 0.053 / 0.088, MRR 0.045, median rank 230; brain -> text 0.013 / 0.052 / 0.088, 0.041, 217. Hard negatives are
only slightly harder than random pools of the same size (text -> brain R@10 0.242 vs 0.283).
