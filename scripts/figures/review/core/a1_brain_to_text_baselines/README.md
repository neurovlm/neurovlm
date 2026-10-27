# Brain-to-text: generation vs nearest-neighbour retrieval (comments 1.1, 3.3)
**Comments:** 1.1 "comparison between generated outputs and their nearest training examples"; 3.3 "show that the generated outputs are more accurate and useful
than retrieved outputs". **Goes to** the reviewer letter (one table).
```bash
.venv/bin/python scripts/figures/review/core/a1_brain_to_text_baselines/run_analysis.py --model-dir scripts/outputs/runs/final --device cuda --force   # GPU, 15 min
.venv/bin/python scripts/figures/review/core/a1_brain_to_text_baselines/make_figure.py                                                                # table, < 1 s
```
**Inputs:** the final run (autoencoder, InfoNCE heads, stage 2 MLP projector; `_final_model.py`), NeuroQwen3-0.6B, MiniLM and deberta-xlarge-mnli (BERTScore) at
their pinned revisions, pinned PubMed texts and LLM summaries (`_hf_pins.py`), leakage exclusions. Generations are cached in `outputs/cache/` under a hash of the settings.
**Method:** for each PubMed test map (2,971) the generated description (Fig. 4 recipe) is compared with retrieval baselines (the training text nearest to the map in
the contrastive space; the text of the nearest training map), a random training text and a shuffled null, all scored against the paper's title + LLM summary by MiniLM
cosine and BERTScore F1, neither part of NeuroVLM.
**Outputs** (`outputs/`): `brain_to_text_baselines_table.{md,csv}` (the table), `results_table.csv` (all metrics, CIs, paired tests), `per_sample_metrics.parquet`, `texts_by_method.parquet`.
**Result:** MiniLM ranks generation first (0.464 vs 0.428); BERTScore ranks it below retrieval (0.552 vs 0.588) and a random training text (0.580). Generated texts are no closer to the
training corpus than real held-out texts (median nearest-training MiniLM cosine 0.784 vs 0.790).
