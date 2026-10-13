# Data-flow diagram: which data train which component, and what is held out (comments 1.2, 2.2)
**Comment 1.2:** a data-flow diagram of the data used for autoencoder training, contrastive training, language-model adaptation, Q-Former training, synthetic
fine-tuning, validation and testing. **Goes to** the supplement (one figure).
```bash
.venv/bin/python scripts/figures/review/core/a4_dataflow_diagram/run_analysis.py --model-dir scripts/outputs/runs/final --force   # counts, CPU, 5 s
.venv/bin/python scripts/figures/review/core/a4_dataflow_diagram/make_figure.py                                                  # figure, 1 s
```
**Inputs:** pinned HF data (`_hf_pins.py`; data-dir copies written by `scripts/data/fetch_hf.py` are sha256-checked), the run's split and stage 11 / 12 / 14
summaries, `scripts/eval_exclusions/` (evaluation counts only). Arrows follow `scripts/models/`: 08 -> 09 -> 12 and 13 -> 14 (12 does not load 13).
**Outputs** (`outputs/`): `a4_dataflow.{png,svg}`, `dataflow_counts.json` (every number in the figure), `a4_dataflow.dot` (detailed graph, not in the response).
**Result:** split by paper 23,894 / 2,987 / 2,987; 2,971 test papers evaluated after the leakage audit. Anatomically centred head: 6,309 / 64 synthetic pairs built
from training-split maps only (stage 11 `--splits train`; the notebook recipe pooled 1,596 test and 1,623 val papers, see `scripts/README.md`, "Stage 11 split
leak (fixed)"). Stage 1 projector 22,547 / 2,837 papers with an LLM summary; stage 2 projector 972 / 134 of 1,338 curated texts. NeuroVault (305 papers, 3,154
maps) and the network maps (145 atlas maps, 119 of them with a network label, + 40 ICA) are evaluation only. The NeuroQwen3 box (1.23 M PubMed titles and abstracts, 1,219,296 train /
12,317 val, from the DAPT notebook log) states the corpus rule of `scripts/models/00_neuroqwen_dapt.py`, whose training script excludes the
validation, test, and NeuroVault evaluation papers. The shipped NeuroQwen3-0.6B (rev 23f50c9) predates that filter and holds 2,906 of the 2,971
evaluated test papers, so the box (and the paper's sentence) holds only after the retrain (final_thoughts.md, open item 1).
