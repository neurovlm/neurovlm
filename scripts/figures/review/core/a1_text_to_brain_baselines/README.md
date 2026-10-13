# Text-to-brain: NeuroVLM vs simple baselines (comments 1.1, 2.5, 3.2)
**Comment 1.1:** score generated maps independently, against nearest-neighbour retrieval, mean or domain maps, linear regression and atlas labelling. **Goes to** the supplement (figure + table).
```bash
.venv/bin/python scripts/figures/review/core/a1_text_to_brain_baselines/run_text_to_brain.py --model-dir scripts/outputs/runs/final --device cuda   # 1 min (CPU 2 min); the first run adds 20 min of CPU for Text2Brain (cached)
.venv/bin/python scripts/figures/review/core/a1_text_to_brain_baselines/make_figure.py                                                              # 3 s
```
**Inputs:** the final run (`latent_neuro.pt`, `pmids_split.pt`, `autoencoder.pt`, `proj_head_text_mse.pt` = head v2, `proj_head_text_bce.pt` = submitted head),
pinned HF data (`_hf_pins.py`), leakage exclusions (`scripts/eval_exclusions/`). External baselines: Neurosynth v7 (NiMARE, `~/nimare_data`), NeuroQuery
(`fetch_neuroquery_model`), Harvard-Oxford atlas (nilearn), Text2Brain (below); their maps are cached in `outputs/cache/` (the Neurosynth term maps are reused by the a6_* analyses).
**Text2Brain** (Ngo et al., MICCAI 2021): the released checkpoint (Google Drive, sha256 `08e8af9a...`) and the upstream source (github.com/ngohgia/text2brain @ 6f29389, MIT)
are mirrored on `neurovlm/text2brain-mirror` (built by `mirror_text2brain.py`) and pinned in `scripts/hf_manifest.json` with SciBERT (`allenai/scibert_scivocab_uncased`; the checkpoint replaces every weight).
`text2brain/` is the upstream model code with two fixes for transformers 4.x: the encoder is called with `return_dict=False` (otherwise `_, embedding = ...` unpacks
the output's key names), and `_text2brain.py` drops the checkpoint key `encoder.embeddings.position_ids` (an `arange(512)` buffer no longer saved) before the strict load.
Inputs follow the model's device instead of CUDA whenever a GPU is visible; maps are computed on the CPU (`--t2b-device`), one text at a time with "/" removed, as
`predict.py`. Its 46x55x46 4 mm output grid is the mask grid. On 100 test texts the maps match the unmodified upstream code run under Python 3.8, torch 1.5.1 and
transformers 3.3.1 (r > 0.999999, differences below 4e-6 of the map peak, identical 70th-percentile masks), and two CPU runs give byte-identical maps.
**Method:** PubMed test set (2,971 papers). Each method maps title + abstract to a brain map, scored against the autoencoder reconstruction of the paper's map:
Dice of 70th-percentile masks and voxel Pearson r. Ridge = SPECTER2 -> latent, alpha chosen on validation Dice, same decoder. Domain mean = k-means (k = 10) on
training latents, query -> nearest domain by SPECTER2. Atlas = union of the Harvard-Oxford regions named in the text (2,597 queries), else the mean map.
Null = NeuroVLM map vs a shuffled paper. Nearest training paper (SPECTER2) and the submitted head are in the csv only. Training-corpus overlap (dagger): 1,295 test papers
are in the NeuroQuery corpus (NeuroQuery's training set and the pool of Text2Brain's unreleased training split), 1,386 in the Neurosynth v7 database (PMID match);
`text_to_brain_baselines_subsets.csv` scores every method again on the 1,676 papers outside the NeuroQuery corpus and the 1,157 outside both.
**Outputs** (`outputs/`): `text_to_brain_baselines.{png,svg}`, `text_to_brain_baselines_table.{csv,tex}`, `text_to_brain_baselines_subsets.csv`, `results_table_full.csv`, `ordering_tests.csv`, `per_query_metrics.parquet`, `run_config.json`.
**Result:** Dice / r: NeuroVLM 0.531 / 0.322, ridge 0.526 / 0.311, submitted head 0.492 / 0.321, Text2Brain 0.485 / 0.293, every other baseline <= 0.448 / 0.251.
Outside the NeuroQuery corpus (n = 1,676): NeuroVLM 0.525 / 0.300, ridge 0.521 / 0.288, Text2Brain 0.472 / 0.264, NeuroQuery 0.406 / 0.167.
