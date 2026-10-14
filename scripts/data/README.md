# `scripts/data/` — data preprocessing (ports of `docs/02_data/01..07`)

```bash
PY=.venv/bin/python                       # from the repo root
$PY scripts/data/fetch_hf.py              # every input, verbatim from Hugging Face at pinned commits -> ~/.cache/neurovlm
$PY scripts/data/fetch_hf.py --check      # verify the data dir and HF cache against the manifest, download nothing
$PY scripts/data/run_all.py               # same as fetch_hf.py (--from-hf is the default)
$PY scripts/data/run_all.py --regenerate --smoke   # every stage on 64 samples (~1 min) -> scripts/outputs/data_0N_*/smoke_artifacts/
$PY scripts/data/run_all.py --regenerate --skip-optional   # provenance: stages 01-07 -> scripts/outputs/data_regenerated/
$PY scripts/data/07_transform_text.py --regenerate --batch-size 32 --artifact-dir scripts/outputs/data_regenerated
$PY scripts/data/08_synthetic_texts.py --limit 6 --n-per-seed 2   # proof: re-create synthetic texts through the HF router
$PY scripts/data/09_summaries.py                                  # proof: re-create 6 test-set summaries through the HF router
```

## Inputs: `fetch_hf.py` and `scripts/config/hf_manifest.json`
Training, validation and figures read only files pinned in `scripts/config/hf_manifest.json`: 120 files in 23 Hugging Face
repos (the `neurovlm` datasets and models plus SPECTER2, all-MiniLM-L6-v2, deberta-xlarge-mnli, Qwen3-0.6B-Base and
wikitext), each with its commit, sha256, size, the name it gets in the data dir (`local_name`), its groups (`pipeline`,
`figures`, `regenerate`, `extra`) and the scripts that read it. Three more repos are pinned by revision only (`pins`:
Llama-3.1-8B for `07 --regenerate`, the roberta-large BERTScore fallback, the BrainGPT split). `fetch_hf.py` downloads
the files at those commits and checks every sha256; `--only pipeline|figures|all` (default all, ~23 GB), `--data-dir`,
`--check` (verify, download nothing), `--overwrite`. Files the library loaders read from the HF cache stay there, and
the cache's `refs/main` is set to the pinned commit (the loaders call `hf_hub_download` without a revision, local files
first). A file that is present but differs from the pinned one is reported and never replaced without `--overwrite`: a
stale copy would load other data or weights (NeuroAutoEncoder's weights on HF changed in May and went back in July).

Inputs that were not published before went into three dataset repos on 2026-10-09: `neurovlm/neurovault_maps` (the
NeuroVault scrape, the 4,611 maps as a tar with a sha256 list, and stage 03's `neurovault.pt` /
`neurovault_publications.parquet`), `neurovlm/synthetic_pairs` (`synth.parquet`, `text_synth_less.parquet`) and
`neurovlm/pubmed_ale_maps` (the original `neuro_vectors_ale.pt`). In the released `neurovault_collections.parquet`, 24
e-mail addresses (22 owner names, one contributor, one description) were replaced with `[email removed]`; no script
reads those columns. An unredacted local copy from before the release is therefore reported as a mismatch by
`--check`: replace it with `--overwrite`, or use a fresh data dir (`--data-dir`, `reproduce.sh --home`).

## Regeneration (provenance)
Each stage writes the same files as its notebook to `--artifact-dir` (stage default `neurovlm.data.data_dir`;
`run_all.py --regenerate` passes `scripts/outputs/data_regenerated/` so the fetched release is never overwritten;
`<out>/smoke_artifacts/` with `--smoke`). The recipe is the notebook's; only batching, parallelism, caching and paths
differ (listed per stage below). Stage 01 (`--ijk-mode trunc`, default) gives the released ALE maps and the released
map tensor to about 1e-4 (see below); stage 02 runs SPECTER2 in bf16 and is within 0.002 of the release. The pipeline
reads the released files.

Common flags: `--out` (timings, default `scripts/outputs/data_0N_*/`), `--device`, `--seed`, `--force` (recompute;
otherwise existing outputs and caches are reused), `--smoke` (64 samples; 01: 2 figure pairs, 06: `--min-count 2`,
07: 64 new tokens), `--max-samples`, `--batch-size`, `--artifact-dir`, `--amp-dtype bf16|fp16|fp32` (SPECTER2 on
CUDA; bf16 is the default, ~2x faster and ~1e-2 relative difference per dimension; fp32 = notebook numerics).

## `run_all.py`
`--from-hf` (default) runs `fetch_hf.py` (`--inputs pipeline|figures|all`, `--data-dir`, `--overwrite`, `--check`) and no
stage. `--regenerate` first fetches the pinned raw inputs (`--no-fetch` skips that), then runs 01..07 in order (they
only read released HF data, not each other) and passes `--smoke`, `--force`, `--device`, `--artifact-dir` to every
stage (`--smoke` also skips the fetch).  `--only 01 06`, `--skip 03`, `--skip-optional` (04, 05),
`--stage-args 07="--regenerate"`, `--continue-on-error`.  Prints a timing table and writes
`scripts/outputs/data_run_all/run_all[_smoke].json`.

## Stages

| # | script | outputs (artifact dir) | smoke¹ | full run |
|---|---|---|---|---|
| 01 | `01_coordinate.py` | `neuro_vectors_ale.pt`, `neuro_vectors.pt`, `difumo_examples/` | 13.0 s | ≈ 6–7 min (extrapolated) |
| 02 | `02_pubmed.py` | `latent_specter2_adhoc.pt`, `specter/` | 4.6 s | ≈ 45 s (extrapolated) |
| 03 | `03_neurovault.py` | `neurovault_publications.parquet`, `neuro_clust.pt`, `neurovault.pt`, `neurovault_cache/` | 10.9 s | ≈ 5.5 min (extrapolated, upper bound) |
| 04 | `04_neurowiki.py` (optional) | `latent_specter2_wiki.pt`, `latent_specter_wiki_aligned_adhoc_{mse,infonce}.pt`, `specter_wiki/` | 5.4 s | ≈ 50 s (extrapolated) |
| 05 | `05_cogatlas.py` (optional) | `latent_cogatlas.pt` | 4.7 s | 2.5 s (measured) |
| 06 | `06_n_grams.py` | `ngram_matrix.npy`, `ngram_labels.npy`, `ngram_emb.pt`, `ngram_cache/` | 4.6 s | 25 s (measured) |
| 07 | `07_transform_text.py` | `neuro_summaries.{jsonl,parquet}`, `neuro_summaries_sample.csv` | 2.6 s | 2.2 s release (measured); LLM see below |
| 08 | `08_synthetic_texts.py` (LLM proof) | `synthetic_texts.parquet` in `--out`, `examples/synthetic_texts_examples.md` | n/a | 12 requests, 6 s |
| 09 | `09_summaries.py` (LLM proof) | `summaries.parquet` in `--out`, `examples/summaries_examples.md` | n/a | 6 requests, 4 s |

¹ `run_all.py --smoke --force` on an RTX 5090 / 16 cores with HF caches populated.  Per-block timings are logged
in `scripts/outputs/data_0N_*/timing*/timings.jsonl`.  Only 05, 06
and 07 (release) were run in full (into a scratch `--artifact-dir`); 01–04 were timed on subsets at full settings,
because their real artifacts already exist in `~/.cache/neurovlm`.

### 01 — ALE smoothing + DiFuMo-512
* Inputs: HF `pubmed_text`, `pubmed_coordinates`, the masker; nilearn DiFuMo-512 and MNI152 (`~/nilearn_data`); `nimare`.
* `neuro_vectors_ale.pt` `{"neuro_vectors": [N, 28542] f32, "pmid"}` (3.5 GB): nimare ALE kernel, FWHM 9 mm, divided by
  the kernel peak, clamped to 1.  `neuro_vectors.pt` (same keys): DiFuMo maps resampled (nearest) to the mask,
  `X_hat = (X @ P / s) @ P.T`, all-zero rows dropped, divided by the row's 0.9999 quantile, clamped to [0, 1].
  `difumo_examples/NN_{ale,difumo}.png`: 20 seed-0 example pairs.
* vs notebook: ALE in parallel pmid chunks (`--n-jobs`; exact, maps are per study), projection in row chunks
  (`--chunk-rows`) in full fp32.  The pmids of `pubmed_text` and `pubmed_coordinates` must be identical (checked: 30,826).
* `--ijk-mode trunc|round` (default `trunc`): how an MNI focus becomes a voxel index of the 4 mm mask. The released
  maps (2025-12-18) were built from the current coordinate table with nimare <= 0.15, whose `mm2vox` truncates; nimare
  0.22 (the locked version) rounds, which moves 66 % of the foci by one voxel (ALE rows r ~0.63 vs the release, 48
  instead of 42 empty rows). Kernel, mask, coordinates and DiFuMo projection are unchanged. The script sets the indices
  itself, so the installed nimare does not matter. With `trunc` the ALE maps equal the original `neuro_vectors_ale.pt`
  (now `neurovlm/pubmed_ale_maps`) bit for bit on all 346 + 1,500 rows checked, and their projection gives the released
  tensor to 1.3e-4 (CPU vs GPU float32). `round` gives the nimare 0.22 maps.

### 02 — SPECTER2 of PubMed `title[SEP]abstract`
* `allenai/specter2_aug2023refresh` + `adhoc_query` adapter (`neurovlm.models.Specter`), unit norm, sorted by pmid:
  `latent_specter2_adhoc.pt` `{"latent": [N, 768], "pmid"}`; shards `specter/encoded_text_specter2_adhoc_*.pt`.
* An existing file without `--force` is extended with the missing pmids only (the notebook's `overwrite=False`).
* vs notebook: length-sorted batches of 128, bf16 on CUDA, shards of 1024 rows (`--shard-size`).

### 03 — NeuroVault evaluation set
* Inputs, from `neurovlm/neurovault_maps` via `fetch_hf.py` (`--input-dir`, default `data_dir`): `neurovault_collections.parquet`,
  `neurovault_images.parquet`, `neurovault_abstracts.parquet` (`--rescrape` rebuilds them from the NeuroVault API and
  CrossRef, slow), the hand-curated `neurovault_abstracts_missing.csv`, and `neurovault_images/neurovault_img_{id}.nii.gz`
  (4,611 files; `--download-images` fetches missing ones from the URLs in `neurovault_images.parquet`).
  `docs/figures/12` uses the released versions of the outputs instead (`load_dataset("neurovault_images")` etc.).
* Recipe: regex-cleaned abstracts -> `neurovault_publications.parquet`; unthresholded maps without ids 22136–22138 and
  without parcellation / anatomical / ROI/mask / variance maps; NaN -> 0, skip maps with < 10 unique values, nearest
  resample, mask, clip negatives; drop empty maps and maps with <= 50 unique values; `threshold_img("99%",
  cluster_threshold=50, two_sided=False)`, drop maps active in >= 10 % of voxels and empty maps; drop collections whose
  DOI is a PubMed training paper; SPECTER2 of titles, abstracts and `title[SEP]abstract`.
* `neuro_clust.pt` (clustered maps), `neurovault.pt` (`df_neuro`, `df_pubs`, `neuro`, `neuro_clustered`, `text_emb_*`).
* vs notebook: resampling and cluster thresholding run under joblib (`--n-jobs`) and are cached in
  `neurovault_cache/{neuro.pt, df.csv, neuro_clustered.npy}` (the notebook wrote these to its CWD).
* Full set: 4,611 images -> 3,183 maps / 312 publications.

### 04 — NeuroWiki (optional)
Nothing downstream reads it (the release ships `latent_specter_wiki.pt`).  SPECTER2 of `title [SEP] summary` for the
38,410 articles of HF `neuro_wiki`, id = `sha256(text)[:8]`, not normalised -> `latent_specter2_wiki.pt`
`{"latent": [N, 768], "id"}`; projected by the released `proj_head_text_mse` / `proj_head_text_infonce` and unit
normalised -> `latent_specter_wiki_aligned_adhoc_{mse,infonce}.pt` (`[N, 384]`).

### 05 — Cognitive Atlas (optional)
Nothing downstream reads it.  Lower-cased terms without "active cognitive inhibition" and "transfer data"; child -> parent
graph; the 300 nodes with the highest degree centrality; SPECTER2 of `term[SEP]definition` -> `latent_cogatlas.pt`
(bare `[295, 768]` tensor; 295 of the 300 nodes have a definition).

### 06 — n-gram vocabulary
* Corpus: `name [SEP] description` of `pubmed_text`.  One `CountVectorizer(ngram_range=(1, 3), stop_words="english")`
  pass (same features and counts as the notebook's three), count >= 100 (`--min-count`), drop terms matching a substring/regex list, rows in
  ascending pmid order, then the notebook's cleaning: multi-word "nouny" terms, unigrams of 2–6 letters, force-keep
  precuneus / working memory / putamen, sub-phrases merged into the longest kept phrase (OR of columns).
  Ties between super-phrases are broken alphabetically; the notebook used `set` order, which changes between runs.
* Manual filter: the notebook keeps only the terms of a hand-edited `corpus.txt`, **not in the repo or on HF**.
  `--corpus PATH` takes it (CSV with a `terms` column, or a `.npy` of strings).  `~/.cache/neurovlm/ngram_labels.npy`
  (2,458 terms, the notebook's output) reproduces the notebook: same labels in the same order, 99.83 % identical matrix
  cells (127 columns differ through the tie-break).  Default `--whitelist none`: no manual filter, 3,152 terms (a
  superset of the 2,458).  `--whitelist release`: intersect with `load_dataset("ngrams")` (3,298 terms of a later
  vocabulary; 1,295 overlap).
* `ngram_matrix.npy` (bool `[30826, n_terms]`, rows by ascending pmid), `ngram_labels.npy`, `ngram_emb.pt`
  (`[n_terms, 768]`, unit norm), cache `ngram_cache/`.

### 07 — LLM topic summaries
* Default (`--from-release`): writes the released `load_dataset("pubmed_summaries")` (30,826 rows, this notebook's
  output) to the notebook's three files.
* The released summaries were made with Qwen3.5-27B. `--regenerate` keeps the notebook's local model.
  `09_summaries.py` calls Qwen3.5-27B through the Hugging Face router.
* `--regenerate`: `meta-llama/Meta-Llama-3.1-8B-Instruct` (gated, ~16 GB, needs `HF_TOKEN`) in bf16, left padding,
  chat template with the notebook's system prompt, `temperature=0.3, top_p=0.9, top_k=50`, 256 new tokens, batch 16.
  Prompts run longest first; the jsonl is appended after every batch and a rerun resumes from it.  `--model`,
  `--max-new-tokens`, `--max-input-tokens`, `--prompt-format auto|chat|plain`.  With `--smoke`, if Llama is not in the
  HF cache, the cached `Qwen/Qwen3-0.6B-Base` is used with plain prompts (tests the code path only).
* Outputs (`--summaries-dir`, default the artifact dir; the notebook wrote to `docs/02_data/`): `neuro_summaries.jsonl`
  (`{doi, summary, pmid}`), `neuro_summaries.parquet` (`pubmed_text` columns + `summary`), `neuro_summaries_sample.csv`.
  Summaries are matched on pmid because 1,493 papers have no DOI and 1,498 DOIs repeat.
* The notebook's serial cell called `pipeline(...)` instead of `pipe(...)`; only the batched path is kept.
* LLM runtime: Qwen3-0.6B, 128 abstracts x 256 new tokens, batch 16: 63.8 s (332 tok/s) -> ≈ 4.3 h for 30,826.
  Llama-3.1-8B was not measured (weights not available here); decoding scales with weight size (~13x), so ≈ 55 h at
  batch 16 and ≈ 15–20 h at batch 48–64.

### 08 — synthetic texts (LLM proof of concept)
The curated [NETWORK]/[REGION]/[FUNCTION] texts of stage 14 and the LLM rows of `synth.parquet` came from one prompt,
the synthetic-text prompt of `docs/01_data/07_transform_text_gen.ipynb` (cell 8), sent to Qwen3.5-27B through the
Hugging Face Inference router with sampling, so that one concept gets several re-wordings. The original generation code
was not kept. This script repeats the process and is not run by `run_all.py` (it needs a token and the router bills it).
* Seeds: region names of the nilearn atlases (AAL SPM12 as in stage 11, labels written out without hemisphere, and
  Harvard-Oxford cortical and subcortical), Cognitive Atlas concept names (pinned `cogatlas.parquet`, without stage 05's
  two dropped terms), and the network names of the `networks` dataset (the eight standard names of `network_text.parquet`,
  then the atlas names of `networks_arrays.pkl.gz` without ICA components). 131 network, 108 region and 908 function
  seeds after removing duplicates.
* Call (cell 9): `InferenceClient(base_url="https://router.huggingface.co/v1")`, chat completion, system message = the
  prompt, user message = the seed alone, `temperature 0.1, top_p 0.9, max_tokens 512`, thinking off
  (`chat_template_kwargs`), JSON reply parsed into title and description, 4 retries. `--model` (default the notebook's
  `Qwen/Qwen3.5-27B:cheapest`), `--n-per-seed`, `--temperature`, `--top-p`, `--max-new-tokens`, `--workers`.
* Output `--out/synthetic_texts.parquet`: `level, seed, sample, title, description, raw_text, model, text, model_title,
  settings, timestamp`. As in `synth.parquet`, `title` is the seed and `text` is `seed [sep] description` in lower case.
* `--limit N` (default 6): N/3 seeds per level, seeds with a released text of the same name first, in random order
  (`--seed`). The run then writes `examples/synthetic_texts_examples.md` with each output, the released text of the
  same name and their MiniLM cosine. `--limit 0` runs every seed and writes no examples.
* Proof (2026-10-13, `--limit 6 --n-per-seed 2`): both `Qwen/Qwen3.5-27B` and `Qwen/Qwen3.5-27B:cheapest` answered
  through the router. Cosine to the released text of the same name 0.84 to 0.92 (12 texts), against a median of 0.48
  to all released texts.

### 09 — PubMed summaries (LLM proof of concept)
* The summary prompt of `docs/02_data/07_transform_text.ipynb` (cell 3) as system message, `title[SEP]abstract` of the
  pinned `publications.parquet` as user message, `Qwen/Qwen3.5-27B` through the router as in 08 (`--model`), thinking
  off (`chat_template_kwargs`), `temperature 0.3, top_p 0.9, max_tokens 256` and `top_k 50` sent as an extra field
  (`--top-k 0` leaves it out). Decoding samples, so the text differs on each run. Not run by `run_all.py`.
* `--limit N` (default 6): random test-set papers (`--seed`) that have both released summaries. The run writes
  `examples/summaries_examples.md` with the new summary next to `neuro_summaries.parquet` and `pubmed_summary_qa.parquet`
  and their MiniLM cosine and a format check (paragraphs, words, digits, non-Latin characters, finish reason).
  `--limit 0` summarizes every paper. Output `--out/summaries.parquet`.
* Proof (2026-10-14, Qwen/Qwen3.5-27B): mean cosine 0.88 to the released `neuro_summaries.parquet` text of the same
  paper, against 0.39 for other papers. Every summary is one paragraph of five sentences in Latin script, stops before
  the token limit and names no sample size, statistic or coordinate. New summaries are longer (153 words against 128).
  `pubmed_summary_qa.parquet` (59 words) gives 0.79.
