# a4_leakage_audit: duplicates of held-out papers (comment 2.4)

Reviewer 2.4: DOI filtering alone may miss duplicate records, preprint and journal versions, or repeated coordinate tables.
From the repository root (CPU; ~40 s with the cached MiniLM embeddings, ~15 min without):
- `.venv/bin/python scripts/figures/review/core/a4_leakage_audit/run_analysis.py --force` compares every test, validation and NeuroVault
  paper with the training papers, the other held-out split and the NeuroVault papers by DOI / PMID, fuzzy title (token-set ratio >= 95 and
  token-sort >= 90), MiniLM title + abstract cosine >= 0.95 and identical or near-identical (Jaccard >= 0.8, >= 5 peaks) coordinate tables.
  `--write-exclusions scripts/eval_exclusions` also rewrites the lists every evaluation reads (`run_all.sh` runs without it).
- `.venv/bin/python scripts/figures/review/core/a4_leakage_audit/make_figure.py` -> `outputs/leakage_table.{md,csv}` (the letter's table).

Inputs: pinned HF publications, coordinates, split and pubmed_summary_qa; the released NeuroVault tables; the NeuroVault and synthetic tables
that `scripts/data/fetch_hf.py` puts in the data dir. Outputs: `removal_list.csv`, `neurovault_exclusions.csv`, `flagged_pairs.csv`,
`max_similarity_*.csv`, `summary.json`, `summary_table.csv`, embeddings in `cache/`. Result: no DOI or PMID is shared; 16 test, 26 validation
and 7 NeuroVault papers (29 maps) are removed; synthetic texts are at most 0.84 cosine from any held-out paper. The stage-11 leak of the
synthetic maps and its fix: `scripts/README.md`, "Stage 11 split leak (fixed)".
