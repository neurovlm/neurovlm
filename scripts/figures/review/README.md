# Response to reviewers (`scripts/figures/review/`)

```
review/
  RESPONSE.md              the response: letter to the editor, then one short answer per comment with its figure or table
  response_letter.tex      the LaTeX response letter: title page with linked contents, each reviewer from a new page,
                           comments one after another (quoted verbatim in a small shaded box), the RESPONSE.md answers
                           with links between comments, figures Fig. R1, R2, ... and tables Table R1, R2, ..., each
                           with its caption (written here only)
  response_letter_with_manuscript.tex
                           the letter followed by the revised manuscript with the changes marked
                           (../../../../paper/sn-article-diff.pdf, latexdiff from the submitted version: additions in
                           blue, deletions in red), one page per page, with every line and page reference of the
                           changes lines linked to the marked page that holds it
  check_letter_sync.py     checks that the letter carries the RESPONSE.md text, tables, figure numbers and comment links
                           (--counts: words and numbers per answer)
  reviewer_01/ .. 03/      per reviewer: README.md (comment -> response -> figure with its letter caption -> files) + the
                           figures / tables, named by comment;
                           <name>_parts.tex lists the pages of a multi-page figure the letter shows
  build_reviewer_dirs.py   rebuilds reviewer_0N/ from RESPONSE.md and core/ outputs (CPU, a few s)
  core/
    a1_* .. a8_*           one analysis per directory: run_analysis.py (recomputes outputs/), make_figure.py
                           (re-plots from outputs/ only, CPU, seconds), README.md, outputs/
    run_all.sh             runs every analysis in order, one GPU job at a time; logs in core/logs/
```

The paper figures (Figs. 2-6, S2-S4) are in [`../main/`](../main/README.md).  Every analysis drops the leakage-audit
items of `scripts/eval_exclusions/` (16 test-set papers, 7 NeuroVault papers / 29 maps) and uses the final model
`scripts/outputs/runs/final` unless its README says otherwise.

## Reproduce

```bash
cd neurovlm
bash scripts/pipeline/reproduce.sh --run-name final                          # model, validation (read by a5_mlp_projector) and
                                                                             # paper figures (read by a7_failure_cases_percentiles)
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
bash scripts/figures/review/core/run_all.sh                                  # every analysis (~2.5 h), then reviewer_0N/
FIGURES_ONLY=1 bash scripts/figures/review/core/run_all.sh                   # re-plot everything from outputs/ (CPU, ~2 min)
bash scripts/figures/review/core/run_all.sh a3_neurovault_bounds             # one analysis
.venv/bin/python scripts/figures/review/build_reviewer_dirs.py               # after editing RESPONSE.md or a figure
.venv/bin/python scripts/figures/review/check_letter_sync.py                 # letter = RESPONSE.md (0 mismatches)
cd scripts/figures/review && pdflatex response_letter.tex && pdflatex response_letter.tex   # two passes: contents, Fig. R refs
cd scripts/figures/review && for i in 1 2 3; do pdflatex response_letter_with_manuscript.tex; done
                                                         # after the paper and marked-paper builds (three passes: contents, refs, links)
```

The letter is set on US letter paper with 1 in margins (text block 6.5 x 9 in).  Each figure is shown at the full text
width, or at the fraction of it that `\rfigwidth{label}{fraction}` sets in the preamble (paper figures at their 5.15 in
print width, letter plots drawn 6.5 in wide at the width their content needs), and never taller than 0.8 of the text
height.  Reviewer comments and the changes lines are set in small type, captions in footnote size.  A reviewer's general
remarks (comment x.0) follow the reviewer heading directly, without a heading or contents entry.  Figures float (here, at the top of a page or on a page of its own)
but never past the end of its comment (`\FloatBarrier`); a figure with several pages (`<name>_p2`, ...) is shown one
page per letter page (only the figures RESPONSE.md shows; 2.12's other failure-case sets are in reviewer_02/ only).  The redrawn paper Figs. 2, 3, S3, and S4 are shown from the reviewer folders and Fig. 4 (Fig. R2, comment 1.1) from `../main/`, since build_reviewer_dirs.py copies no Fig. 4 into reviewer_01/.  Figures are numbered in order of use; each is shown once, and a later comment that needs it links to it ("Fig. R3",
with the comment it belongs to); a table is not repeated either (2.7 points to the 1.5 table).  RESPONSE.md names each figure in its image alt text
(`![Fig. R7](...)`) and refers to it in the text as "Fig. R7"; the letter uses `\ref`, and the checker compares both.
Tables work the same way: "Table R7" in RESPONSE.md, `Table~\ref{tab:...}` in the letter (the `\label` sits in the `\rtab`
caption; tables are numbered by the order of the `\rtab` calls).  A fenced code block of RESPONSE.md must equal a verbatim
box of the letter (same words), and Markdown list items match the letter's `\item`s.
A comment named in a response links to it: `[2.4](#2.4)` in RESPONSE.md (anchors `<a id="2.4"></a>` above the
headings), `\cmt{2.4}` in the letter; the checker requires the same links in both.  Figure and table captions are
written in the letter only; `build_reviewer_dirs.py` copies the figure captions into the reviewer READMEs.
Each "Changes to the manuscript" line names the paper section and its lines in the marked manuscript PDF
(`../../../../paper/sn-article-diff.pdf`, the copy the editor reads; `../../../../manuscript_changes.md`): `\msl{359}{363}` prints "l. 359–363", `\msp{14}` "p. 14" and
`\mspp{11}{12}` "pp. 11–12".  The standalone letter prints them as text.  `response_letter_with_manuscript.tex` reads
the letter unchanged, appends the marked manuscript with pdfpages (each page gets the target `ms.<page>`, opened to
fit the window) and defines `\mslink`, so each reference links to the page that holds it (46 numbered lines per manuscript
page in the marked PDF as in the clean one, so line l is on page (l + 22) / 46 as `\numexpr` rounds it).  After a paper
edit, rebuild the marked copy, re-measure the line numbers on it (words of the old and new marked PDFs aligned), update manuscript_changes.md and the changes lines, and rebuild both PDFs.  Facts only the authors can supply
stay as `[AUTHOR: ...]` placeholders, bold in the text with a red-on-yellow note (`\authornote`) in the letter and
red on yellow in the paper (`\authorfact`).

## What goes where

*Supplement* = a supplementary figure or table in the manuscript; *reviewer letter* = evidence in the response letter.
The same table opens [`RESPONSE.md`](RESPONSE.md).

| Item | Comments | Where it goes | Directory (`core/`) |
|---|---|---|---|
| Text-to-brain baselines, figure with 2 panels; retrieval metrics and hard negatives, table | 1.1, 2.5, 3.2 | supplement | `a1_text_to_brain_baselines`, `a1_retrieval_metrics_table` |
| Brain-to-text generation versus retrieval, table | 1.1, 3.3 | reviewer letter | `a1_brain_to_text_baselines` |
| Granularity / projection, figure with 3 panels | 3.3, 3.5 | supplement | `a2_granularity_tradeoff` |
| NeuroVault bounds and map types, figure with 2 panels and a table | 1.3, 2.3, 2.8 | supplement | `a3_neurovault_bounds` |
| Leakage audit, table | 2.4, 1.2 | reviewer letter; one sentence in Methods | `a4_leakage_audit` |
| Dataset counts and per-stage sizes, tables | 1.2, 2.2, 1.5 | reviewer letter (per-stage sizes table; the counts are stated in 1.2 "Final counts") | `a4_counts_table` |
| Data-flow diagram | 1.2, 2.2 | supplement | `a4_dataflow_diagram` |
| Table-caption audit, example table and figure | 3.8 | reviewer letter; one sentence in the Discussion | `a4_caption_audit` |
| MLP projector versus Qformer, figure with 3 panels | 2.1 | reviewer letter only (the manuscript names the MLP projector and no longer discusses the Q-Former) | `a5_mlp_projector` |
| Context-free encoder (word2vec), figure with 2 panels | 3.4, 3.7 | supplement | `a5_encoder_swap` |
| Latent dimensions, figure with 2 panels | 3.5, 3.1 | supplement | `a6_latent_interpretation` |
| Perturbation probes, figure with 3 panels and a table. Loss ablation, table | 3.5 | probes in the reviewer letter, loss ablation in the supplement (Table S10) | `a6_perturbation_probes`, `a5_loss_ablation` |
| [NETWORK] / [REGION] / [FUNCTION] correctness, figure; network-, anatomy- or cognition-only text to brain, table | 3.6 | reviewer letter | `a6_steering_tokens`, `a6_modality_restricted_t2b` |
| Failure cases (six sets of 3 pages, two cases per page; the letter shows the worst brain-to-text set) and example percentiles | 2.12, 3.9 | reviewer letter (the other five sets as separate files); percentiles in the Fig. 2, 3 and S4 captions and a letter table | `a7_failure_cases_percentiles` |
| Out-of-distribution detection, figure with 2 panels | 1.4 | supplement | `a7_ood_confidence` |
| Memory and latency, table | 1.8 | reviewer letter; one sentence in the paper | `a8_edge_benchmark` |
| Training details per component, table | 2.11 | supplement (shown in the letter) | `../../README.md`, `../../outputs/runs/final/*.summary.json`, NeuroQwen3-0.6B model card |
| Paper figures from the final model | 1.9, 2.8, 3.9 | paper (replace Figs. 2-6 and S2-S4) | `../main/` |

Comments no item answers (text-only replies in RESPONSE.md): 1.1 expert ratings (not run, existing expert labels and
automatic checks instead); 1.2 contrast-level evaluation (no contrast identifiers in the PubMed data, so the NeuroVault
per-map AUC of a3 stands in) and the NeuroQwen3-0.6B corpus (the text says the training script removes the validation,
test, and NeuroVault evaluation papers; the shipped model predates that filter and holds 2,906 of the 2,971 test
abstracts until the author's retrain, see `OPEN_ITEMS.md`; memorization check in `core/a4_memorization_check`); 2.6
summary and synthetic-text prompts and examples (the summary prompt, model and sampling settings come from
`docs/02_data/07_transform_text.ipynb`); 3.7 second half (line 672, no scaling analysis); 1.6, 1.7, 1.9 (text changes); 2.11
(table from the run summaries and the NeuroQwen3-0.6B model card); 1.0, 1.10, 2.0 and 3.0 (general remarks, overall
assessment and summary, answered by pointing to the comments); 1.11 (repository: `scripts/pipeline/reproduce.sh`, the CI
smoke test `.github/workflows/smoke.yml` that trains every stage for one batch on CPU, and `.coveragerc`).

## Dependencies beyond the base environment

Installed into `.venv`: `neuroquery`, `nimare`, `sentence-transformers`
(all-MiniLM-L6-v2 cached), `bert-score`, `rapidfuzz`, `pyahocorasick`, `gensim` (word2vec-google-news-300 cached).  Not
available offline: GloVe vectors, a larger sentence encoder, the 1.2 M-article Qwen adaptation corpus, the `dot` binary
(matplotlib fallback is used).  Text2Brain runs from its released checkpoint, mirrored and pinned on
`neurovlm/text2brain-mirror` (`core/a1_text_to_brain_baselines/README.md`).
