# NeuroVLM ALE3DCNN/GNN Evaluation Plan

This plan makes semantic ranking the central comparison axis for the pretrained NeuroVLM MLP baseline, ALE3DCNN, CoordGNN, CoordDeepSet, ALEFlatMLP, and DiFuMo GAT runs. Exact PubMed paper retrieval stays in the report, but it is treated as a strict diagnostic rather than the main success criterion.

## Official PubMed Splits

All model comparisons should load the PubMed/publications dataframe and use the existing split columns when present:

- `test == True` defines the held-out test PMIDs.
- `val == True` defines validation PMIDs when available.
- `train == True` defines training PMIDs when available.
- A deterministic random split is allowed only when the split columns are missing, and the run must print a warning.

Every run must save:

- `train_pmids.json`
- `val_pmids.json`
- `test_pmids.json`

The run log should print/assert the count of PMIDs in each split. These manifests are the comparison contract: future CNN/GNN runs should use the same held-out PMIDs as the NeuroVLM MLP baseline.

## A. Exact PubMed Paper Retrieval

Artifact prefix: `exact_pmid_retrieval`

Report:

- paper-style full recall-curve AUC over `k=1...N`
- recall@1, recall@5, recall@10, recall@50
- MRR
- median rank
- random recall@10

Interpretation: this asks whether the brain map ranks the exact source paper highest. It is deliberately strict and can be low even when the model is semantically useful.

## B. Network Labeling

Use raw network maps from `neurovlm/embedded_text`, file `networks_arrays.pkl.gz`. Preprocess all models with the shared NeuroVLM network resampling/thresholding logic. The only model-specific step should be the encoder:

- NeuroVLM MLP: `masker.transform -> autoencoder.encoder -> proj_head_image`
- ALE3DCNN DiFuMo-compatible: convert to the trained DiFuMo-compatible 3D CNN input volume.
- ALE3DCNN atlas-free: convert to the trained atlas-free CNN input volume without DiFuMo.
- ALEFlatMLP: convert to the trained flat ALE vector.
- CoordGNN/CoordDeepSet: either document active-voxel coordinate extraction or skip as too artificial.
- DiFuMo GAT: use the same DiFuMo projection and graph construction as training.

Label text should be embedded as:

```text
{term} [SEP] {definition}
```

Candidate labels should include at least visual, motor/sensorimotor, auditory, language, default mode, frontoparietal control, attention, and cingulo-opercular/salience.

Save:

- `network_labeling_metrics.json`
- `network_labeling_predictions.csv`
- `network_confusion_matrix.png`
- `network_one_vs_rest_auc.png`

## C. PubMed MeSH Term Ranking

Use the PMID-to-MeSH JSON map as multi-positive semantic ground truth. For each PubMed test paper, rank MeSH term candidates using brain embedding vs projected SPECTER/SPECTER2 term embeddings.

Candidate text should use term plus definition when available:

```text
{term} [SEP] {definition}
```

Report:

- MeSH recall@1, recall@5, recall@10, recall@50
- MAP
- MRR
- NDCG@10
- median best-rank of any true MeSH term
- average rank of true MeSH terms

Save:

- `mesh_term_ranking_metrics.json`
- `mesh_term_predictions.csv`
- `mesh_term_topk_examples.csv`
- `mesh_term_candidate_corpus.json`

## D. Other Semantic Term Ranking

When cognitive, anatomical, neuroscience concept, or phenotype/disease corpora are available, rank those candidates with the same term-plus-definition embedding pattern. Save separate metrics files:

- `cognitive_term_ranking_metrics.json`
- `anatomical_term_ranking_metrics.json`
- `phenotype_term_ranking_metrics.json`

## E. Semantic-Neighborhood Paper Retrieval

For each test paper, compute nearest neighbors in SPECTER/SPECTER2 text space. Count exact paper plus top 5 or top 10 text-nearest papers as acceptable semantic positives.

Report:

- semantic recall@10
- semantic recall@50
- semantic MRR
- semantic paper-style recall-curve AUC when practical

Save:

- `semantic_neighbor_retrieval_metrics.json`
- `semantic_neighbor_examples.csv`

## Main Comparison Table

The comparison notebook should include:

- Exact PubMed paper retrieval: paper recall-curve AUC, recall@1/5/10/50, MRR, median rank.
- Network labeling: accuracy, top-2 accuracy, macro AUC.
- MeSH term ranking: recall@5, recall@10, MAP, MRR, median best true term rank.
- Other term ranking: cognitive/anatomical/phenotype metrics when available.
- Resource use: params, peak VRAM, training time per epoch.

## Interpretation

If a model does not improve exact PMID recall, that is not automatically a failure. Exact paper retrieval is strict and may be underdetermined by coordinate-derived maps. The intended use case is semantic interpretation: better network labels, MeSH tags, cognitive/anatomical/phenotype terms, and semantic-neighborhood retrieval.

CNN/GNN variants should be considered better than the NeuroVLM MLP baseline only if they improve at least one semantic evaluation on the same official test split.
