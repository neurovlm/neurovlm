# NeuroVLM semantic evaluation resources

Upload this folder once to Google Drive as:

`/content/drive/MyDrive/neurovlm_evaluation_resources`

The ALE3DCNN, DiFuMo GAT, and CoordGNN Colab notebooks now all read that same
resource directory. Model outputs are still isolated in model-specific folders,
for example `runs_ale_3dcnn`, `data_difumo_gat`, and `checkpoints_coord_gnn`.

Contents:

- `networks_labels/network_test_set_labels.csv`: reusable ground-truth mapping from raw network labels to canonical network classes.
- `networks_labels/network_terms_with_definitions.csv`: optional richer corpus for network-term ranking, created by `experiments/create_network_term_definition_corpus.ipynb` from `network_name`, `cognitive_terms`, and `region_terms`.
- `mesh_kg/mesh_annotations.json`: `{pmid: [mesh_term, ...]}` semantic labels for PubMed MeSH ranking.
- `mesh_kg/mesh_kg_nodes.parquet`: MeSH term semantic types used to filter MeSH ranking to `cognitive_construct`, `biological_process`, `anatomical_region`, and `disorder`. The main MeSH metric intentionally excludes `molecular`, `organism`, `other`, `method`, and `demographic`.

The raw network arrays (`networks_arrays.pkl.gz`), PubMed text/brain latents, masks, model heads, and MeSH definition table are loaded through NeuroVLM retrieval-resource functions or Hugging Face, so they are not duplicated here.
