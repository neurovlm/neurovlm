# Unified Knowledge Graph Pipeline

Documents how the three source knowledge graphs — MeSH, NLP co-occurrence, and
CogAtlas — are built, typed, and merged into the final unified KG used by the R-GCN.

---

## Notebook Index

All pipeline notebooks live in `experiments/`. Run them in order to reproduce the
unified KG from scratch.

| # | notebook | what it does | key outputs |
|---|---|---|---|
| 1 | `mesh_kg.ipynb` | Parses `desc2026.xml` to build the MeSH descriptor table and hierarchy edges. Assigns semantic node types from tree-number prefixes. Produces the MeSH node table and hierarchy edge file. | `data/mesh_kg/mesh_descriptors.parquet`, `mesh_kg_nodes.parquet`, `mesh_kg_edges_hierarchy.parquet` |
| 2 | `mesh_kg_cooccurrence.ipynb` | Builds co-occurrence and typed relation edges from PubMed MeSH annotations. Combines hierarchy + co-occurrence + typed edges into the full MeSH edge file. | `data/mesh_kg/mesh_kg_edges_all.parquet` (305,227 edges) |
| 3 | `nlp_graph.ipynb` | Filters 133K raw NLP nodes down to 12,841 typed nodes using degree/strength/community/alias thresholds. Filters edges to both-endpoint survivors. Exports in unified KG schema. | `data/nlp_kg/nlp_kg_nodes.parquet`, `nlp_kg_edges.parquet` (155,352 edges) |
| 4 | `kg_unification.ipynb` | Merges MeSH + CogAtlas + NLP into a single canonical entity table. Assigns canonical IDs (MeSH UI > cogat_ > nlp_). Remaps NLP edges to canonical IDs. Saves initial unified KG. | `data/unified_kg/unified_kg_nodes.parquet` (33,784 nodes), `unified_kg_edges.parquet` (329,566 edges — pre-qualifier snapshot) |
| 5 | `mesh_qualifier_re.ipynb` | Extracts 14.79M semantically-typed edges from MeSH annotation qualifiers. Replaces generic `co_occurs_with` edges with `implicated_in`, `associated_with_disorder`, `treated_by`, `used_in` using qualifier suffix mappings. | `data/nlp_kg/nlp_kg_edges_qualified.parquet` (14,790,106 edges) |
| 6 | `llm_relation_extraction.ipynb` | Uses an LLM to extract a small set of high-confidence typed edges from paper abstracts that were missed by the qualifier approach. | `data/nlp_kg/nlp_kg_edges_llm.parquet` (52 edges) |
| 7 | `kg_edge_merge.ipynb` | Merges the original 329K edges + 14.79M qualifier edges + 52 LLM edges. Deduplicates by keeping the highest-weight edge per (subject, relation, object) triple. Overwrites `unified_kg_edges.parquet` with the final 15.1M-edge file. | `data/unified_kg/unified_kg_edges.parquet` (15,113,176 edges — final) |
| 8 | `rgcn_kg_colab.ipynb` | Trains the R-GCN link prediction model on the unified KG on Google Colab (A100). See `rgcn_pipeline.md` for full training details. | `MyDrive/neurovlm/embeddings/entity_embeddings_v2.pt` |

**Track 2 notebooks (coordinate GNN — independent of the KG pipeline):**

| notebook | what it does |
|---|---|
| `coord_gnn_colab.ipynb` | Trains the atlas-free coordinate GNN (Track 2) on Colab. Uses fMRI coordinate data, not the unified KG. |

**Also in `experiments/`:**

| file | description |
|---|---|
| `train_coord_gnn.py` | Local training script for the coordinate GNN (Track 2 equivalent of the Colab notebook) |
| `train_difumo_gat.py` | Training script for the DiFuMo soft atlas GAT (early Track 1 experiment, predates the KG approach) |
| `difumo_gat.ipynb` | Interactive walkthrough of the DiFuMo/GAT approach — early Track 1 experiment, kept for reference |
| `rgcn_pipeline.md` | Architecture, training config, and embedding details for the R-GCN (Phase 3) |
| `unified_kg_pipeline.md` | This file — documents the KG build pipeline (Phases 1–2) |
| `data.zip` | Old data snapshot (predates the qualifier-typed edges and v2 embeddings — **do not use as the authoritative data source**) |

> **Note on notebook output staleness:** `kg_unification.ipynb` outputs show
> 329,566 edges because it was last run before the qualifier step existed. The
> files on disk reflect the merged 15.1M-edge state produced by `kg_edge_merge.ipynb`.

---

## Overview

```
PubMed abstracts (1,231,613)
    │
    ├─► MeSH annotations ──► mesh_kg.ipynb ──────────────────────────────┐
    │                         mesh_kg_cooccurrence.ipynb                  │ MeSH KG
    │                         305,227 edges                               │
    │                                                                     ▼
    ├─► NLP extraction ────► nlp_graph.ipynb                       kg_unification.ipynb
    │                         12,841 nodes, 155,352 edges         (entity normalisation)
    │                                                                     │
    └─► CogAtlas scrape ───► (in kg_unification.ipynb)                   │
                              752 new cogat_ nodes, 0 edges              ▼
                                                              unified_kg_nodes.parquet
                                                              unified_kg_edges.parquet
                                                              (33,784 nodes, 329,566 edges)
                                                                         │
                                                                         ▼
    MeSH annotation qualifiers ──► mesh_qualifier_re.ipynb          kg_edge_merge.ipynb
    14,790,106 typed edges                                               │
                                                                         ▼
    LLM relation extraction ────► llm_relation_extraction.ipynb   unified_kg_edges.parquet
    52 typed edges                                                (15,113,176 edges — FINAL)
```

---

## Stage 1 — MeSH Knowledge Graph

**Notebooks:** `mesh_kg.ipynb`, `mesh_kg_cooccurrence.ipynb`  
**Code:** `src/neurovlm/gnn/mesh.py`

### Nodes

All 31,110 MeSH descriptors are retained. Node types are assigned from tree-number prefixes:

| node_type | MeSH tree prefix | count |
|---|---|---|
| molecular | D01–D27, D08 | 10,671 |
| disorder | C | 5,087 |
| organism | B | 3,938 |
| other | various | 3,196 |
| method | E | 3,023 |
| anatomical_region | A | 1,908 |
| biological_process | G | 1,886 |
| cognitive_construct | F | 1,056 |
| demographic | M | 345 |

### Edges

Three edge layers combined into `mesh_kg_edges_all.parquet`:

| relation_type | count | construction |
|---|---|---|
| `co_occurs_with` | 257,290 | Descriptor pairs co-annotating the same abstract; ≥1 endpoint in neuroimaging subtrees |
| `narrower_term_of` | 42,519 | Direct MeSH tree-number parent → child (ontological hierarchy) |
| `associated_with_disorder` | 2,107 | Curated disorder–anatomy/gene associations from MeSH supplementary records |
| `implicated_in` | 1,803 | Gene/molecule → biological process from MeSH qualifier pairings |
| `co_activates_with` | 931 | Region–region co-activation edges from neuroimaging metadata |
| `expressed_in` | 577 | Gene → anatomical region expression links |

---

## Stage 2 — NLP Co-occurrence Graph

**Notebook:** `nlp_graph.ipynb`

Raw graph from 1,231,613 PubMed abstracts: 133,625 typed nodes, 11,043,916 edges (min weight 50 — floor set at extraction time, cannot be lowered here).

Node filters applied:

| filter | rationale |
|---|---|
| `degree > 10` | removes rare terms too sparse to be reliable |
| `degree < 10,000` | removes hub non-terms (*study*, *model*, *data*) |
| `strength < 2,500,000` | removes hyper-frequent stop-like terms |
| community size ≥ 20 | Louvain micro-clusters are noise |
| `alias_count < limit` (type-conditional) | collapses near-duplicate surface forms; 50 for anatomical/cognitive types, 15–20 for method/metric/modality |
| `node_label != "other"` | `other` nodes have all category scores = 0 — no semantic type |

After filtering: **12,841 nodes, 155,352 edges**.

---

## Stage 3 — CogAtlas Knowledge Graph

**Code:** `src/neurovlm/gnn/normalize.py` (run inside `kg_unification.ipynb`)

918 cognitive concepts scraped from cognitiveatlas.org. 166 map to existing MeSH descriptors via string matching; 752 become new `cogat_` nodes. No relation edges are available (the CogAtlas API returns HTTP 500 for bulk relation requests; the `kind_of` scrape produced zero edges that survived endpoint remapping). CogAtlas contributes **nodes only** to the unified graph.

---

## Stage 4 — Entity Normalisation and Unification

**Notebook:** `kg_unification.ipynb`  
**Code:** `src/neurovlm/gnn/normalize.py`

MeSH DescriptorUI is the canonical backbone. Concepts resolved to MeSH inherit its UI; others get `cogat_trm_*` or `nlp_*` IDs.

**NLP matching cascade (type-conditional):**

```
Step 1  exact MeSH synonym match          → MeSH UI                 (all types)
Step 2  exact CogAtlas name match         → cogat_ or MeSH UI       (all types)
Step 3  word-boundary substring match     → MeSH UI                 (KEEP_TYPES only)
Step 4  keep as nlp_ node                 → nlp_{slug}              (KEEP_TYPES only)
Step 5  discard                           → dropped                 (DISCARD_TYPES if no exact match)
```

DISCARD_TYPES (`modality`, `method`, `metric`): exact match only — unmatched terms are too noisy.  
KEEP_TYPES (`disorder`, `anatomical_region`, `cognitive_construct`, `network`, `intervention`): proceed through all five steps — valid concepts often absent from MeSH (*default mode network*, *frontoparietal*, *endovascular treatment*).

Two LLM audit passes (`qwen2.5:7b-instruct` via Ollama) were run post-hoc to flag and remove bad substring matches and miscategorised `nlp_` nodes.

**NLP match summary:**

| match_type | count |
|---|---|
| discarded | 7,267 |
| new_nlp_node | 1,922 (after LLM audit from 3,296) |
| mesh_synonym_match | 1,662 |
| mesh_substring_match | 300 (after LLM audit from 543) |
| cogat_name_match | 73 |

**Master entity table: 33,784 canonical entities.**

Initial unified edge file: **329,566 edges** (MeSH 305,227 + NLP 24,339).

---

## Stage 5 — Qualifier-Based Edge Typing

**Notebook:** `mesh_qualifier_re.ipynb`  
**Code:** `src/neurovlm/gnn/mesh_re.py`

MeSH annotations include qualifier suffixes (e.g. `Brain/pathology`, `Surgery/methods`) that encode semantic relationships. This notebook maps qualifier suffixes to relation types:

| qualifier group | relation_type | example |
|---|---|---|
| pathology, physiology, metabolism, genetics, … | `implicated_in` | Brain/pathology |
| diagnostic imaging, diagnosis, complications, etiology, … | `associated_with_disorder` | Hippocampus/diagnostic imaging |
| surgery, therapeutic use, therapy, drug therapy | `treated_by` | Brain Neoplasms/surgery |
| methods, instrumentation | `used_in` | MRI/methods |

83.8% of qualified annotation rows are covered by the mapping. After pairing terms within the same abstract and filtering to unified KG node endpoints: **14,790,106 typed edges** saved to `nlp_kg/nlp_kg_edges_qualified.parquet`.

---

## Stage 6 — LLM Relation Extraction

**Notebook:** `llm_relation_extraction.ipynb`

Extracts a small set of typed edges from raw abstract text using an LLM for cases the qualifier approach misses. Produces 52 high-confidence edges saved to `nlp_kg/nlp_kg_edges_llm.parquet`.

---

## Stage 7 — Final Edge Merge

**Notebook:** `kg_edge_merge.ipynb`

Concatenates the three edge sources and deduplicates: for the same `(subject, relation, object)` triple appearing in multiple sources, the row with the highest weight is kept (ties broken by source priority: `mesh > mesh_qualifier > llm_re > nlp`). Self-loops are dropped.

| source | edges in | edges after dedup |
|---|---|---|
| original unified (MeSH + NLP) | 329,566 | — |
| qualifier-typed | 14,790,106 | — |
| LLM-RE | 52 | — |
| **merged total** | **15,119,724** | **15,113,176** |

**Final relation distribution:**

| relation_type | count | % | primary source |
|---|---|---|---|
| `implicated_in` | 6,619,101 | 43.8% | mesh_qualifier |
| `associated_with_disorder` | 4,185,364 | 27.7% | mesh_qualifier |
| `treated_by` | 2,695,381 | 17.8% | mesh_qualifier |
| `used_in` | 1,290,297 | 8.5% | mesh_qualifier |
| `co_occurs_with` | 278,985 | 1.8% | mesh / nlp |
| `narrower_term_of` | 42,519 | 0.3% | mesh |
| `co_activates_with` | 933 | 0.0% | mesh |
| `expressed_in` | 596 | 0.0% | mesh |

---

## Output Files

All outputs in `experiments/data/`:

| file | description |
|---|---|
| `unified_kg/unified_kg_nodes.parquet` | 33,784 canonical entities — `canonical_id`, `name`, `node_type`, `primary_source`, `sources` |
| `unified_kg/unified_kg_edges.parquet` | 15,113,176 typed edges — `subject_id`, `relation_type`, `object_id`, `source`, `weight`, `source_kg` |
| `unified_kg/unified_kg_edges_backup.parquet` | Pre-merge backup (329,566 edges, pre-qualifier state) |
| `unified_kg/cogat_kg_nodes.parquet` | 918 CogAtlas concepts with match metadata |
| `unified_kg/merge_log.json` | Audit log — one record per mapped/discarded NLP and CogAtlas entity |
| `mesh_kg/mesh_descriptors.parquet` | 31,110 MeSH descriptors with tree numbers and synonyms |
| `mesh_kg/mesh_kg_nodes.parquet` | MeSH node table with semantic node types |
| `mesh_kg/mesh_kg_edges_all.parquet` | Combined MeSH edges (305,227) |
| `nlp_kg/nlp_kg_nodes.parquet` | 12,841 filtered NLP nodes |
| `nlp_kg/nlp_kg_edges.parquet` | 155,352 NLP co-occurrence edges |
| `nlp_kg/nlp_kg_edges_qualified.parquet` | 14,790,106 qualifier-typed edges |
| `nlp_kg/nlp_kg_edges_llm.parquet` | 52 LLM-extracted typed edges |
