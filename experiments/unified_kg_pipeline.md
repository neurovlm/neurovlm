# Unified Knowledge Graph Pipeline

Documents how the three source knowledge graphs — MeSH, NLP co-occurrence, and CogAtlas — are built and merged into a single canonical entity graph for the NeuroVLM R-GCN.

---

## Overview

```
PubMed abstracts (1.2 M)
    │
    ├─► MeSH annotations ──────► MeSH KG          (31,110 nodes, 305,227 edges)
    │                                 │
    ├─► NLP extraction ────────► NLP KG            (12,841 nodes, 155,352 edges)
    │                                 │
    └─► CogAtlas scrape ───────► CogAtlas KG       (918 nodes)
                                      │
                                      ▼
                               Entity normalisation
                                      │
                                      ▼
                               Unified KG            (33,784 nodes, 329,566 edges)
```

---

## Stage 1 — MeSH Knowledge Graph

**Source:** PubMed MeSH annotations for 1,231,613 neuroimaging abstracts.  
**Code:** `src/neurovlm/gnn/mesh.py`

### Nodes

All 31,110 MeSH descriptors are retained. Each descriptor has a `DescriptorUI` (e.g. `D006624`), a preferred name, a set of synonyms (entry terms), and one or more tree numbers that encode its position in the MeSH hierarchy.

Node types are assigned by mapping tree-number prefixes to semantic categories:

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

Three edge layers are combined into `mesh_kg_edges_all.parquet`:

| relation_type | count | construction method |
|---|---|---|
| `co_occurs_with` | 257,290 | Abstract-level co-occurrence: an edge `(A, B)` is added when descriptors A and B both annotate the same abstract, with weight = number of shared abstracts. Only edges with at least one endpoint in the neuroimaging-relevant subtrees are kept. |
| `narrower_term_of` | 42,519 | Direct MeSH tree-number parent → child relationships. Encodes the ontological hierarchy. |
| `associated_with_disorder` | 2,107 | Curated disorder–anatomy / disorder–gene associations derived from MeSH supplementary records. |
| `implicated_in` | 1,803 | Gene/molecule → biological process links from MeSH qualifier pairings. |
| `co_activates_with` | 931 | Region–region co-activation edges sourced from neuroimaging co-activation metadata. |
| `expressed_in` | 577 | Gene → anatomical region expression links. |

**Total: 305,227 edges.**

The MeSH KG is the canonical backbone. Every concept that can be mapped to a MeSH DescriptorUI inherits that UI as its canonical identifier throughout the pipeline.

---

## Stage 2 — NLP Co-occurrence Graph

**Source:** Entity extraction from the same 1,231,613 PubMed abstracts using an NLP pipeline (spaCy + custom NER + community detection).  
**Code:** `experiments/nlp_graph.ipynb`

### Construction

**Extraction:** Named entities and noun phrases were extracted from abstracts, normalised (lowercased, punctuation-stripped, whitespace-collapsed), and counted for pairwise co-occurrence within each abstract.

**Raw graph:**
- 214,495 unique terms (nodes)
- 11,043,916 co-occurrence edges (minimum weight 50 — floor set upstream at extraction time)

**Node typing:** A multi-class NLP classifier assigns each term one of eight semantic categories: `anatomical_region`, `cognitive_construct`, `disorder`, `intervention`, `method`, `metric`, `modality`, `network`, or `other`. The classifier outputs a confidence score per class; the argmax is taken as the label.

`other` nodes have `label_confidence = 1.0` with all eight category scores at `0.0` — the classifier is maximally certain these terms have no neuroscience semantic type (generic vocabulary: *data*, *findings*, *patient*, *age*, …). They are dropped entirely.

**Node filters** applied to the remaining 133,625 typed nodes:

| filter | rationale |
|---|---|
| `degree > 10` | removes rare terms too sparse to be reliable |
| `degree < 10,000` | removes hub non-terms (generic words like *study*, *model*) |
| `strength < 2,500,000` | removes hyper-frequent stop-like terms |
| community size ≥ 20 | Louvain communities with fewer than 20 members are considered noise |
| `alias_count < limit` (type-conditional) | collapses near-duplicate surface forms; limit is 50 for anatomical/cognitive types, 15–20 for method/metric/modality |
| `node_label != "other"` | excludes untyped terms |

**After filtering:**

| node_type | count | retention |
|---|---|---|
| modality | 4,759 | 28.7% |
| method | 1,833 | 24.7% |
| anatomical_region | 1,783 | 27.1% |
| metric | 1,470 | 30.4% |
| disorder | 1,397 | 31.9% |
| cognitive_construct | 708 | 30.3% |
| intervention | 677 | 27.0% |
| network | 214 | 29.4% |
| **total** | **12,841** | |

**Edge filter:** Edges where either endpoint was removed by node filtering are dropped.  
155,352 edges remain (from 11,043,916), all typed `co_occurs_with`, weight range [50, 17,708].

---

## Stage 3 — CogAtlas Knowledge Graph

**Source:** [cognitiveatlas.org](https://www.cognitiveatlas.org) — a community-curated cognitive ontology.  
**Code:** `src/neurovlm/cogatlas.py`, `src/neurovlm/gnn/normalize.py`

### Construction

The CogAtlas JSON API returns HTTP 500 for bulk requests. Concepts were scraped from the HTML listing pages (`/concepts/a-z`) using BeautifulSoup, yielding 918 unique concepts with their `trm_id` and preferred name.

All CogAtlas concepts are typed `cognitive_construct`. No relation edges are available from the scrape (the API would be required for those).

---

## Stage 4 — Entity Normalisation and Unification

**Code:** `src/neurovlm/gnn/normalize.py`, `experiments/kg_unification.ipynb`

### Canonical ID assignment

MeSH `DescriptorUI` is the canonical backbone. Every concept that can be resolved to a MeSH descriptor inherits its UI.

**MeSH synonym lookup** is built from all 31,110 descriptors expanded with their full synonym lists (entry terms):

```
245,494 normalised-string → DescriptorUI mappings
```

String normalisation (`_norm`): lowercase → strip punctuation → collapse whitespace.

### CogAtlas mapping

Each CogAtlas concept is string-matched against the MeSH synonym lookup:

| outcome | count |
|---|---|
| Mapped to MeSH (`mesh_synonym_match`) | 166 |
| New node (`cogat_{trm_id}`) | 752 |

Unmapped concepts retain their CogAtlas identifier as `cogat_trm_XXXXX`.

### NLP entity normalisation

NLP terms go through a type-conditional five-step matching cascade:

```
Step 1  exact MeSH synonym match          → canonical MeSH UI
Step 2  exact CogAtlas name match         → cogat_ or MeSH UI (via cogat mapping)
Step 3  word-boundary substring match     → canonical MeSH UI      (KEEP_TYPES only)
Step 4  keep as nlp_ node                 → nlp_{slug}             (KEEP_TYPES only)
Step 5  discard                           → dropped                (DISCARD_TYPES if no exact match)
```

**DISCARD_TYPES** (`modality`, `method`, `metric`): Exact match only. These types contain high noise from the NLP classifier; unmatched terms are dropped rather than kept as new nodes, since a generic term like *analysis* or *scan* should not become a graph node.

**KEEP_TYPES** (`disorder`, `anatomical_region`, `cognitive_construct`, `network`, `intervention`): Proceed through all five steps. Unmatched terms are retained as `nlp_{slug}` nodes because these types frequently contain valid concepts absent from MeSH (e.g. *default mode network*, *frontoparietal*, *endovascular treatment*).

**Substring matching** (Step 3): Word-boundary aware — implemented as `" term " in " syn "` using space-padding to prevent false matches (e.g. `brain` does not match `brainstem`). Minimum term length 6 characters. Primary descriptor names are preferred over synonyms in case of ambiguity (prevents e.g. `alzheimer` resolving to *Amyloid beta-Peptides* instead of *Alzheimer Disease*).

**Secondary type filter** (`NLP_NODE_FILTERS`): After Step 3 fails, a per-type regex discards known-bad residuals before they become `nlp_` nodes. Example for `anatomical_region`:
```
\b(?:activity|activation)\b | \bmri\b | ^(?:hemisphere|lobes|whole-brain|whole brain)$
```
This discards functional states and modality contamination that the NLP classifier mislabelled as anatomical regions.

**LLM audits (post-hoc):** Two Ollama (`qwen2.5:7b-instruct`) verification passes were run after the programmatic pipeline:

1. **nlp_ node audit** — checked all 3,296 kept `nlp_` nodes; flagged 1,603 as NO/UNCERTAIN. Flagged nodes are removed from the canonical mapping (treated as discarded).
2. **Substring match audit** — checked all 543 substring matches for correctness of the MeSH mapping (ignoring node_type). Flagged KEEP_TYPE nodes are remapped to `nlp_{slug}` instead of the wrong MeSH ID; DISCARD_TYPE nodes are dropped.

**FORCE_MAP:** A manually maintained override dict corrects known-bad matches that are too ambiguous for programmatic resolution (e.g. `occipital → Occipital Bone` should be `occipital → Occipital Lobe`).

**NLP match summary:**

| match_type | count |
|---|---|
| discarded | 7,267 |
| new_nlp_node | 3,296 |
| mesh_synonym_match | 1,662 |
| mesh_substring_match | 543 |
| cogat_name_match | 73 |

### Master entity table

The three node sets are merged into a single canonical entity table. When multiple sources map to the same canonical ID, the `sources` column records all contributing KGs.

| primary_source | count |
|---|---|
| mesh | 31,110 |
| nlp | 1,922 |
| cogatlas | 752 |
| **total** | **33,784** |

### Edge remapping

NLP edges are remapped: both endpoints are replaced with their canonical IDs. Edges where either endpoint has no canonical mapping (discarded terms) are dropped.

| source | edges retained |
|---|---|
| MeSH (identity mapping) | 305,227 |
| NLP (after remapping) | 24,339 (of 155,352) |
| **Unified total** | **329,566** |

**Relation types in unified graph:**

| relation_type | count |
|---|---|
| `co_occurs_with` | 281,629 |
| `narrower_term_of` | 42,519 |
| `associated_with_disorder` | 2,107 |
| `implicated_in` | 1,803 |
| `co_activates_with` | 931 |
| `expressed_in` | 577 |

---

## Output Files

All outputs written to `experiments/data/unified_kg/`:

| file | description |
|---|---|
| `unified_kg_nodes.parquet` | 33,784 canonical entities with `canonical_id`, `name`, `node_type`, `primary_source`, `sources` |
| `unified_kg_edges.parquet` | 329,566 edges with `subject_id`, `relation_type`, `object_id`, `source`, `weight`, `source_kg` |
| `cogat_kg_nodes.parquet` | 918 CogAtlas concepts with match metadata |
| `merge_log.json` | Full audit log — one record per mapped/discarded NLP and CogAtlas entity |
