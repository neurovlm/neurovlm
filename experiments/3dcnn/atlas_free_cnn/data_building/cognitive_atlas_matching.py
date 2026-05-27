"""Optional conservative Cognitive Atlas exact matching, disabled by default."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from atlas_free_cnn.data_building.definitions import POSITIVE_WEIGHTS, normalize_key, text_pair


@dataclass(frozen=True)
class CogAtlasConcept:
    concept_id: str
    name: str
    definition: str
    aliases: tuple[str, ...] = ()


class ConservativeCogAtlasMatcher:
    """Exact/synonym matcher only; no embeddings or fuzzy matching."""

    def __init__(self, concepts: Iterable[CogAtlasConcept]):
        self.lookup: dict[str, tuple[CogAtlasConcept, str]] = {}
        for concept in concepts:
            self.lookup[normalize_key(concept.name)] = (concept, "exact_name")
            for alias in concept.aliases:
                self.lookup[normalize_key(alias)] = (concept, "exact_alias")

    def match(self, term: str) -> dict | None:
        hit = self.lookup.get(normalize_key(term))
        if hit is None:
            return None
        concept, match_type = hit
        return {
            "text": text_pair(concept.name, concept.definition),
            "term": concept.name,
            "category": "cognitive_construct",
            "source": "cognitive_atlas_exact_match",
            "weight": POSITIVE_WEIGHTS["cognitive_atlas_exact_match"],
            "reliability": "strong",
            "cognitive_atlas_id": concept.concept_id,
            "matched_from": term,
            "match_type": match_type,
        }


def concepts_from_dataframe(df) -> list[CogAtlasConcept]:
    concepts: list[CogAtlasConcept] = []
    name_col = next((c for c in ("name", "term", "concept") if c in df.columns), None)
    def_col = next((c for c in ("definition", "description") if c in df.columns), None)
    id_col = next((c for c in ("id", "concept_id", "cogatlas_id") if c in df.columns), None)
    if not name_col or not def_col:
        return concepts
    for _, row in df.iterrows():
        name = str(row[name_col]).strip()
        definition = str(row[def_col]).strip()
        concept_id = str(row[id_col]).strip() if id_col else name
        if name and definition and definition.lower() != "nan":
            concepts.append(CogAtlasConcept(concept_id, name, definition))
    return concepts

