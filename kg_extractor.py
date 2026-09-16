"""Reusable, dependency-light stateful KG extraction primitives."""

from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import asdict, dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Dict, List, Set


@dataclass
class Triple:
    subject: str
    predicate: str
    object: str
    source_chunk: int
    confidence: float = 0.8


class StatefulKGExtractor:
    """Base class for chunking, entity clustering, and triple consolidation."""

    def __init__(self):
        self.raw_triples: List[Triple] = []
        self.consolidated_triples: List[Triple] = []
        self.entity_clusters: Dict[str, Set[str]] = {}
        self.processing_state = {
            "chunks_processed": 0,
            "total_raw_triples": 0,
            "total_consolidated_triples": 0,
        }

    def chunk_document(
        self, text: str, chunk_size: int = 180, overlap: int = 30
    ) -> List[Dict]:
        words = text.split()
        if not words:
            return []
        if chunk_size <= 0 or overlap < 0 or overlap >= chunk_size:
            raise ValueError("chunk_size must be positive and overlap must be smaller")

        chunks = []
        step = chunk_size - overlap
        for chunk_id, start in enumerate(range(0, len(words), step)):
            selected = words[start : start + chunk_size]
            if not selected:
                break
            chunks.append({
                "id": chunk_id,
                "text": " ".join(selected),
                "start_word": start,
                "end_word": start + len(selected),
            })
            if start + chunk_size >= len(words):
                break
        return chunks

    def extract_from_chunk(self, chunk: Dict, use_llm: bool = False) -> List[Triple]:
        """Small deterministic fallback for offline demos."""
        text = chunk.get("text", "")
        triples: List[Triple] = []
        patterns = [
            (r"([A-Z][\w-]*(?:\s+[A-Z][\w-]*){0,5})\s+(supports|opposes)\s+([A-Z][\w-]*(?:\s+[A-Z][\w-]*){0,5})", None),
            (r"([A-Z][\w-]*(?:\s+[A-Z][\w-]*){0,5})\s+is\s+(?:a|an|the)\s+([a-z][\w-]*(?:\s+[a-z][\w-]*){0,4})", "IS_A"),
        ]
        for pattern, fixed_predicate in patterns:
            for match in re.finditer(pattern, text):
                if fixed_predicate:
                    subject, obj = match.groups()
                    predicate = fixed_predicate
                else:
                    subject, predicate, obj = match.groups()
                    predicate = predicate.upper()
                triples.append(Triple(
                    subject=subject.strip(),
                    predicate=predicate.strip().replace(" ", "_"),
                    object=obj.strip(),
                    source_chunk=int(chunk["id"]),
                    confidence=0.55,
                ))

        self.raw_triples.extend(triples)
        self.processing_state["chunks_processed"] += 1
        self.processing_state["total_raw_triples"] = len(self.raw_triples)
        return triples

    @staticmethod
    def _normalise(value: str) -> str:
        value = re.sub(r"[^a-z0-9 ]+", " ", value.lower())
        return re.sub(r"\s+", " ", value).strip()

    def _entity_similarity(self, left: str, right: str) -> float:
        left_norm = self._normalise(left)
        right_norm = self._normalise(right)
        if left_norm == right_norm:
            return 1.0
        return SequenceMatcher(None, left_norm, right_norm).ratio()

    def cluster_entities(self, similarity_threshold: float = 0.82):
        entities = sorted({
            value.strip()
            for triple in self.raw_triples
            for value in (triple.subject, triple.object)
            if value and value.strip()
        })
        parent = {entity: entity for entity in entities}

        def find(entity: str) -> str:
            while parent[entity] != entity:
                parent[entity] = parent[parent[entity]]
                entity = parent[entity]
            return entity

        def union(left: str, right: str):
            root_left, root_right = find(left), find(right)
            if root_left != root_right:
                parent[root_right] = root_left

        for index, left in enumerate(entities):
            for right in entities[index + 1 :]:
                if self._entity_similarity(left, right) >= similarity_threshold:
                    union(left, right)

        groups: Dict[str, Set[str]] = {}
        for entity in entities:
            groups.setdefault(find(entity), set()).add(entity)

        counts = Counter(
            value
            for triple in self.raw_triples
            for value in (triple.subject, triple.object)
        )
        clusters: Dict[str, Set[str]] = {}
        for members in groups.values():
            canonical = max(members, key=lambda value: (counts[value], -len(value)))
            clusters[canonical] = set(members)
        self.entity_clusters = clusters
        return clusters

    def consolidate_triples(self) -> List[Triple]:
        if not self.entity_clusters:
            self.cluster_entities()
        mapping = {
            variant: canonical
            for canonical, variants in self.entity_clusters.items()
            for variant in variants
        }
        consolidated: Dict[tuple, Triple] = {}
        for triple in self.raw_triples:
            subject = mapping.get(triple.subject, triple.subject)
            obj = mapping.get(triple.object, triple.object)
            key = (self._normalise(subject), triple.predicate.upper(), self._normalise(obj))
            candidate = Triple(subject, triple.predicate.upper(), obj, triple.source_chunk, triple.confidence)
            previous = consolidated.get(key)
            if previous is None or candidate.confidence > previous.confidence:
                consolidated[key] = candidate
        self.consolidated_triples = list(consolidated.values())
        self.processing_state["total_consolidated_triples"] = len(self.consolidated_triples)
        return self.consolidated_triples

    def get_state_snapshot(self) -> Dict:
        raw_count = len(self.raw_triples)
        consolidated_count = len(self.consolidated_triples)
        return {
            "raw_triples": raw_count,
            "consolidated_triples": consolidated_count,
            "deduplication_ratio": (
                1 - consolidated_count / raw_count if raw_count else 0.0
            ),
            "chunks_processed": self.processing_state["chunks_processed"],
            "entity_clusters": len(self.entity_clusters),
        }

    def export_to_json(self, filepath: str):
        output = Path(filepath)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", encoding="utf-8") as handle:
            json.dump({
                "triples": [asdict(triple) for triple in self.consolidated_triples],
                "state": self.get_state_snapshot(),
                "entity_clusters": {
                    key: sorted(value) for key, value in self.entity_clusters.items()
                },
            }, handle, indent=2, ensure_ascii=False)

