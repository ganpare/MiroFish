"""
Genre-aware ontology schema registry.
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


SCHEMA_DIR = Path(__file__).resolve().parent.parent / "schemas"


def _unique_list(values: List[str]) -> List[str]:
    seen = set()
    result: List[str] = []
    for value in values:
        if not value or value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


class GenreSchemaRegistry:
    """Loads and merges genre schema definitions from YAML files."""

    def __init__(self, schema_dir: Optional[Path] = None):
        self.schema_dir = Path(schema_dir or SCHEMA_DIR)
        self._cache: Optional[Dict[str, Dict[str, Any]]] = None

    def _ensure_loaded(self) -> None:
        if self._cache is not None:
            return

        cache: Dict[str, Dict[str, Any]] = {}
        for path in sorted(self.schema_dir.glob("*.yaml")):
            with open(path, "r", encoding="utf-8") as f:
                raw = yaml.safe_load(f) or {}
            schema = self._normalize_schema(raw)
            cache[schema["genre"]] = schema

        self._cache = cache

    def _normalize_schema(self, raw: Dict[str, Any]) -> Dict[str, Any]:
        genre = str(raw.get("genre", "")).strip()
        if not genre:
            raise ValueError("Schema file missing required 'genre' field")

        report_template = raw.get("report_template") or {}
        sections = report_template.get("sections", []) if isinstance(report_template, dict) else []

        return {
            "genre": genre,
            "display_name": raw.get("display_name", genre),
            "description": raw.get("description", ""),
            "keywords": _unique_list(list(raw.get("keywords", []) or [])),
            "required_entity_types": _unique_list(list(raw.get("required_entity_types", []) or [])),
            "entity_types": _unique_list(list(raw.get("entity_types", []) or [])),
            "relation_types": _unique_list(list(raw.get("relation_types", []) or [])),
            "agentizable_types": _unique_list(list(raw.get("agentizable_types", []) or [])),
            "non_agentizable_types": _unique_list(list(raw.get("non_agentizable_types", []) or [])),
            "simulation_grammar": dict(raw.get("simulation_grammar", {}) or {}),
            "report_template": {
                "sections": _unique_list(list(sections or []))
            },
        }

    def list_schemas(self) -> List[Dict[str, Any]]:
        self._ensure_loaded()
        assert self._cache is not None
        return [deepcopy(schema) for schema in self._cache.values()]

    def get_schema(self, genre: str) -> Dict[str, Any]:
        self._ensure_loaded()
        assert self._cache is not None
        if genre not in self._cache:
            raise KeyError(f"Unknown schema genre: {genre}")
        return deepcopy(self._cache[genre])

    def get_schema_names(self) -> List[str]:
        self._ensure_loaded()
        assert self._cache is not None
        return list(self._cache.keys())

    def merge_schema(self, base_genre: str, overlay_genres: Optional[List[str]] = None) -> Dict[str, Any]:
        merged = self.get_schema(base_genre)
        overlays = overlay_genres or []

        for overlay_genre in overlays:
            overlay = self.get_schema(overlay_genre)
            merged["entity_types"] = _unique_list(merged["entity_types"] + overlay["entity_types"])
            merged["relation_types"] = _unique_list(merged["relation_types"] + overlay["relation_types"])
            merged["agentizable_types"] = _unique_list(merged["agentizable_types"] + overlay["agentizable_types"])
            merged["non_agentizable_types"] = _unique_list(merged["non_agentizable_types"] + overlay["non_agentizable_types"])
            merged["keywords"] = _unique_list(merged["keywords"] + overlay["keywords"])
            merged["required_entity_types"] = _unique_list(
                merged["required_entity_types"] + overlay["required_entity_types"]
            )
            merged["report_template"]["sections"] = _unique_list(
                merged["report_template"]["sections"] + overlay["report_template"]["sections"]
            )
            merged["simulation_grammar"].update(overlay["simulation_grammar"])

        merged["schema_overlays"] = list(overlays)
        return merged

    def heuristic_candidates(self, text: str, limit: int = 3) -> List[Dict[str, Any]]:
        self._ensure_loaded()
        assert self._cache is not None

        normalized = text.lower()
        scored = []
        for schema in self._cache.values():
            score = 0
            for keyword in schema.get("keywords", []):
                if keyword.lower() in normalized:
                    score += 1
            scored.append({
                "genre": schema["genre"],
                "score": score,
                "reason": f"keyword_hits={score}"
            })

        scored.sort(key=lambda item: item["score"], reverse=True)
        return scored[:limit]
