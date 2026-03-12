"""
LLM-based entity/relation extractor for the local Neo4j graph backend.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional

from ..config import Config
from ..utils.llm_client import LLMClient
from ..utils.logger import get_logger

logger = get_logger("mirofish.local_graph_extractor")


class LocalGraphExtractor:
    def __init__(self, llm: Optional[LLMClient] = None):
        self.llm = llm or LLMClient(
            api_key=Config.EXTRACT_API_KEY,
            base_url=Config.EXTRACT_BASE_URL,
            model=Config.EXTRACT_MODEL_NAME,
        )

    @staticmethod
    def _is_data_inspection_failed(err: Exception) -> bool:
        try:
            body = getattr(err, "body", None)
            if isinstance(body, dict):
                code = ((body.get("error") or {}).get("code") or "").lower()
                message = ((body.get("error") or {}).get("message") or "").lower()
                return "data_inspection_failed" in code or "inappropriate" in message
        except Exception:
            pass
        text = (str(err) or "").lower()
        return "data_inspection_failed" in text or "inappropriate content" in text

    def _extract_safe(self, text: str, ontology: Dict[str, Any]) -> Dict[str, Any]:
        entity_types = [item.get("name") for item in (ontology or {}).get("entity_types", []) if item.get("name")]
        edge_types = [item.get("name") for item in (ontology or {}).get("edge_types", []) if item.get("name")]

        system = (
            "You are a strict JSON-only information extractor.\n"
            "Return ONLY a valid JSON object.\n"
            "Safety: do not output explicit/sexual/violent/hateful/self-harm content.\n"
            "If the input might trigger moderation, redact details using '[REDACTED]' and keep outputs minimal.\n"
        )
        user = {
            "text": text,
            "allowed_entity_types": entity_types,
            "allowed_relation_types": edge_types,
            "requirements": {
                "only_use_allowed_types": True,
                "deduplicate_entities_by_name_and_type": True,
                "do_not_guess": True,
                "return_empty_when_none": True,
                "avoid_quoting_input": True,
            },
            "output_schema": {
                "entities": [{"name": "string", "type": "string", "summary": "", "attributes": {}}],
                "relations": [
                    {
                        "source": "string",
                        "source_type": "string",
                        "target": "string",
                        "target_type": "string",
                        "relation": "string",
                        "fact": "",
                        "attributes": {},
                    }
                ],
            },
        }
        return self.llm.chat_json(
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": json.dumps(user, ensure_ascii=False)},
            ],
            temperature=0.0,
            max_tokens=1536,
        )

    @staticmethod
    def _split_sentences(text: str) -> List[str]:
        sentences = re.split(r"(?<=[。！？!?\.])\s+|\n+", text)
        return [sentence.strip() for sentence in sentences if sentence and sentence.strip()]

    def extract_heuristic(self, text: str, ontology: Dict[str, Any], reason: str = "heuristic") -> Dict[str, Any]:
        entity_defs = (ontology or {}).get("entity_types", []) or []
        relation_defs = (ontology or {}).get("edge_types", []) or []
        sentences = self._split_sentences(text)

        entities: List[Dict[str, Any]] = []
        entity_keys = set()
        entities_by_type: Dict[str, List[Dict[str, Any]]] = {}

        for entity_def in entity_defs:
            entity_type = str(entity_def.get("name", "")).strip()
            if not entity_type:
                continue

            candidates = []
            for candidate in entity_def.get("examples") or []:
                value = str(candidate).strip()
                if value:
                    candidates.append(value)
            type_name = entity_type.strip()
            if type_name and type_name not in candidates:
                candidates.append(type_name)

            for candidate in candidates:
                if candidate not in text:
                    continue

                summary = next((sentence for sentence in sentences if candidate in sentence), entity_def.get("description", ""))
                key = (entity_type.lower(), candidate.lower())
                if key in entity_keys:
                    continue

                entity = {
                    "name": candidate,
                    "type": entity_type,
                    "summary": summary[:240],
                    "attributes": {
                        "extraction_mode": "heuristic",
                    },
                }
                entities.append(entity)
                entities_by_type.setdefault(entity_type, []).append(entity)
                entity_keys.add(key)

        relations: List[Dict[str, Any]] = []
        relation_keys = set()
        for relation_def in relation_defs:
            relation_name = str(relation_def.get("name", "")).strip()
            if not relation_name:
                continue
            for source_target in relation_def.get("source_targets", []) or []:
                source_type = str(source_target.get("source", "")).strip()
                target_type = str(source_target.get("target", "")).strip()
                source_entities = entities_by_type.get(source_type) or []
                target_entities = entities_by_type.get(target_type) or []
                if not source_entities or not target_entities:
                    continue

                matched_sentence = None
                matched_pair = None
                for sentence in sentences:
                    for source_entity in source_entities:
                        if source_entity["name"] not in sentence:
                            continue
                        for target_entity in target_entities:
                            if source_entity["name"] == target_entity["name"]:
                                continue
                            if target_entity["name"] in sentence:
                                matched_sentence = sentence
                                matched_pair = (source_entity, target_entity)
                                break
                        if matched_pair:
                            break
                    if matched_pair:
                        break

                if not matched_pair:
                    continue

                source_entity, target_entity = matched_pair
                relation_key = (
                    source_entity["name"].lower(),
                    target_entity["name"].lower(),
                    relation_name.lower(),
                )
                if relation_key in relation_keys:
                    continue

                relations.append(
                    {
                        "source": source_entity["name"],
                        "source_type": source_type,
                        "target": target_entity["name"],
                        "target_type": target_type,
                        "relation": relation_name,
                        "fact": (matched_sentence or "")[:280],
                        "attributes": {
                            "extraction_mode": "heuristic",
                        },
                    }
                )
                relation_keys.add(relation_key)

        return {
            "entities": entities,
            "relations": relations,
            "_strategy": "heuristic",
            "_reason": reason,
        }

    def extract(self, text: str, ontology: Dict[str, Any]) -> Dict[str, Any]:
        entity_types = [item.get("name") for item in (ontology or {}).get("entity_types", []) if item.get("name")]
        edge_types = [item.get("name") for item in (ontology or {}).get("edge_types", []) if item.get("name")]

        system = (
            "You are a strict JSON-only information extractor. "
            "Extract entities and relations from the provided text. "
            "Return exactly one JSON object and nothing else."
        )
        user = {
            "text": text,
            "allowed_entity_types": entity_types,
            "allowed_relation_types": edge_types,
            "requirements": {
                "only_use_allowed_types": True,
                "deduplicate_entities_by_name_and_type": True,
                "do_not_guess": True,
                "return_empty_when_none": True,
            },
            "output_schema": {
                "entities": [{"name": "string", "type": "string", "summary": "string", "attributes": {"key": "value"}}],
                "relations": [
                    {
                        "source": "string",
                        "source_type": "string",
                        "target": "string",
                        "target_type": "string",
                        "relation": "string",
                        "fact": "string",
                        "attributes": {"key": "value"},
                    }
                ],
            },
        }

        try:
            result = self.llm.chat_json(
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": json.dumps(user, ensure_ascii=False)},
                ],
                temperature=0.1,
                max_tokens=1536,
            )
        except Exception as exc:
            if self._is_data_inspection_failed(exc):
                logger.warning("LLM extract blocked by provider moderation; retrying in safe mode.")
                try:
                    result = self._extract_safe(text=text, ontology=ontology)
                except Exception as safe_exc:
                    logger.error(f"Safe-mode extract still failed: {safe_exc}")
                    return self.extract_heuristic(text=text, ontology=ontology, reason="safe_mode_failed")
            else:
                logger.error(f"LLM extract failed: {exc}")
                return self.extract_heuristic(text=text, ontology=ontology, reason="llm_failed")

        cleaned_entities: List[Dict[str, Any]] = []
        for entity in result.get("entities") or []:
            name = (entity or {}).get("name")
            entity_type = (entity or {}).get("type")
            if not name or not entity_type:
                continue
            if entity_types and entity_type not in entity_types:
                continue
            cleaned_entities.append(
                {
                    "name": str(name).strip(),
                    "type": str(entity_type).strip(),
                    "summary": str((entity or {}).get("summary") or "").strip(),
                    "attributes": (entity or {}).get("attributes") or {},
                }
            )

        cleaned_relations: List[Dict[str, Any]] = []
        for relation in result.get("relations") or []:
            item = relation or {}
            source = item.get("source")
            target = item.get("target")
            source_type = item.get("source_type")
            target_type = item.get("target_type")
            relation_name = item.get("relation")
            if not source or not target or not source_type or not target_type or not relation_name:
                continue
            if edge_types and relation_name not in edge_types:
                continue
            cleaned_relations.append(
                {
                    "source": str(source).strip(),
                    "source_type": str(source_type).strip(),
                    "target": str(target).strip(),
                    "target_type": str(target_type).strip(),
                    "relation": str(relation_name).strip(),
                    "fact": str(item.get("fact") or "").strip(),
                    "attributes": item.get("attributes") or {},
                }
            )

        if not cleaned_entities and not cleaned_relations:
            logger.warning("LLM extract returned no usable JSON payload; falling back to heuristic extraction.")
            return self.extract_heuristic(text=text, ontology=ontology, reason="empty_llm_payload")

        return {
            "entities": cleaned_entities,
            "relations": cleaned_relations,
            "_strategy": "llm",
        }
