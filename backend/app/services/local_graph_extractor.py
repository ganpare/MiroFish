"""
LLM-based entity/relation extractor for the local Neo4j graph backend.
"""

from __future__ import annotations

import json
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

    def extract(self, text: str, ontology: Dict[str, Any]) -> Dict[str, Any]:
        entity_types = [item.get("name") for item in (ontology or {}).get("entity_types", []) if item.get("name")]
        edge_types = [item.get("name") for item in (ontology or {}).get("edge_types", []) if item.get("name")]

        system = (
            "你是一个严格输出 JSON 的信息抽取器。"
            "你的任务是从给定文本中抽取实体与实体间关系，输出必须是 JSON 对象。"
            "不要输出任何解释或多余文本。"
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
                    {"role": "user", "content": json.dumps(user, ensure_ascii=False, indent=2)},
                ],
                temperature=0.2,
                max_tokens=2048,
            )
        except Exception as exc:
            if self._is_data_inspection_failed(exc):
                logger.warning("LLM extract blocked by provider moderation; retrying in safe mode.")
                try:
                    result = self._extract_safe(text=text, ontology=ontology)
                except Exception as safe_exc:
                    logger.error(f"Safe-mode extract still failed: {safe_exc}")
                    return {"entities": [], "relations": []}
            else:
                logger.error(f"LLM extract failed: {exc}")
                raise

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

        return {"entities": cleaned_entities, "relations": cleaned_relations}
