"""
Local graph builder.

Builds a knowledge graph into Neo4j using cloud LLM extraction, and optionally
stores chunk embeddings into Qdrant.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Callable, Dict, List, Optional, Tuple

from ..config import Config
from ..utils.llm_client import LLMClient
from ..utils.logger import get_logger
from .entity_type_normalizer import canonicalize_entity_type
from .local_graph_extractor import LocalGraphExtractor
from .local_graph_store import LocalEntity, LocalNeo4jGraphStore, LocalRelation
from .local_vector_store import QdrantChunkStore
from .text_processor import TextProcessor

logger = get_logger("mirofish.local_graph_builder")


def _now_iso() -> str:
    return datetime.now().isoformat()


class LocalGraphBuilderService:
    def __init__(self):
        self.store = LocalNeo4jGraphStore()
        self.extractor = LocalGraphExtractor()
        self.vector_store = None

    def _seed_graph_from_ontology(self, project_id: str, graph_id: str, ontology: Dict) -> None:
        """Create a minimal graph from ontology examples when extraction yields nothing."""
        seeded_entities: List[LocalEntity] = []
        for entity_def in ontology.get("entity_types", []):
            entity_type = canonicalize_entity_type(entity_def.get("name", "Entity"))
            examples = entity_def.get("examples") or []
            name = ""
            for example in examples:
                text = str(example).strip()
                if text:
                    name = text
                    break
            if not name:
                name = entity_type
            seeded_entities.append(
                LocalEntity(
                    project_id=project_id,
                    graph_id=graph_id,
                    name=name,
                    entity_type=entity_type,
                    summary=entity_def.get("description", ""),
                    attributes={"seeded_from": "ontology_examples"},
                    source_entity_types=[entity_def.get("name", entity_type)],
                    created_at=_now_iso(),
                )
            )

        if not seeded_entities:
            return

        entity_uuids = self.store.upsert_entities(seeded_entities)
        relations: List[LocalRelation] = []
        relation_defs = ontology.get("edge_types") or []
        entity_by_type = {entity.entity_type: entity for entity in seeded_entities}
        for relation_def in relation_defs:
            relation_name = relation_def.get("name", "")
            for source_target in relation_def.get("source_targets", [])[:1]:
                source_entity = entity_by_type.get(canonicalize_entity_type(source_target.get("source")))
                target_entity = entity_by_type.get(canonicalize_entity_type(source_target.get("target")))
                if not source_entity or not target_entity:
                    continue
                relations.append(
                    LocalRelation(
                        project_id=project_id,
                        graph_id=graph_id,
                        source_uuid=source_entity.uuid,
                        target_uuid=target_entity.uuid,
                        relation_name=relation_name,
                        fact=f"{source_entity.name} {relation_name} {target_entity.name}",
                        attributes={"seeded_from": "ontology_examples"},
                        created_at=_now_iso(),
                    )
                )
                break

        self.store.upsert_relations(relations)
        logger.warning(
            f"抽出結果が空だったため、ontology の examples から最小グラフを生成しました "
            f"(graph_id={graph_id}, entities={len(entity_uuids)}, relations={len(relations)})"
        )

    def _get_vector_store(self) -> Optional[QdrantChunkStore]:
        if Config.VECTOR_BACKEND != "qdrant":
            return None
        if self.vector_store is not None:
            return self.vector_store
        try:
            self.vector_store = QdrantChunkStore()
        except Exception as exc:
            logger.warning(f"Qdrant init failed, vector features disabled: {exc}")
            self.vector_store = None
        return self.vector_store

    def create_graph(self, project_id: str, name: str, ontology: Optional[Dict] = None) -> str:
        return self.store.create_graph(project_id=project_id, name=name, ontology=ontology)

    def delete_graph(self, graph_id: str):
        return self.store.delete_graph(graph_id)

    def get_graph_data(self, graph_id: str) -> Dict:
        return self.store.get_graph_data(graph_id)

    def build_graph_from_text(
        self,
        project_id: str,
        text: str,
        ontology: Dict,
        graph_name: str,
        chunk_size: int,
        chunk_overlap: int,
        progress_callback: Optional[Callable[[str, float], None]] = None,
    ) -> Tuple[str, Dict]:
        if progress_callback:
            progress_callback("ローカルグラフを構築しています（Neo4j）...", 0.02)

        graph_id = self.create_graph(project_id=project_id, name=graph_name, ontology=ontology)
        chunks = TextProcessor.split_text(text, chunk_size, chunk_overlap)
        total = max(len(chunks), 1)
        failed_extract_chunks = 0
        heuristic_fallback_chunks = 0
        heuristic_only_mode = False
        consecutive_non_llm_chunks = 0
        heuristic_only_threshold = 1 if LLMClient.uses_reasoning_chat_semantics(Config.EXTRACT_MODEL_NAME) else 3

        for index, chunk in enumerate(chunks):
            ratio = index / total
            if progress_callback:
                progress_callback(f"エンティティ/関係を抽出中: {index + 1}/{len(chunks)}", 0.05 + ratio * 0.85)

            chunk_id = f"chunk_{uuid.uuid4().hex[:12]}"
            self.store.upsert_chunk(project_id=project_id, graph_id=graph_id, chunk_id=chunk_id, text=chunk)

            vector_store = self._get_vector_store()
            if vector_store is not None:
                try:
                    vector_store.add_chunk(
                        project_id=project_id,
                        graph_id=graph_id,
                        chunk_id=chunk_id,
                        text=chunk,
                        extra_payload={"type": "chunk"},
                    )
                except Exception as exc:
                    logger.warning(f"Qdrant add_chunk failed, continue without vectors: {exc}")

            if heuristic_only_mode:
                extracted = self.extractor.extract_heuristic(
                    chunk,
                    ontology=ontology,
                    reason="builder_heuristic_only",
                )
            else:
                try:
                    extracted = self.extractor.extract(chunk, ontology=ontology)
                except Exception as exc:
                    failed_extract_chunks += 1
                    logger.warning(f"Extractor failed for chunk {index + 1}/{len(chunks)}; skipping: {exc}")
                    continue

            strategy = extracted.get("_strategy", "llm")
            if strategy != "llm":
                heuristic_fallback_chunks += 1
                consecutive_non_llm_chunks += 1
                if not heuristic_only_mode and consecutive_non_llm_chunks >= heuristic_only_threshold:
                    heuristic_only_mode = True
                    logger.warning(
                        "LLM 抽出の失敗が続いたため、残りの chunk は heuristic 抽出に切り替えます "
                        f"(after {index + 1} chunks, model={Config.EXTRACT_MODEL_NAME})"
                    )
            else:
                consecutive_non_llm_chunks = 0

            entities_in_chunk = extracted.get("entities") or []
            relations_in_chunk = extracted.get("relations") or []

            entities: List[LocalEntity] = []
            for item in entities_in_chunk:
                raw_type = item.get("type", "")
                canonical_type = canonicalize_entity_type(raw_type)
                entities.append(
                    LocalEntity(
                        project_id=project_id,
                        graph_id=graph_id,
                        name=item.get("name", ""),
                        entity_type=canonical_type,
                        summary=item.get("summary", ""),
                        attributes=item.get("attributes") or {},
                        source_entity_types=[raw_type] if raw_type else [],
                        created_at=_now_iso(),
                    )
                )

            entity_uuids = self.store.upsert_entities(entities)
            self.store.link_mentions(chunk_id=chunk_id, entity_uuids=entity_uuids, graph_id=graph_id)

            uuid_by_key: Dict[str, str] = {}
            for entity in entities:
                uuid_by_key[f"{entity.entity_type}:{entity.name}".lower()] = entity.uuid

            relations: List[LocalRelation] = []
            for item in relations_in_chunk:
                source_type = canonicalize_entity_type(item.get("source_type"))
                target_type = canonicalize_entity_type(item.get("target_type"))
                source_uuid = uuid_by_key.get(f"{source_type}:{item.get('source')}".lower())
                target_uuid = uuid_by_key.get(f"{target_type}:{item.get('target')}".lower())
                if not source_uuid or not target_uuid:
                    continue
                relations.append(
                    LocalRelation(
                        project_id=project_id,
                        graph_id=graph_id,
                        source_uuid=source_uuid,
                        target_uuid=target_uuid,
                        relation_name=item.get("relation", ""),
                        fact=item.get("fact", ""),
                        attributes=item.get("attributes") or {},
                        created_at=_now_iso(),
                    )
                )

            self.store.upsert_relations(relations)

        if progress_callback:
            progress_callback("グラフデータを読み込み中...", 0.95)

        graph_data = self.get_graph_data(graph_id)
        if graph_data.get("node_count", 0) == 0:
            self._seed_graph_from_ontology(project_id=project_id, graph_id=graph_id, ontology=ontology)
            graph_data = self.get_graph_data(graph_id)
        build_warnings: List[str] = []
        if failed_extract_chunks:
            build_warnings.append(
                f"Extractor failed for {failed_extract_chunks}/{len(chunks)} chunks; graph may be incomplete."
            )
        if heuristic_fallback_chunks:
            build_warnings.append(
                f"Heuristic extraction was used for {heuristic_fallback_chunks}/{len(chunks)} chunks."
            )
        if heuristic_only_mode:
            build_warnings.append(
                f"Switched to heuristic-only extraction after repeated LLM failures (model={Config.EXTRACT_MODEL_NAME})."
            )
        if build_warnings:
            graph_data["build_warnings"] = build_warnings
        if progress_callback:
            progress_callback("完了", 1.0)

        return graph_id, graph_data
