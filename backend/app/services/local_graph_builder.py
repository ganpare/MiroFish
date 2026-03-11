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
            progress_callback("创建本地图谱（Neo4j）...", 0.02)

        graph_id = self.create_graph(project_id=project_id, name=graph_name, ontology=ontology)
        chunks = TextProcessor.split_text(text, chunk_size, chunk_overlap)
        total = max(len(chunks), 1)
        failed_extract_chunks = 0

        for index, chunk in enumerate(chunks):
            ratio = index / total
            if progress_callback:
                progress_callback(f"抽取实体/关系: {index + 1}/{len(chunks)}", 0.05 + ratio * 0.85)

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

            try:
                extracted = self.extractor.extract(chunk, ontology=ontology)
            except Exception as exc:
                failed_extract_chunks += 1
                logger.warning(f"Extractor failed for chunk {index + 1}/{len(chunks)}; skipping: {exc}")
                continue

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
            progress_callback("读取图谱数据...", 0.95)

        graph_data = self.get_graph_data(graph_id)
        if failed_extract_chunks:
            graph_data["build_warnings"] = [
                f"Extractor failed for {failed_extract_chunks}/{len(chunks)} chunks; graph may be incomplete."
            ]
        if progress_callback:
            progress_callback("完成", 1.0)

        return graph_id, graph_data
