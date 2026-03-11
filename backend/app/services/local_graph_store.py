"""
Local graph store (Neo4j).

Graph data is isolated by project_id and graph_id.
"""

from __future__ import annotations

import hashlib
import json
import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional

from neo4j import Driver, GraphDatabase

from ..config import Config
from ..utils.logger import get_logger

logger = get_logger("mirofish.local_graph_store")


def _now_iso() -> str:
    return datetime.now().isoformat()


def _stable_entity_uuid(project_id: str, entity_type: str, name: str) -> str:
    normalized = (name or "").strip().lower()
    digest = hashlib.sha1(f"{project_id}:{entity_type}:{normalized}".encode("utf-8")).hexdigest()[:16]
    return f"ent_{digest}"


@dataclass(frozen=True)
class LocalEntity:
    project_id: str
    graph_id: str
    name: str
    entity_type: str
    summary: str = ""
    attributes: Optional[Dict[str, Any]] = None
    source_entity_types: Optional[List[str]] = None
    created_at: Optional[str] = None

    @property
    def uuid(self) -> str:
        return _stable_entity_uuid(self.project_id, self.entity_type, self.name)


@dataclass(frozen=True)
class LocalRelation:
    project_id: str
    graph_id: str
    source_uuid: str
    target_uuid: str
    relation_name: str
    fact: str = ""
    attributes: Optional[Dict[str, Any]] = None
    created_at: Optional[str] = None
    uuid: str = ""


class LocalNeo4jGraphStore:
    def __init__(self):
        self._driver: Driver = GraphDatabase.driver(
            Config.NEO4J_URI,
            auth=(Config.NEO4J_USER, Config.NEO4J_PASSWORD),
        )
        self._database = Config.NEO4J_DATABASE
        self._ensure_schema()

    def close(self):
        try:
            self._driver.close()
        except Exception:
            pass

    def _ensure_schema(self) -> None:
        statements = [
            "CREATE CONSTRAINT graph_id_unique IF NOT EXISTS FOR (g:Graph) REQUIRE g.graph_id IS UNIQUE",
            "CREATE CONSTRAINT entity_uuid_unique IF NOT EXISTS FOR (e:Entity) REQUIRE e.uuid IS UNIQUE",
            "CREATE INDEX entity_graph_id IF NOT EXISTS FOR (e:Entity) ON (e.graph_id)",
            "CREATE INDEX entity_project_id IF NOT EXISTS FOR (e:Entity) ON (e.project_id)",
            "CREATE INDEX relation_graph_id IF NOT EXISTS FOR ()-[r:REL]-() ON (r.graph_id)",
            "CREATE INDEX chunk_graph_id IF NOT EXISTS FOR (c:Chunk) ON (c.graph_id)",
        ]

        with self._driver.session(database=self._database) as session:
            for cypher in statements:
                try:
                    session.run(cypher)
                except Exception as exc:
                    logger.warning(f"Neo4j schema statement failed: {cypher} err={str(exc)[:120]}")

    def create_graph(self, project_id: str, name: str, ontology: Optional[Dict[str, Any]] = None) -> str:
        graph_id = f"mirofish_local_{uuid.uuid4().hex[:16]}"
        with self._driver.session(database=self._database) as session:
            session.run(
                """
                CREATE (g:Graph {
                    graph_id: $graph_id,
                    project_id: $project_id,
                    name: $name,
                    ontology_json: $ontology_json,
                    created_at: $created_at
                })
                """,
                graph_id=graph_id,
                project_id=project_id,
                name=name,
                ontology_json=json.dumps(ontology or {}, ensure_ascii=False),
                created_at=_now_iso(),
            )
        return graph_id

    def delete_graph(self, graph_id: str) -> None:
        with self._driver.session(database=self._database) as session:
            session.run("MATCH (g:Graph {graph_id: $graph_id}) DETACH DELETE g", graph_id=graph_id)
            session.run("MATCH (c:Chunk {graph_id: $graph_id}) DETACH DELETE c", graph_id=graph_id)
            session.run("MATCH ()-[r:REL {graph_id: $graph_id}]->() DELETE r", graph_id=graph_id)
            session.run("MATCH (e:Entity {graph_id: $graph_id}) DETACH DELETE e", graph_id=graph_id)

    def upsert_entities(self, entities: Iterable[LocalEntity]) -> List[str]:
        uuids: List[str] = []
        with self._driver.session(database=self._database) as session:
            for entity in entities:
                uuids.append(entity.uuid)
                session.run(
                    """
                    MERGE (e:Entity {uuid: $uuid})
                    SET e.project_id = $project_id,
                        e.graph_id = $graph_id,
                        e.name = $name,
                        e.entity_type = $entity_type,
                        e.summary = CASE
                            WHEN $summary IS NULL OR $summary = "" THEN COALESCE(e.summary, "")
                            ELSE $summary
                        END,
                        e.attributes_json = CASE
                            WHEN $attributes_json IS NULL OR $attributes_json = "{}" THEN COALESCE(e.attributes_json, "{}")
                            ELSE $attributes_json
                        END,
                        e.source_entity_types = CASE
                            WHEN e.source_entity_types IS NULL THEN $source_entity_types
                            ELSE e.source_entity_types + [t IN $source_entity_types WHERE NOT t IN e.source_entity_types]
                        END,
                        e.created_at = COALESCE(e.created_at, $created_at)
                    """,
                    uuid=entity.uuid,
                    project_id=entity.project_id,
                    graph_id=entity.graph_id,
                    name=entity.name,
                    entity_type=entity.entity_type,
                    summary=entity.summary or "",
                    attributes_json=json.dumps(entity.attributes or {}, ensure_ascii=False),
                    source_entity_types=list(dict.fromkeys([t for t in (entity.source_entity_types or []) if t])),
                    created_at=entity.created_at or _now_iso(),
                )
        return uuids

    def upsert_chunk(self, project_id: str, graph_id: str, chunk_id: str, text: str) -> None:
        with self._driver.session(database=self._database) as session:
            session.run(
                """
                MERGE (c:Chunk {chunk_id: $chunk_id})
                SET c.project_id = $project_id,
                    c.graph_id = $graph_id,
                    c.text = $text,
                    c.created_at = COALESCE(c.created_at, $created_at)
                WITH c
                MATCH (g:Graph {graph_id: $graph_id})
                MERGE (g)-[:HAS_CHUNK]->(c)
                """,
                chunk_id=chunk_id,
                project_id=project_id,
                graph_id=graph_id,
                text=text,
                created_at=_now_iso(),
            )

    def link_mentions(self, chunk_id: str, entity_uuids: Iterable[str], graph_id: str) -> None:
        uuids = list(entity_uuids)
        if not uuids:
            return
        with self._driver.session(database=self._database) as session:
            session.run(
                """
                MATCH (c:Chunk {chunk_id: $chunk_id, graph_id: $graph_id})
                UNWIND $entity_uuids AS entity_uuid
                MATCH (e:Entity {uuid: entity_uuid, graph_id: $graph_id})
                MERGE (c)-[:MENTIONS]->(e)
                """,
                chunk_id=chunk_id,
                graph_id=graph_id,
                entity_uuids=uuids,
            )

    def upsert_relations(self, relations: Iterable[LocalRelation]) -> None:
        with self._driver.session(database=self._database) as session:
            for relation in relations:
                rel_uuid = relation.uuid or f"rel_{uuid.uuid4().hex[:16]}"
                session.run(
                    """
                    MATCH (s:Entity {uuid: $source_uuid, graph_id: $graph_id})
                    MATCH (t:Entity {uuid: $target_uuid, graph_id: $graph_id})
                    MERGE (s)-[r:REL {uuid: $uuid}]->(t)
                    SET r.project_id = $project_id,
                        r.graph_id = $graph_id,
                        r.name = $relation_name,
                        r.fact = $fact,
                        r.fact_type = $relation_name,
                        r.attributes_json = $attributes_json,
                        r.created_at = COALESCE(r.created_at, $created_at)
                    """,
                    uuid=rel_uuid,
                    project_id=relation.project_id,
                    graph_id=relation.graph_id,
                    source_uuid=relation.source_uuid,
                    target_uuid=relation.target_uuid,
                    relation_name=relation.relation_name,
                    fact=relation.fact or "",
                    attributes_json=json.dumps(relation.attributes or {}, ensure_ascii=False),
                    created_at=relation.created_at or _now_iso(),
                )

    def get_graph_data(self, graph_id: str) -> Dict[str, Any]:
        with self._driver.session(database=self._database) as session:
            node_records = session.run(
                """
                MATCH (e:Entity {graph_id: $graph_id})
                RETURN e.uuid AS uuid, e.name AS name, e.entity_type AS entity_type,
                       e.summary AS summary, e.attributes_json AS attributes_json,
                       e.source_entity_types AS source_entity_types,
                       e.created_at AS created_at
                """,
                graph_id=graph_id,
            )

            nodes: List[Dict[str, Any]] = []
            node_name_map: Dict[str, str] = {}
            for record in node_records:
                try:
                    attributes = json.loads(record.get("attributes_json") or "{}")
                except Exception:
                    attributes = {}
                source_types = record.get("source_entity_types")
                if isinstance(source_types, list):
                    attributes["source_entity_types"] = source_types

                uuid_ = record.get("uuid")
                name = record.get("name") or ""
                node_name_map[uuid_] = name
                entity_type = record.get("entity_type") or "Entity"
                nodes.append(
                    {
                        "uuid": uuid_,
                        "name": name,
                        "labels": ["Entity", entity_type],
                        "summary": record.get("summary") or "",
                        "attributes": attributes,
                        "created_at": record.get("created_at"),
                    }
                )

            edge_records = session.run(
                """
                MATCH (s:Entity {graph_id: $graph_id})-[r:REL {graph_id: $graph_id}]->(t:Entity {graph_id: $graph_id})
                RETURN r.uuid AS uuid, r.name AS name, r.fact AS fact, r.fact_type AS fact_type,
                       r.attributes_json AS attributes_json, r.created_at AS created_at,
                       s.uuid AS source_uuid, t.uuid AS target_uuid
                """,
                graph_id=graph_id,
            )

            edges: List[Dict[str, Any]] = []
            for record in edge_records:
                try:
                    attributes = json.loads(record.get("attributes_json") or "{}")
                except Exception:
                    attributes = {}
                source_uuid = record.get("source_uuid")
                target_uuid = record.get("target_uuid")
                edges.append(
                    {
                        "uuid": record.get("uuid"),
                        "name": record.get("name") or "",
                        "fact": record.get("fact") or "",
                        "fact_type": record.get("fact_type") or (record.get("name") or ""),
                        "source_node_uuid": source_uuid,
                        "target_node_uuid": target_uuid,
                        "source_node_name": node_name_map.get(source_uuid, ""),
                        "target_node_name": node_name_map.get(target_uuid, ""),
                        "attributes": attributes,
                        "created_at": record.get("created_at"),
                        "valid_at": None,
                        "invalid_at": None,
                        "expired_at": None,
                        "episodes": [],
                    }
                )

        return {
            "graph_id": graph_id,
            "nodes": nodes,
            "edges": edges,
            "node_count": len(nodes),
            "edge_count": len(edges),
        }
