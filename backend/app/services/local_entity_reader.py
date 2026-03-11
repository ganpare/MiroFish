"""
Local entity reader for Neo4j graph backend.

Implements the same high-level API as ZepEntityReader so the rest of the codebase
can keep using FilteredEntities/EntityNode data structures.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Set

from .entity_type_normalizer import canonicalize_entity_type
from .local_graph_store import LocalNeo4jGraphStore
from .zep_entity_reader import EntityNode, FilteredEntities


class LocalEntityReader:
    def __init__(self):
        self.store = LocalNeo4jGraphStore()

    def filter_defined_entities(
        self,
        graph_id: str,
        defined_entity_types: Optional[List[str]] = None,
        enrich_with_edges: bool = True,
    ) -> FilteredEntities:
        graph_data = self.store.get_graph_data(graph_id)
        nodes = graph_data.get("nodes") or []
        edges = graph_data.get("edges") or []
        total_count = len(nodes)

        def _entity_type(node: Dict[str, Any]) -> Optional[str]:
            for label in node.get("labels") or []:
                if label not in ["Entity", "Node"]:
                    return label
            return None

        filtered_nodes = []
        entity_types: Set[str] = set()
        defined_set = set(defined_entity_types or [])
        canonical_defined_set = {canonicalize_entity_type(item) for item in defined_set} if defined_set else set()

        for node in nodes:
            entity_type = _entity_type(node)
            if entity_type:
                entity_types.add(entity_type)
            if defined_set:
                if entity_type not in canonical_defined_set:
                    source_types = (node.get("attributes") or {}).get("source_entity_types") or []
                    if not (set(source_types) & defined_set):
                        continue
            filtered_nodes.append(node)

        filtered_uuids = {node.get("uuid") for node in filtered_nodes if node.get("uuid")}
        related_edges_by_uuid: Dict[str, List[Dict[str, Any]]] = {uuid_: [] for uuid_ in filtered_uuids}
        related_nodes_by_uuid: Dict[str, List[Dict[str, Any]]] = {uuid_: [] for uuid_ in filtered_uuids}

        if enrich_with_edges:
            for edge in edges:
                source_uuid = edge.get("source_node_uuid")
                target_uuid = edge.get("target_node_uuid")
                if source_uuid in filtered_uuids:
                    related_edges_by_uuid[source_uuid].append(edge)
                if target_uuid in filtered_uuids and target_uuid != source_uuid:
                    related_edges_by_uuid[target_uuid].append(edge)

            node_lookup = {node.get("uuid"): node for node in nodes if node.get("uuid")}
            for uuid_ in filtered_uuids:
                related_ids = set()
                for edge in related_edges_by_uuid.get(uuid_, []):
                    source_uuid = edge.get("source_node_uuid")
                    target_uuid = edge.get("target_node_uuid")
                    other_uuid = target_uuid if source_uuid == uuid_ else source_uuid
                    if other_uuid and other_uuid in node_lookup:
                        related_ids.add(other_uuid)
                related_nodes_by_uuid[uuid_] = [node_lookup[item] for item in related_ids]

        entities: List[EntityNode] = []
        for node in filtered_nodes:
            uuid_ = node.get("uuid") or ""
            entities.append(
                EntityNode(
                    uuid=uuid_,
                    name=node.get("name") or "",
                    labels=node.get("labels") or ["Entity"],
                    summary=node.get("summary") or "",
                    attributes=node.get("attributes") or {},
                    related_edges=related_edges_by_uuid.get(uuid_, []),
                    related_nodes=related_nodes_by_uuid.get(uuid_, []),
                )
            )

        return FilteredEntities(
            entities=entities,
            entity_types=entity_types,
            total_count=total_count,
            filtered_count=len(entities),
        )

    def get_entity_with_context(self, graph_id: str, entity_uuid: str) -> Optional[EntityNode]:
        filtered = self.filter_defined_entities(graph_id=graph_id, defined_entity_types=None, enrich_with_edges=True)
        for entity in filtered.entities:
            if entity.uuid == entity_uuid:
                return entity
        return None

    def get_entities_by_type(
        self,
        graph_id: str,
        entity_type: str,
        enrich_with_edges: bool = True,
    ) -> List[EntityNode]:
        filtered = self.filter_defined_entities(
            graph_id=graph_id,
            defined_entity_types=[entity_type],
            enrich_with_edges=enrich_with_edges,
        )
        return filtered.entities
