"""
Local tools service for ReportAgent when GRAPH_BACKEND=local.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from ..config import Config
from ..utils.logger import get_logger
from .local_graph_store import LocalNeo4jGraphStore
from .local_vector_store import QdrantChunkStore
from .zep_tools import (
    AgentInterview,
    EdgeInfo,
    InsightForgeResult,
    InterviewResult,
    NodeInfo,
    PanoramaResult,
    SearchResult,
)

logger = get_logger("mirofish.local_tools")


class LocalToolsService:
    def __init__(self):
        self.graph_store = LocalNeo4jGraphStore()
        self.vector_store = None
        if Config.VECTOR_BACKEND == "qdrant":
            try:
                self.vector_store = QdrantChunkStore()
            except Exception as exc:
                logger.warning(f"Qdrant init failed, semantic search disabled: {exc}")
                self.vector_store = None

    @staticmethod
    def _load_agent_profiles(simulation_id: str) -> List[Dict[str, Any]]:
        import csv
        import json
        import os

        sim_dir = os.path.join(Config.OASIS_SIMULATION_DATA_DIR, simulation_id)
        profiles: List[Dict[str, Any]] = []

        reddit_profile_path = os.path.join(sim_dir, "reddit_profiles.json")
        if os.path.exists(reddit_profile_path):
            try:
                with open(reddit_profile_path, "r", encoding="utf-8") as handle:
                    profiles = json.load(handle)
                return profiles if isinstance(profiles, list) else []
            except Exception as exc:
                logger.warning(f"Read reddit_profiles.json failed: {exc}")

        twitter_profile_path = os.path.join(sim_dir, "twitter_profiles.csv")
        if os.path.exists(twitter_profile_path):
            try:
                with open(twitter_profile_path, "r", encoding="utf-8") as handle:
                    reader = csv.DictReader(handle)
                    for row in reader:
                        profiles.append(
                            {
                                "realname": row.get("name", ""),
                                "username": row.get("username", ""),
                                "bio": row.get("description", ""),
                                "persona": row.get("user_char", ""),
                                "profession": "未知",
                            }
                        )
                return profiles
            except Exception as exc:
                logger.warning(f"Read twitter_profiles.csv failed: {exc}")

        return []

    def interview_agents(
        self,
        simulation_id: str,
        interview_requirement: str,
        simulation_requirement: str = "",
        max_agents: int = 5,
        custom_questions: Optional[List[str]] = None,
        **_: Any,
    ) -> InterviewResult:
        from .simulation_runner import SimulationRunner

        result = InterviewResult(
            interview_topic=interview_requirement,
            interview_questions=custom_questions or [],
        )

        profiles = self._load_agent_profiles(simulation_id)
        if not profiles:
            result.summary = "未找到可采访的 Agent 人设文件（请先完成 Step2 环境准备）"
            return result

        result.total_agents = len(profiles)
        count = max(0, min(int(max_agents or 0), len(profiles)))
        if count == 0:
            result.summary = "max_agents=0，未执行采访"
            return result

        if count >= len(profiles):
            selected_indices = list(range(len(profiles)))
        else:
            step = max(1, len(profiles) // count)
            selected_indices = list(dict.fromkeys([index * step for index in range(count)]))[:count]

        if custom_questions:
            combined_prompt = "\n".join([item.strip() for item in custom_questions if item and item.strip()][:5])
        else:
            combined_prompt = (
                "请以你的身份与立场，回答以下采访主题，并给出清晰、具体的观点与理由。\n\n"
                f"【模拟背景】{simulation_requirement}\n"
                f"【采访主题】{interview_requirement}\n"
                "要求：不要使用标题；可以分点；尽量引用你在模拟中的观察/经历（如有）。"
            )
            result.interview_questions = [interview_requirement]

        interviews_request = [{"agent_id": idx, "prompt": combined_prompt} for idx in selected_indices]
        if not SimulationRunner.check_env_alive(simulation_id):
            result.summary = "采访失败：模拟环境未运行或已关闭（请保持模拟环境处于运行状态）"
            return result

        api_result = SimulationRunner.interview_agents_batch(
            simulation_id=simulation_id,
            interviews=interviews_request,
            platform=None,
            timeout=180.0,
        )
        if not api_result.get("success", False):
            result.summary = f"采访API调用失败：{api_result.get('error', '未知错误')}"
            return result

        api_data = api_result.get("result", {})
        results_dict = api_data.get("results", {}) if isinstance(api_data, dict) else {}
        for agent_idx in selected_indices:
            agent = profiles[agent_idx] if agent_idx < len(profiles) else {}
            agent_name = agent.get("realname", agent.get("username", f"Agent_{agent_idx}"))
            agent_role = agent.get("profession", "未知")
            agent_bio = agent.get("bio", "")

            twitter_result = results_dict.get(f"twitter_{agent_idx}", {}) or {}
            reddit_result = results_dict.get(f"reddit_{agent_idx}", {}) or {}
            twitter_response = twitter_result.get("response", "") or ""
            reddit_response = reddit_result.get("response", "") or ""

            parts = []
            if twitter_response:
                parts.append(f"【Twitter平台回答】\n{twitter_response}")
            if reddit_response:
                parts.append(f"【Reddit平台回答】\n{reddit_response}")
            response_text = "\n\n".join(parts) if parts else "[无回复]"

            result.interviews.append(
                AgentInterview(
                    agent_name=agent_name,
                    agent_role=agent_role,
                    agent_bio=(agent_bio or "")[:1000],
                    question=combined_prompt,
                    response=response_text,
                    key_quotes=[],
                )
            )

        result.interviewed_count = len(result.interviews)
        result.summary = f"已采访 {result.interviewed_count} 位Agent（本地模式）" if result.interviews else "未获得有效采访回复"
        return result

    def quick_search(self, graph_id: str, query: str, limit: int = 10) -> SearchResult:
        facts: List[str] = []
        if self.vector_store is not None:
            try:
                items = self.vector_store.search_chunks(
                    project_id=None,
                    graph_id=graph_id,
                    query=query,
                    limit=limit,
                )
                facts = [item.get("text", "") for item in items if item.get("text")]
            except Exception as exc:
                logger.warning(f"Local quick_search vector failed: {exc}")

        if not facts:
            graph = self.graph_store.get_graph_data(graph_id)
            for edge in (graph.get("edges") or [])[:limit]:
                fact = edge.get("fact") or ""
                if fact:
                    facts.append(fact)

        return SearchResult(
            facts=facts[:limit],
            edges=[],
            nodes=[],
            query=query,
            total_count=len(facts[:limit]),
        )

    def search_graph(self, graph_id: str, query: str, limit: int = 10, scope: str = "edges") -> SearchResult:
        _ = scope
        return self.quick_search(graph_id=graph_id, query=query, limit=limit)

    def panorama_search(self, graph_id: str, query: str, include_expired: bool = True) -> PanoramaResult:
        _ = include_expired
        graph = self.graph_store.get_graph_data(graph_id)
        nodes = graph.get("nodes") or []
        edges = graph.get("edges") or []

        facts: List[str] = []
        if self.vector_store is not None and query:
            try:
                items = self.vector_store.search_chunks(
                    project_id=None,
                    graph_id=graph_id,
                    query=query,
                    limit=30,
                )
                facts = [item.get("text", "") for item in items if item.get("text")]
            except Exception:
                facts = []
        if not facts:
            facts = [edge.get("fact") for edge in edges if edge.get("fact")]

        node_infos = [
            NodeInfo(
                uuid=node.get("uuid", ""),
                name=node.get("name", ""),
                labels=node.get("labels", []) or ["Entity"],
                summary=node.get("summary", ""),
                attributes=node.get("attributes", {}) or {},
            )
            for node in nodes
        ]
        edge_infos = [
            EdgeInfo(
                uuid=edge.get("uuid", ""),
                name=edge.get("name", ""),
                fact=edge.get("fact", ""),
                source_node_uuid=edge.get("source_node_uuid", ""),
                target_node_uuid=edge.get("target_node_uuid", ""),
                source_node_name=edge.get("source_node_name"),
                target_node_name=edge.get("target_node_name"),
                created_at=edge.get("created_at"),
                valid_at=edge.get("valid_at"),
                invalid_at=edge.get("invalid_at"),
                expired_at=edge.get("expired_at"),
            )
            for edge in edges
        ]

        result = PanoramaResult(query=query)
        result.all_nodes = node_infos
        result.all_edges = edge_infos
        result.total_nodes = len(node_infos)
        result.total_edges = len(edge_infos)
        result.active_facts = facts
        result.historical_facts = []
        result.active_count = len(result.active_facts)
        result.historical_count = 0
        return result

    def insight_forge(
        self,
        graph_id: str,
        query: str,
        simulation_requirement: str = "",
        report_context: str = "",
    ) -> InsightForgeResult:
        _ = report_context
        search = self.quick_search(graph_id=graph_id, query=query, limit=15)
        graph = self.graph_store.get_graph_data(graph_id)
        nodes = graph.get("nodes") or []
        edges = graph.get("edges") or []

        entity_insights: List[Dict[str, Any]] = []
        for node in nodes[:10]:
            entity_type = next((label for label in (node.get("labels") or []) if label not in ["Entity", "Node"]), "实体")
            entity_insights.append(
                {
                    "name": node.get("name", ""),
                    "type": entity_type,
                    "summary": node.get("summary", ""),
                    "related_facts": [],
                }
            )

        relationship_chains = []
        for edge in edges[:20]:
            source = edge.get("source_node_name") or (edge.get("source_node_uuid", "")[:8])
            target = edge.get("target_node_name") or (edge.get("target_node_uuid", "")[:8])
            relation = edge.get("name") or edge.get("fact_type") or "REL"
            relationship_chains.append(f"{source} --[{relation}]--> {target}")

        result = InsightForgeResult(
            query=query,
            simulation_requirement=simulation_requirement or "",
            sub_queries=[query],
        )
        result.semantic_facts = search.facts
        result.entity_insights = entity_insights
        result.relationship_chains = relationship_chains
        result.total_facts = len(result.semantic_facts)
        result.total_entities = len(result.entity_insights)
        result.total_relationships = len(result.relationship_chains)
        return result

    def get_graph_statistics(self, graph_id: str) -> Dict[str, Any]:
        graph = self.graph_store.get_graph_data(graph_id)
        return {
            "graph_id": graph_id,
            "node_count": graph.get("node_count", 0),
            "edge_count": graph.get("edge_count", 0),
        }

    def get_entities_by_type(self, graph_id: str, entity_type: str) -> List[NodeInfo]:
        graph = self.graph_store.get_graph_data(graph_id)
        result: List[NodeInfo] = []
        for node in graph.get("nodes") or []:
            labels = node.get("labels") or []
            if entity_type and entity_type not in labels:
                continue
            result.append(
                NodeInfo(
                    uuid=node.get("uuid", ""),
                    name=node.get("name", ""),
                    labels=labels or ["Entity"],
                    summary=node.get("summary", ""),
                    attributes=node.get("attributes", {}) or {},
                )
            )
        return result

    def get_entity_summary(self, graph_id: str, entity_name: str) -> Dict[str, Any]:
        graph = self.graph_store.get_graph_data(graph_id)
        for node in graph.get("nodes") or []:
            if (node.get("name") or "").strip() == (entity_name or "").strip():
                entity_type = next((label for label in (node.get("labels") or []) if label not in ["Entity", "Node"]), "实体")
                return {
                    "name": node.get("name", ""),
                    "type": entity_type,
                    "summary": node.get("summary", ""),
                    "attributes": node.get("attributes", {}) or {},
                }
        return {"name": entity_name, "summary": "", "attributes": {}}

    def get_simulation_context(
        self,
        graph_id: str,
        simulation_requirement: str = "",
        limit: int = 30,
        query: Optional[str] = None,
        **_: Any,
    ) -> Dict[str, Any]:
        effective_query = (query or simulation_requirement or "").strip()
        search = self.quick_search(graph_id=graph_id, query=effective_query, limit=limit)
        stats = self.get_graph_statistics(graph_id)
        graph = self.graph_store.get_graph_data(graph_id)

        entities: List[Dict[str, Any]] = []
        for node in graph.get("nodes") or []:
            labels = node.get("labels") or []
            custom_labels = [label for label in labels if label not in ["Entity", "Node"]]
            if not custom_labels:
                continue
            entities.append(
                {
                    "name": node.get("name", ""),
                    "type": custom_labels[0],
                    "summary": node.get("summary", ""),
                }
            )

        return {
            "simulation_requirement": simulation_requirement,
            "related_facts": search.facts,
            "graph_statistics": stats,
            "entities": entities[:limit],
            "total_entities": len(entities),
        }
