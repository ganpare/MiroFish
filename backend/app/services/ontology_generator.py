"""
Ontology generation service.
"""

import json
from typing import Any, Dict, List, Optional

from ..config import Config
from ..utils.llm_client import LLMClient
from .genre_schema import GenreSchemaRegistry


ONTOLOGY_BASE_SYSTEM_PROMPT = """你是一个专业的知识图谱本体设计专家，同时负责把文本解释成适合仿真的 ontology schema。

你必须同时区分四层：
1. entity types: 哪些类型是图里的节点
2. relation types: 节点之间的主要关系
3. agentizable types: 哪些节点允许被 agent 化
4. simulation grammar: 在该文本类型里，行动的基本动词是什么

请严格遵守：
- 只输出有效 JSON，不要输出任何额外说明
- entity_types 最多 10 个，edge_types 最多 10 个
- entity_types 使用英文 PascalCase
- edge_types 使用英文 UPPER_SNAKE_CASE
- agentizable_types 必须是 entity_types 的子集
- non_agentizable_types 只能来自 entity_types
- simulation_grammar 只为 agentizable_types 提供动作列表
- 属性名不能使用 name、uuid、group_id、created_at、summary
- 如果文本更适合概念、制度、派别、国家等主体，不要强行人格化为普通聊天角色

输出 JSON 结构：
{
  "genre": "主 genre",
  "schema_overlays": ["可选 overlay"],
  "entity_types": [
    {
      "name": "EntityType",
      "description": "Short English description",
      "attributes": [
        {
          "name": "attribute_name",
          "type": "text",
          "description": "Attribute description"
        }
      ],
      "examples": ["Example 1", "Example 2"]
    }
  ],
  "edge_types": [
    {
      "name": "RELATION_NAME",
      "description": "Short English description",
      "source_targets": [{"source": "EntityA", "target": "EntityB"}],
      "attributes": []
    }
  ],
  "agentizable_types": ["EntityType"],
  "non_agentizable_types": ["EntityType"],
  "simulation_grammar": {
    "EntityType": ["verb_a", "verb_b"]
  },
  "report_template": {
    "sections": ["Section A", "Section B"]
  },
  "analysis_summary": "中文简要总结"
}
"""


class OntologyGenerator:
    """
    本体生成器
    分析文本内容，生成实体和关系类型定义
    """
    
    MAX_TEXT_LENGTH_FOR_LLM = 50000
    MAX_TEXT_LENGTH_FOR_GENRE_INFERENCE = 12000
    MAX_ENTITY_TYPES = 10
    MAX_EDGE_TYPES = 10
    RESERVED_ATTRIBUTE_NAMES = {"name", "uuid", "group_id", "created_at", "summary"}

    def __init__(
        self,
        llm_client: Optional[LLMClient] = None,
        schema_registry: Optional[GenreSchemaRegistry] = None
    ):
        self.llm_client = llm_client or LLMClient(
            api_key=Config.EXTRACT_API_KEY,
            base_url=Config.EXTRACT_BASE_URL,
            model=Config.EXTRACT_MODEL_NAME,
        )
        self.schema_registry = schema_registry or GenreSchemaRegistry()
    
    def generate(
        self,
        document_texts: List[str],
        simulation_requirement: str,
        additional_context: Optional[str] = None,
        preferred_genre: Optional[str] = None,
        overlay_genres: Optional[List[str]] = None,
        auto_detect_genre: bool = True
    ) -> Dict[str, Any]:
        """
        生成本体定义
        
        Args:
            document_texts: 文档文本列表
            simulation_requirement: 模拟需求描述
            additional_context: 额外上下文
            
        Returns:
            本体定义（entity_types, edge_types等）
        """
        schema_context = self._resolve_schema_context(
            document_texts=document_texts,
            simulation_requirement=simulation_requirement,
            additional_context=additional_context,
            preferred_genre=preferred_genre,
            overlay_genres=overlay_genres,
            auto_detect_genre=auto_detect_genre,
        )

        user_message = self._build_user_message(
            document_texts, 
            simulation_requirement,
            additional_context,
            schema_context,
        )
        
        messages = [
            {"role": "system", "content": self._build_system_prompt(schema_context)},
            {"role": "user", "content": user_message}
        ]
        
        # 调用LLM
        result = self.llm_client.chat_json(
            messages=messages,
            temperature=0.3,
            max_tokens=4096
        )
        
        # 验证和后处理
        result = self._validate_and_process(result, schema_context)
        
        return result

    def _resolve_schema_context(
        self,
        document_texts: List[str],
        simulation_requirement: str,
        additional_context: Optional[str],
        preferred_genre: Optional[str],
        overlay_genres: Optional[List[str]],
        auto_detect_genre: bool,
    ) -> Dict[str, Any]:
        schema_names = set(self.schema_registry.get_schema_names())
        cleaned_overlays = [g for g in (overlay_genres or []) if g in schema_names]

        if preferred_genre and preferred_genre in schema_names:
            primary_genre = preferred_genre
            candidates = [{"genre": preferred_genre, "reason": "user_selected"}]
            reasoning = "使用用户指定的 genre schema"
        elif auto_detect_genre:
            selection = self._infer_schema_selection(
                document_texts=document_texts,
                simulation_requirement=simulation_requirement,
                additional_context=additional_context,
            )
            primary_genre = selection["genre"]
            cleaned_overlays = selection["schema_overlays"] + [g for g in cleaned_overlays if g != selection["genre"]]
            cleaned_overlays = list(dict.fromkeys([g for g in cleaned_overlays if g != primary_genre]))
            candidates = selection["genre_candidates"]
            reasoning = selection["genre_inference_reasoning"]
        else:
            primary_genre = "public_opinion"
            candidates = [{"genre": "public_opinion", "reason": "default"}]
            reasoning = "未启用自动 genre 推断，使用默认 schema"

        schema_context = self.schema_registry.merge_schema(primary_genre, cleaned_overlays)
        schema_context["genre_candidates"] = candidates
        schema_context["genre_inference_reasoning"] = reasoning
        return schema_context

    def _infer_schema_selection(
        self,
        document_texts: List[str],
        simulation_requirement: str,
        additional_context: Optional[str],
    ) -> Dict[str, Any]:
        combined_text = "\n\n---\n\n".join(document_texts)
        excerpt = combined_text[:self.MAX_TEXT_LENGTH_FOR_GENRE_INFERENCE]
        if len(combined_text) > self.MAX_TEXT_LENGTH_FOR_GENRE_INFERENCE:
            excerpt += "\n...(文档已截断，仅用于 genre 推断)..."

        schema_descriptions = []
        for schema in self.schema_registry.list_schemas():
            schema_descriptions.append(
                f"- {schema['genre']}: {schema['description']} | entity_types={', '.join(schema['entity_types'][:8])}"
            )

        prompt = f"""请判断下面文本最适合哪种 schema genre，并给出最多 2 个 overlay。

可选 genre:
{chr(10).join(schema_descriptions)}

模拟需求:
{simulation_requirement}

文本摘录:
{excerpt}

额外说明:
{additional_context or "无"}

请输出 JSON:
{{
  "genre": "主 genre",
  "schema_overlays": ["overlay1", "overlay2"],
  "genre_candidates": [
    {{"genre": "候选 genre", "reason": "简短原因"}}
  ],
  "genre_inference_reasoning": "中文简短说明"
}}
"""

        try:
            result = self.llm_client.chat_json(
                messages=[
                    {
                        "role": "system",
                        "content": "你是文本体裁与 ontology schema 匹配专家。只返回 JSON。"
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.1,
                max_tokens=1200,
            )
            valid_names = set(self.schema_registry.get_schema_names())
            genre = result.get("genre", "public_opinion")
            if genre not in valid_names:
                genre = "public_opinion"

            overlays = [
                item for item in result.get("schema_overlays", [])
                if item in valid_names and item != genre
            ][:2]

            candidates = []
            for item in result.get("genre_candidates", []):
                if not isinstance(item, dict):
                    continue
                candidate_genre = item.get("genre")
                if candidate_genre not in valid_names:
                    continue
                candidates.append({
                    "genre": candidate_genre,
                    "reason": item.get("reason", "")
                })

            if not candidates:
                candidates = [{"genre": genre, "reason": "llm_selected"}]

            return {
                "genre": genre,
                "schema_overlays": overlays,
                "genre_candidates": candidates[:3],
                "genre_inference_reasoning": result.get("genre_inference_reasoning", ""),
            }
        except Exception:
            fallback_candidates = self.schema_registry.heuristic_candidates(
                f"{simulation_requirement}\n{combined_text if 'combined_text' in locals() else excerpt}"
            )
            fallback_genre = fallback_candidates[0]["genre"] if fallback_candidates else "public_opinion"
            return {
                "genre": fallback_genre,
                "schema_overlays": [],
                "genre_candidates": [
                    {"genre": item["genre"], "reason": item["reason"]} for item in fallback_candidates
                ] or [{"genre": fallback_genre, "reason": "heuristic_default"}],
                "genre_inference_reasoning": "LLM 推断失败，使用关键词启发式匹配",
            }

    def _build_system_prompt(self, schema_context: Dict[str, Any]) -> str:
        genre = schema_context["genre"]
        overlays = schema_context.get("schema_overlays", [])
        overlay_text = ", ".join(overlays) if overlays else "none"
        entity_types = ", ".join(schema_context.get("entity_types", []))
        relation_types = ", ".join(schema_context.get("relation_types", []))
        agentizable = ", ".join(schema_context.get("agentizable_types", []))
        report_sections = ", ".join(schema_context.get("report_template", {}).get("sections", []))

        return (
            ONTOLOGY_BASE_SYSTEM_PROMPT
            + "\n\n## 当前 schema 约束\n"
            + f"- primary_genre: {genre}\n"
            + f"- overlays: {overlay_text}\n"
            + f"- schema_description: {schema_context.get('description', '')}\n"
            + f"- candidate_entity_types: {entity_types}\n"
            + f"- candidate_relation_types: {relation_types}\n"
            + f"- preferred_agentizable_types: {agentizable}\n"
            + f"- preferred_report_sections: {report_sections}\n"
            + f"- genre_inference_reasoning: {schema_context.get('genre_inference_reasoning', '')}\n"
            + "\n设计时优先遵循该 schema；如果文本需要，可以做少量偏离，但必须保持语义收敛。"
        )
    
    def _build_user_message(
        self,
        document_texts: List[str],
        simulation_requirement: str,
        additional_context: Optional[str],
        schema_context: Dict[str, Any],
    ) -> str:
        """构建用户消息"""
        
        # 合并文本
        combined_text = "\n\n---\n\n".join(document_texts)
        original_length = len(combined_text)
        
        # 如果文本超过5万字，截断（仅影响传给LLM的内容，不影响图谱构建）
        if len(combined_text) > self.MAX_TEXT_LENGTH_FOR_LLM:
            combined_text = combined_text[:self.MAX_TEXT_LENGTH_FOR_LLM]
            combined_text += f"\n\n...(原文共{original_length}字，已截取前{self.MAX_TEXT_LENGTH_FOR_LLM}字用于本体分析)..."
        
        message = f"""## 模拟需求

{simulation_requirement}

## 文档内容

{combined_text}
"""
        
        if additional_context:
            message += f"""
## 额外说明

{additional_context}
"""
        
        candidate_summary = "\n".join(
            f"- {item['genre']}: {item.get('reason', '')}"
            for item in schema_context.get("genre_candidates", [])
        ) or "- 无"

        message += f"""
## 目标 schema

- primary_genre: {schema_context.get("genre")}
- overlays: {", ".join(schema_context.get("schema_overlays", [])) or "none"}
- candidate genres:
{candidate_summary}

## 设计要求

1. entity_types 最多 {self.MAX_ENTITY_TYPES} 个，edge_types 最多 {self.MAX_EDGE_TYPES} 个
2. agentizable_types 必须是 entity_types 的子集
3. non_agentizable_types 必须来自 entity_types
4. simulation_grammar 只为 agentizable_types 提供动作
5. 重要但不该行动的对象，也要纳入 entity_types，并放入 non_agentizable_types
6. 若 schema 中建议以 Concept、School、Institution、State、Faction 等为主体，不要硬转成人类聊天账号
7. report_template.sections 应与 genre 对应
"""
        
        return message
    
    def _sanitize_attributes(self, attributes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        sanitized = []
        for attr in attributes[:3]:
            name = str(attr.get("name", "")).strip()
            if not name or name in self.RESERVED_ATTRIBUTE_NAMES:
                continue
            sanitized.append({
                "name": name,
                "type": attr.get("type", "text"),
                "description": str(attr.get("description", name))[:100],
            })
        return sanitized

    def _build_required_entity_definition(self, name: str) -> Dict[str, Any]:
        if name == "Person":
            return {
                "name": "Person",
                "description": "Any individual person not fitting other specific person types.",
                "attributes": [
                    {"name": "full_name", "type": "text", "description": "Full name of the person"},
                    {"name": "role", "type": "text", "description": "Role or occupation"},
                ],
                "examples": ["ordinary citizen", "anonymous netizen"],
            }
        if name == "Organization":
            return {
                "name": "Organization",
                "description": "Any organization not fitting other specific organization types.",
                "attributes": [
                    {"name": "org_name", "type": "text", "description": "Name of the organization"},
                    {"name": "org_type", "type": "text", "description": "Type of organization"},
                ],
                "examples": ["small business", "community group"],
            }
        return {
            "name": name,
            "description": f"{name} entity relevant to the selected genre schema.",
            "attributes": [],
            "examples": [],
        }

    def _normalize_report_template(self, report_template: Any, fallback_sections: List[str]) -> Dict[str, Any]:
        if isinstance(report_template, dict):
            sections = report_template.get("sections", [])
        elif isinstance(report_template, list):
            sections = report_template
        else:
            sections = []

        normalized_sections: List[str] = []
        for section in sections or fallback_sections:
            text = str(section).strip()
            if text and text not in normalized_sections:
                normalized_sections.append(text)

        return {"sections": normalized_sections}

    def _validate_and_process(self, result: Dict[str, Any], schema_context: Dict[str, Any]) -> Dict[str, Any]:
        """验证和后处理结果"""
        
        # 确保必要字段存在
        if "entity_types" not in result:
            result["entity_types"] = []
        if "edge_types" not in result:
            result["edge_types"] = []
        if "analysis_summary" not in result:
            result["analysis_summary"] = ""
        if "agentizable_types" not in result:
            result["agentizable_types"] = []
        if "non_agentizable_types" not in result:
            result["non_agentizable_types"] = []
        if "simulation_grammar" not in result or not isinstance(result["simulation_grammar"], dict):
            result["simulation_grammar"] = {}
        
        # 验证实体类型
        for entity in result["entity_types"]:
            if "attributes" not in entity:
                entity["attributes"] = []
            if "examples" not in entity:
                entity["examples"] = []
            entity["attributes"] = self._sanitize_attributes(entity.get("attributes", []))
            # 确保description不超过100字符
            if len(entity.get("description", "")) > 100:
                entity["description"] = entity["description"][:97] + "..."
        
        # 验证关系类型
        for edge in result["edge_types"]:
            if "source_targets" not in edge:
                edge["source_targets"] = []
            if "attributes" not in edge:
                edge["attributes"] = []
            edge["attributes"] = self._sanitize_attributes(edge.get("attributes", []))
            if len(edge.get("description", "")) > 100:
                edge["description"] = edge["description"][:97] + "..."
        
        entity_names = {e["name"] for e in result["entity_types"] if e.get("name")}

        missing_required = [
            name for name in schema_context.get("required_entity_types", [])
            if name not in entity_names
        ]
        for name in missing_required:
            if len(result["entity_types"]) >= self.MAX_ENTITY_TYPES:
                result["entity_types"].pop()
            result["entity_types"].append(self._build_required_entity_definition(name))

        if len(result["entity_types"]) > self.MAX_ENTITY_TYPES:
            result["entity_types"] = result["entity_types"][:self.MAX_ENTITY_TYPES]
        if len(result["edge_types"]) > self.MAX_EDGE_TYPES:
            result["edge_types"] = result["edge_types"][:self.MAX_EDGE_TYPES]

        valid_entity_names = [e["name"] for e in result["entity_types"] if e.get("name")]
        valid_entity_set = set(valid_entity_names)

        agentizable_types = [
            item for item in result.get("agentizable_types", [])
            if item in valid_entity_set
        ]
        if not agentizable_types:
            agentizable_types = [
                item for item in schema_context.get("agentizable_types", [])
                if item in valid_entity_set
            ]
        if not agentizable_types:
            agentizable_types = valid_entity_names[: min(2, len(valid_entity_names))]

        non_agentizable_types = [
            item for item in result.get("non_agentizable_types", [])
            if item in valid_entity_set and item not in agentizable_types
        ]
        if not non_agentizable_types:
            non_agentizable_types = [
                item for item in valid_entity_names
                if item not in agentizable_types
            ]

        simulation_grammar: Dict[str, List[str]] = {}
        for entity_type in agentizable_types:
            actions = result.get("simulation_grammar", {}).get(entity_type)
            if not actions:
                actions = schema_context.get("simulation_grammar", {}).get(entity_type, [])
            normalized_actions: List[str] = []
            for action in actions[:12]:
                text = str(action).strip()
                if text and text not in normalized_actions:
                    normalized_actions.append(text)
            simulation_grammar[entity_type] = normalized_actions

        result["genre"] = schema_context["genre"]
        result["schema_overlays"] = schema_context.get("schema_overlays", [])
        result["genre_candidates"] = schema_context.get("genre_candidates", [])
        result["genre_inference_reasoning"] = schema_context.get("genre_inference_reasoning", "")
        result["agentizable_types"] = agentizable_types
        result["non_agentizable_types"] = non_agentizable_types
        result["simulation_grammar"] = simulation_grammar
        result["report_template"] = self._normalize_report_template(
            result.get("report_template"),
            schema_context.get("report_template", {}).get("sections", []),
        )
        
        return result
    
    def generate_python_code(self, ontology: Dict[str, Any]) -> str:
        """
        将本体定义转换为Python代码（类似ontology.py）
        
        Args:
            ontology: 本体定义
            
        Returns:
            Python代码字符串
        """
        code_lines = [
            '"""',
            '自定义实体类型定义',
            '由MiroFish自动生成，用于社会舆论模拟',
            '"""',
            '',
            'from pydantic import Field',
            'from zep_cloud.external_clients.ontology import EntityModel, EntityText, EdgeModel',
            '',
            '',
            '# ============== 实体类型定义 ==============',
            '',
        ]
        
        # 生成实体类型
        for entity in ontology.get("entity_types", []):
            name = entity["name"]
            desc = entity.get("description", f"A {name} entity.")
            
            code_lines.append(f'class {name}(EntityModel):')
            code_lines.append(f'    """{desc}"""')
            
            attrs = entity.get("attributes", [])
            if attrs:
                for attr in attrs:
                    attr_name = attr["name"]
                    attr_desc = attr.get("description", attr_name)
                    code_lines.append(f'    {attr_name}: EntityText = Field(')
                    code_lines.append(f'        description="{attr_desc}",')
                    code_lines.append(f'        default=None')
                    code_lines.append(f'    )')
            else:
                code_lines.append('    pass')
            
            code_lines.append('')
            code_lines.append('')
        
        code_lines.append('# ============== 关系类型定义 ==============')
        code_lines.append('')
        
        # 生成关系类型
        for edge in ontology.get("edge_types", []):
            name = edge["name"]
            # 转换为PascalCase类名
            class_name = ''.join(word.capitalize() for word in name.split('_'))
            desc = edge.get("description", f"A {name} relationship.")
            
            code_lines.append(f'class {class_name}(EdgeModel):')
            code_lines.append(f'    """{desc}"""')
            
            attrs = edge.get("attributes", [])
            if attrs:
                for attr in attrs:
                    attr_name = attr["name"]
                    attr_desc = attr.get("description", attr_name)
                    code_lines.append(f'    {attr_name}: EntityText = Field(')
                    code_lines.append(f'        description="{attr_desc}",')
                    code_lines.append(f'        default=None')
                    code_lines.append(f'    )')
            else:
                code_lines.append('    pass')
            
            code_lines.append('')
            code_lines.append('')
        
        # 生成类型字典
        code_lines.append('# ============== 类型配置 ==============')
        code_lines.append('')
        code_lines.append('ENTITY_TYPES = {')
        for entity in ontology.get("entity_types", []):
            name = entity["name"]
            code_lines.append(f'    "{name}": {name},')
        code_lines.append('}')
        code_lines.append('')
        code_lines.append('EDGE_TYPES = {')
        for edge in ontology.get("edge_types", []):
            name = edge["name"]
            class_name = ''.join(word.capitalize() for word in name.split('_'))
            code_lines.append(f'    "{name}": {class_name},')
        code_lines.append('}')
        code_lines.append('')
        
        # 生成边的source_targets映射
        code_lines.append('EDGE_SOURCE_TARGETS = {')
        for edge in ontology.get("edge_types", []):
            name = edge["name"]
            source_targets = edge.get("source_targets", [])
            if source_targets:
                st_list = ', '.join([
                    f'{{"source": "{st.get("source", "Entity")}", "target": "{st.get("target", "Entity")}"}}'
                    for st in source_targets
                ])
                code_lines.append(f'    "{name}": [{st_list}],')
        code_lines.append('}')
        
        return '\n'.join(code_lines)

