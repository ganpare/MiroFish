"""
LLM客户端封装
统一使用OpenAI格式调用
"""

import json
import re
from typing import Optional, Dict, Any, List
from urllib.parse import urlsplit, urlunsplit
from openai import OpenAI

from ..config import Config


class LLMClient:
    """LLM客户端"""

    @staticmethod
    def _normalize_base_url(url: str) -> str:
        """Normalize OpenAI-compatible base_url without breaking versioned custom paths."""
        if not url:
            return url

        normalized = url.strip().rstrip("/")
        parsed = urlsplit(normalized)
        path_segments = [segment for segment in parsed.path.split("/") if segment]
        if any(re.fullmatch(r"v\d+", segment, flags=re.IGNORECASE) for segment in path_segments):
            return normalized

        new_path = f"{parsed.path.rstrip('/')}/v1"
        return urlunsplit((parsed.scheme, parsed.netloc, new_path, parsed.query, parsed.fragment))

    @staticmethod
    def _extract_json_object(text: str) -> Optional[str]:
        if not text:
            return None
        stripped = text.strip()
        if (stripped.startswith("{") and stripped.endswith("}")) or (
            stripped.startswith("[") and stripped.endswith("]")
        ):
            return stripped

        obj_start = stripped.find("{")
        obj_end = stripped.rfind("}")
        if obj_start != -1 and obj_end != -1 and obj_end > obj_start:
            return stripped[obj_start : obj_end + 1]

        arr_start = stripped.find("[")
        arr_end = stripped.rfind("]")
        if arr_start != -1 and arr_end != -1 and arr_end > arr_start:
            return stripped[arr_start : arr_end + 1]
        return None

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None
    ):
        self.api_key = api_key or Config.LLM_API_KEY
        self.base_url = self._normalize_base_url(base_url or Config.LLM_BASE_URL)
        self.model = model or Config.LLM_MODEL_NAME
        
        if not self.api_key:
            raise ValueError("LLM_API_KEY 未配置")
        
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url
        )
        self._embedding_client = OpenAI(
            api_key=Config.EMBEDDING_API_KEY,
            base_url=self._normalize_base_url(Config.EMBEDDING_BASE_URL),
        )
    
    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 4096,
        response_format: Optional[Dict] = None
    ) -> str:
        """
        发送聊天请求
        
        Args:
            messages: 消息列表
            temperature: 温度参数
            max_tokens: 最大token数
            response_format: 响应格式（如JSON模式）
            
        Returns:
            模型响应文本
        """
        kwargs = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        
        if response_format:
            kwargs["response_format"] = response_format
        
        response = self.client.chat.completions.create(**kwargs)
        content = response.choices[0].message.content
        # 部分模型（如MiniMax M2.5）会在content中包含<think>思考内容，需要移除
        content = re.sub(r'<think>[\s\S]*?</think>', '', content).strip()
        return content
    
    def chat_json(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.3,
        max_tokens: int = 4096
    ) -> Dict[str, Any]:
        """
        发送聊天请求并返回JSON
        
        Args:
            messages: 消息列表
            temperature: 温度参数
            max_tokens: 最大token数
            
        Returns:
            解析后的JSON对象
        """
        try:
            response = self.chat(
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                response_format={"type": "json_object"}
            )
            cleaned_response = response.strip()
            cleaned_response = re.sub(r'^```(?:json)?\s*\n?', '', cleaned_response, flags=re.IGNORECASE)
            cleaned_response = re.sub(r'\n?```\s*$', '', cleaned_response)
            cleaned_response = cleaned_response.strip()
            parsed = json.loads(cleaned_response)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            pass

        fallback_messages = [
            {
                "role": "system",
                "content": (
                    "你必须只输出一个合法的 JSON 对象（以 { 开始，以 } 结束）。"
                    "不要输出任何解释、前后缀、代码块标记。"
                ),
            },
            *messages,
        ]
        fallback_response = self.chat(
            messages=fallback_messages,
            temperature=max(0.0, temperature),
            max_tokens=max_tokens,
            response_format=None,
        )
        json_text = self._extract_json_object(fallback_response)
        if not json_text:
            raise ValueError(f"模型未返回可解析的JSON对象: {fallback_response[:200]}")

        parsed = json.loads(json_text)
        if not isinstance(parsed, dict):
            raise ValueError(f"JSON解析结果不是对象: {type(parsed).__name__}")
        return parsed

    def embed_texts(self, texts: List[str], model: Optional[str] = None) -> List[List[float]]:
        """生成 embeddings（用于向量库）"""
        embedding_model = model or Config.EMBEDDING_MODEL_NAME
        response = self._embedding_client.embeddings.create(
            model=embedding_model,
            input=texts,
        )
        return [item.embedding for item in response.data]

