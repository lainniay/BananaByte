"""Core module for LLM framework.

提供统一的 LLM 接口和消息模型
"""

from core.llm import BaseLLM, GeminiLLM, OpenAILLM, create_llm
from core.schemas import ImageContent, Message, TextContent

__version__ = "0.1.0"

__all__ = [
    # LLM 类
    "BaseLLM",
    "GeminiLLM",
    "ImageContent",
    # 消息模型
    "Message",
    "OpenAILLM",
    "TextContent",
    # 工厂函数
    "create_llm",
]
