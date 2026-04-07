"""
Core module for LLM framework

提供统一的 LLM 接口和消息模型
"""

from core.llm import BaseLLM, OpenAILLM, GeminiLLM, create_llm
from core.schemas import Message, TextContent, ImageContent

__version__ = "0.1.0"

__all__ = [
    # LLM 类
    "BaseLLM",
    "OpenAILLM",
    "GeminiLLM",
    # 工厂函数
    "create_llm",
    # 消息模型
    "Message",
    "TextContent",
    "ImageContent",
]
