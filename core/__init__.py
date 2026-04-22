"""Core module for LLM framework.

提供统一的 LLM 接口和消息模型
"""

from core.llm import GeminiLLM, OpenAILLM
from core.prompt import Prompt, PromptLib
from core.schemas import ImageContent, Message, TextContent

__version__ = "0.1.0"

__all__ = [
    "GeminiLLM",
    "ImageContent",
    "Message",
    "OpenAILLM",
    "Prompt",
    "PromptLib",
    "TextContent",
]
