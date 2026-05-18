"""Core module for LLM framework.

提供统一的 LLM 接口、消息模型和日志工具
"""

from dotenv import load_dotenv

from core.llm import GeminiLLM, OpenAILLM, setup_rich_logging, tool
from core.prompt import Prompt, PromptLib
from core.schemas import ImageContent, Message, TextContent
from core.state import BaseState

load_dotenv()

__version__ = "0.1.0"

__all__ = [
    "BaseState",
    "GeminiLLM",
    "ImageContent",
    "Message",
    "OpenAILLM",
    "Prompt",
    "PromptLib",
    "TextContent",
    "setup_rich_logging",
    "tool",
]
