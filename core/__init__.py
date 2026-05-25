"""Core module for LLM framework.

提供统一的 LLM 接口、消息模型和日志工具
"""

from dotenv import load_dotenv

from core.llm import (
    GeminiLLM as GeminiLLM,
    OpenAILLM as OpenAILLM,
    Tool as Tool,
    setup_rich_logging as setup_rich_logging,
    tool as tool,
)
from core.prompt import Prompt as Prompt, PromptLib as PromptLib
from core.schemas import (
    ImageContent as ImageContent,
    Message as Message,
    TextContent as TextContent,
)
from core.state import BaseState as BaseState

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
    "Tool",
    "setup_rich_logging",
    "tool",
]
