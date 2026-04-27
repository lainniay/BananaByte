"""Core module for LLM framework.

提供统一的 LLM 接口、消息模型和日志工具
"""

import logging

from dotenv import load_dotenv

from core.llm import GeminiLLM, OpenAILLM
from core.prompt import Prompt, PromptLib
from core.schemas import ImageContent, Message, TextContent

load_dotenv()


def setup_rich_logging(level: int = logging.WARNING) -> None:
    """为 core.llm 配置 RichHandler, 以 rich 风格输出重试/错误日志.

    Args:
        level: 日志级别, 默认 WARNING(仅显示重试和错误信息).
    """
    from rich.logging import RichHandler

    handler = RichHandler(rich_tracebacks=True, show_path=False)
    handler.setLevel(level)

    llm_logger = logging.getLogger("core.llm")
    llm_logger.setLevel(level)
    llm_logger.addHandler(handler)
    llm_logger.propagate = False


__version__ = "0.1.0"

__all__ = [
    "GeminiLLM",
    "ImageContent",
    "Message",
    "OpenAILLM",
    "Prompt",
    "PromptLib",
    "TextContent",
    "setup_rich_logging",
]
