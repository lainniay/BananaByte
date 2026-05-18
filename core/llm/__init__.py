import logging

from core.llm.gemini_llm import GeminiLLM
from core.llm.openai_llm import OpenAILLM
from core.llm.tool import Tool, tool

__all__ = ["GeminiLLM", "OpenAILLM", "Tool", "setup_rich_logging", "tool"]


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
