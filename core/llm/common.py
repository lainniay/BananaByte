import logging
from collections.abc import Callable

from tenacity import (
    Retrying,
    before_sleep_log,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

from core.schemas import Message

logger = logging.getLogger("core.llm")


def _ensure_list(messages: Message | list[Message]) -> list[Message]:
    """将单个消息或消息列表规范化为列表."""
    return messages if isinstance(messages, list) else [messages]


def _call_with_retry[T](
    max_retries: int,
    retryable: tuple,
    fn: Callable[[], T],
) -> T:
    """带指数退避重试的 API 调用包裹器.

    Args:
        max_retries: 最大重试次数.
        retryable: 可重试的异常类型元组.
        fn: 无参可调用对象, 返回 API 响应.

    Returns:
        API 调用结果.
    """
    for attempt in Retrying(
        stop=stop_after_attempt(max_retries),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        retry=retry_if_exception(lambda e: isinstance(e, retryable)),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True,
    ):
        with attempt:
            return fn()
    raise RuntimeError("unreachable")  # pragma: no cover
