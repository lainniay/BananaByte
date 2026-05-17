import base64
from typing import Any

from core.llm.tool import Tool
from core.schemas import Message


def _remove_key(key: str, obj: dict[str, Any]) -> None:
    """递归移除 JSON Schema 中指定的键.

    Args:
        key: 需要移除的键名.
        obj: 需要原地修改的字典对象.
    """
    obj.pop(key, None)
    for item in obj.values():
        if isinstance(item, dict):
            _remove_key(key, item)
        elif isinstance(item, list):
            for element in item:
                if isinstance(element, dict):
                    _remove_key(key, element)


def message_to_openai(message: Message) -> dict[str, Any]:
    """将内部消息转换为 OpenAI API 格式.

    Args:
        message: 内部消息对象.

    Returns:
        OpenAI API 格式的消息字典, 包含 ``role`` 和 ``content`` 字段.
        角色 ``model`` 会被转换为 ``assistant``.
    """
    role = "assistant" if message.role == "model" else message.role
    if isinstance(message.content, str):
        return {"role": role, "content": message.content}

    content: list[dict[str, Any]] = []
    for item in message.content:
        if item.type == "text":
            content.append({"type": "text", "text": item.text})
        elif item.type == "image":
            content.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{item.mime_type};base64,{base64.b64encode(item.source).decode('utf-8')}"
                    },
                }
            )
    return {"role": role, "content": content}


def message_to_gemini(message: Message) -> dict[str, Any]:
    """将内部消息转换为 Gemini API 格式.

    Args:
        message: 内部消息对象.

    Returns:
        Gemini API 格式的消息字典, 包含 ``role`` 和 ``parts`` 字段.
    """
    if isinstance(message.content, str):
        return {"role": message.role, "parts": [{"text": message.content}]}

    parts: list[dict[str, Any]] = []
    for item in message.content:
        if item.type == "text":
            parts.append({"text": item.text})
        elif item.type == "image":
            parts.append(
                {"inline_data": {"mime_type": item.mime_type, "data": item.source}}
            )
    return {"role": message.role, "parts": parts}


def tool_to_openai(tool: Tool[Any]) -> dict[str, Any]:
    """将内部工具转换为 OpenAI function tool schema.

    Args:
        tool: 内部工具对象.

    Returns:
        OpenAI Chat Completions API 接受的 tool 定义字典.
    """
    schema = tool.parameters.model_json_schema()
    _remove_key("title", schema)
    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description,
            "parameters": schema,
            "strict": True,
        },
    }


def tool_to_gemini(tool: Tool[Any]) -> dict[str, Any]:
    """将内部工具转换为 Gemini function declaration schema.

    Args:
        tool: 内部工具对象.

    Returns:
        Gemini FunctionDeclaration 可使用的函数字典.
    """
    schema = tool.parameters.model_json_schema()
    _remove_key("title", schema)
    return {"name": tool.name, "description": tool.description, "parameters": schema}
