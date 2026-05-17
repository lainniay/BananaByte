import base64
import json
import os
from collections.abc import Callable
from typing import Annotated, Any, Literal, cast

from openai import (
    APIConnectionError,
    APITimeoutError,
    InternalServerError,
    OpenAI,
    RateLimitError,
)
from openai.types.chat import ChatCompletionMessageParam
from pydantic import BaseModel, Field

from core.llm.adapters import (
    message_to_openai,
    tool_to_openai,
)
from core.llm.common import _call_with_retry, _ensure_list
from core.llm.tool import Tool
from core.schemas import ImageContent, Message, TextContent

_OPENAI_RETRYABLE = (
    RateLimitError,
    InternalServerError,
    APITimeoutError,
    APIConnectionError,
)


class OpenAILLM:
    """OpenAI 兼容的 LLM 实现.

    支持 OpenAI API 及兼容的第三方 API (如 Azure OpenAI, Anthropic 等).

    Attributes:
        model: 模型名称.
        client: OpenAI 客户端实例.
    """

    class GenerateConfig(BaseModel):
        """OpenAI 生成参数配置.

        Attributes:
            timeout: 超时时间 (秒).
            temperature: 采样温度, 范围 0.0 到 2.0.
            top_p: 核采样阈值, 范围 0.0 到 1.0.
            frequency_penalty: 频率惩罚, 范围 -2.0 到 2.0.
            presence_penalty: 存在惩罚, 范围 -2.0 到 2.0.
            n: 每个 prompt 生成的回复数量.
            seed: 随机种子, 用于实现确定性输出.
            prediction: 预期输出内容, 用于加速生成.
            verbosity: 响应的详细程度.
            parallel_tool_calls: 是否允许并行工具调用.
            max_completion_tokens: 模型生成的最大 token 数量.
        """

        timeout: float | None = None  # NOTE:单位是秒
        temperature: Annotated[float, Field(ge=0.0, le=2.0)] = 1.0
        top_p: Annotated[float | None, Field(ge=0.0, le=1.0)] = None
        frequency_penalty: Annotated[float | None, Field(ge=-2.0, le=2.0)] = None
        presence_penalty: Annotated[float | None, Field(ge=-2.0, le=2.0)] = None
        n: Annotated[int, Field(ge=1)] = 1
        seed: int | None = None
        prediction: str | None = None
        verbosity: Literal["low", "medium", "high"] | None = None
        parallel_tool_calls: bool | None = None
        max_completion_tokens: Annotated[int, Field(ge=1)] | None = None

    class EditImageConfig(BaseModel):
        """OpenAI 图像生成/编辑参数.

        Attributes:
            background: 背景处理方式.
            input_fidelity: 输入保真度.
            mask: 蒙版图像, 用于局部重绘 (Inpainting).
            n: 生成图像的数量.
            output_compression: 输出压缩率.
            output_format: 输出图像格式.
            seed: 随机种子.
            size: 图像分辨率, 如 "1024x1024".
            quality: 图像质量级别.
            user: 代表终端用户的唯一标识符.
        """

        background: Literal["transparent", "opaque", "auto"] | None = None
        input_fidelity: Literal["high", "low"] | None = None
        mask: ImageContent | None = None
        n: Annotated[int, Field(ge=1, le=10)] | None = None
        output_compression: Annotated[int, Field(ge=0, le=100)] | None = None
        output_format: Literal["png", "jpeg", "webp"] | None = None
        seed: int | None = None
        # response_format: Literal["url", "b64_json"] | None = None
        size: str | None = None
        quality: Literal["standard", "low", "medium", "high", "auto"] | None = None

    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float = 120,
        max_retries: int = 3,
    ) -> None:
        """初始化 OpenAI LLM 客户端.

        Args:
            model: 模型名称.
            api_key: API 密钥. 如果为 None, 从环境变量 OPENAI_API_KEY 读取.
            base_url: API 基础 URL. 如果为 None, 从环境变量 OPENAI_API_BASE 读取.
            timeout: 请求超时时间, 单位是秒.
            max_retries: 最大重试次数, 默认 3.

        Raises:
            ValueError: 当 API 密钥缺失时.
        """
        super().__init__()
        self.model = model
        self.max_retries = max_retries
        api_key = api_key or os.getenv("OPENAI_API_KEY", "")
        base_url = base_url or os.getenv("OPENAI_API_BASE", "")
        if not api_key:
            raise ValueError("OPENAI_API_KEY 环境变量缺失")
        self.client = OpenAI(api_key=api_key, base_url=base_url, timeout=timeout)

    _RETRYABLE = _OPENAI_RETRYABLE

    def _retry[T](self, fn: Callable[[], T]) -> T:
        """带重试的 API 调用包裹器."""
        return _call_with_retry(self.max_retries, self._RETRYABLE, fn)

    def _prepare_messages(
        self,
        messages: Message | list[Message],
        system_prompt: str | None = None,
    ) -> list:
        """规范化消息列表并可选插入 system prompt.

        Args:
            messages: 单个消息或消息列表.
            system_prompt: 系统指令.

        Returns:
            list: OpenAI 格式的消息列表.
        """
        messages = _ensure_list(messages)
        openai_mes = [message_to_openai(msg) for msg in messages]
        if system_prompt:
            openai_mes.insert(0, {"role": "system", "content": system_prompt})
        return openai_mes

    def generate(
        self,
        messages: Message | list[Message],
        system_prompt: str | None = None,
        config: GenerateConfig | None = None,
    ) -> Message:
        """使用 OpenAI API 生成响应.

        Args:
            messages: 单个消息或消息列表, 包含对话历史.
            system_prompt: 系统指令, 用于设定 LLM 的行为模式.
            config: 生成参数配置, 用于控制采样和工具调用.

        Returns:
            Message: LLM 生成的响应消息, 角色为 "model".
        """
        openai_mes = self._prepare_messages(messages, system_prompt)
        kwargs = self.__translate_generate(config) if config else {}
        res = self._retry(
            lambda: self.client.chat.completions.create(
                messages=cast(list[ChatCompletionMessageParam], openai_mes),
                model=self.model,
                **kwargs,
            ),
        )
        content = res.choices[0].message.content or ""
        return Message(role="model", content=content)

    def generate_with_tool(
        self,
        messages: Message | list[Message],
        tools: list[Tool[Any]],
        system_prompt: str | None = None,
        config: GenerateConfig | None = None,
        max_tool_rounds: int = 5,
    ) -> Message:
        """使用 OpenAI API 生成响应, 并自动执行模型请求的工具调用.

        Args:
            messages: 单个消息或消息列表, 包含对话历史.
            tools: 模型可调用的工具列表.
            system_prompt: 系统指令, 用于设定 LLM 的行为模式.
            config: 生成参数配置.
            max_tool_rounds: 最大工具调用轮数, 防止无限循环.

        Returns:
            Message: LLM 生成的最终响应消息, 角色为 "model".

        Raises:
            ValueError: 当工具列表为空、工具名重复或最大轮数无效时.
            RuntimeError: 当超过最大工具调用轮数时.
        """
        openai_mes, kwargs, tool_map = self.__prepare_tool_loop(
            messages,
            tools,
            system_prompt,
            config,
            max_tool_rounds,
        )

        for _ in range(max_tool_rounds):
            res = self._retry(
                lambda: self.client.chat.completions.create(
                    messages=cast(list[ChatCompletionMessageParam], openai_mes),
                    model=self.model,
                    **kwargs,
                ),
            )
            msg = res.choices[0].message
            if not msg.tool_calls:
                return Message(role="model", content=msg.content or "")

            self.__append_tool_results(openai_mes, msg, tool_map)

        raise RuntimeError("超过最大工具调用轮数")

    def generate_struct(
        self,
        messages: Message | list[Message],
        schema: type[BaseModel],
        system_prompt: str | None = None,
        config: GenerateConfig | None = None,
    ) -> Message:
        """使用 OpenAI API 生成符合 Pydantic 模型的结构化响应.

        Args:
            messages: 单个消息或消息列表, 包含对话历史.
            schema: Pydantic BaseModel 子类, 定义期望的输出结构.
            system_prompt: 系统指令, 用于设定 LLM 的行为模式.
            config: 生成参数配置.

        Returns:
            Message: LLM 生成的响应消息, content 为 JSON 字符串.

        Raises:
            ValueError: 当模型拒绝生成或无法解析输出时.
        """
        openai_mes = self._prepare_messages(messages, system_prompt)
        kwargs = self.__translate_generate(config) if config else {}
        res = self._retry(
            lambda: self.client.chat.completions.parse(
                messages=cast(list[ChatCompletionMessageParam], openai_mes),
                model=self.model,
                response_format=schema,
                **kwargs,
            ),
        )

        message = res.choices[0].message
        if message.refusal:
            raise ValueError(f"模型拒绝生成结构化输出: {message.refusal}")
        if message.parsed is None:
            raise ValueError("模型未返回可解析的结构化输出")
        return Message(role="model", content=message.parsed.model_dump_json())

    def generate_struct_with_tool(
        self,
        messages: Message | list[Message],
        schema: type[BaseModel],
        tools: list[Tool[Any]],
        system_prompt: str | None = None,
        config: GenerateConfig | None = None,
        max_tool_rounds: int = 5,
    ) -> Message:
        """使用 OpenAI API 生成结构化响应, 并自动执行工具调用.

        Args:
            messages: 单个消息或消息列表, 包含对话历史.
            schema: Pydantic BaseModel 子类, 定义期望的输出结构.
            tools: 模型可调用的工具列表.
            system_prompt: 系统指令, 用于设定 LLM 的行为模式.
            config: 生成参数配置.
            max_tool_rounds: 最大工具调用轮数, 防止无限循环.

        Returns:
            Message: LLM 生成的最终响应消息, content 为 JSON 字符串.

        Raises:
            ValueError: 当工具列表为空、工具名重复、最大轮数无效、模型拒绝生成或无法解析输出时.
            RuntimeError: 当超过最大工具调用轮数时.
        """
        openai_mes, kwargs, tool_map = self.__prepare_tool_loop(
            messages,
            tools,
            system_prompt,
            config,
            max_tool_rounds,
        )

        for _ in range(max_tool_rounds):
            res = self._retry(
                lambda: self.client.chat.completions.parse(
                    messages=cast(list[ChatCompletionMessageParam], openai_mes),
                    model=self.model,
                    response_format=schema,
                    **kwargs,
                ),
            )
            msg = res.choices[0].message
            if not msg.tool_calls:
                if msg.refusal:
                    raise ValueError(f"模型拒绝生成结构化输出: {msg.refusal}")
                if msg.parsed is None:
                    raise ValueError("模型未返回可解析的结构化输出")
                return Message(role="model", content=msg.parsed.model_dump_json())

            self.__append_tool_results(openai_mes, msg, tool_map)

        raise RuntimeError("超过最大工具调用轮数")

    def __prepare_tool_loop(
        self,
        messages: Message | list[Message],
        tools: list[Tool[Any]],
        system_prompt: str | None,
        config: GenerateConfig | None,
        max_tool_rounds: int,
    ) -> tuple[list, dict[str, Any], dict[str, Tool[Any]]]:
        """准备 OpenAI 工具调用循环所需的消息、参数和工具映射."""
        tool_map = self.__validate_tool_loop_inputs(tools, max_tool_rounds)
        openai_mes = self._prepare_messages(messages, system_prompt)
        kwargs = self.__translate_generate(config) if config else {}
        kwargs["tools"] = [tool_to_openai(tool) for tool in tools]
        return openai_mes, kwargs, tool_map

    @staticmethod
    def __validate_tool_loop_inputs(
        tools: list[Tool[Any]],
        max_tool_rounds: int,
    ) -> dict[str, Tool[Any]]:
        """校验工具调用循环参数并返回工具映射."""
        if not tools:
            raise ValueError("工具列表不能为空")
        if max_tool_rounds < 1:
            raise ValueError("max_tool_rounds 必须大于等于 1")

        tool_map = {tool.name: tool for tool in tools}
        if len(tool_map) != len(tools):
            raise ValueError("工具名称不能重复")
        return tool_map

    @staticmethod
    def __append_tool_results(
        openai_mes: list,
        msg: Any,
        tool_map: dict[str, Tool[Any]],
    ) -> None:
        """追加 OpenAI 模型工具调用轮次和对应工具返回值."""
        openai_mes.append(
            {
                "role": "assistant",
                "content": msg.content,
                "tool_calls": [
                    tool_call.model_dump(exclude_none=True)
                    for tool_call in msg.tool_calls
                ],
            }
        )

        for tool_call in msg.tool_calls:
            tool_name = tool_call.function.name
            try:
                tool_args = json.loads(tool_call.function.arguments or "{}")
                if not isinstance(tool_args, dict):
                    raise TypeError("工具参数必须是 JSON object")
                tool = tool_map.get(tool_name)
                if tool is None:
                    raise ValueError(f"未知工具: {tool_name}")
                tool_result = {"is_error": False, "result": tool(**tool_args)}
            except Exception as e:
                tool_result = {"is_error": True, "error": str(e)}

            openai_mes.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": json.dumps(
                        tool_result,
                        ensure_ascii=False,
                        default=str,
                    ),
                }
            )

    def edit_image(
        self,
        messages: Message | list[Message],
        system_prompt: str | None = None,
        config: EditImageConfig | None = None,
    ) -> Message:
        """使用 OpenAI Images API 编辑图像.

        Args:
            messages: 单个消息或消息列表, 必须包含至少一张图像. 文本内容会合并为 prompt.
            system_prompt: 可选的系统指令, 会拼接到 prompt 前面.
            config: 图像编辑参数配置, 对应 OpenAI images.edit 参数.

        Returns:
            Message: LLM 生成的响应消息. Base64 图像会转为 ImageContent;
                如果 API 返回 URL, 则以文本形式返回 URL 列表.

        Raises:
            ValueError: 当没有传入图片、prompt 为空或 API 未返回结果时.
        """
        messages = _ensure_list(messages)

        images: list[ImageContent] = []
        prompt_parts: list[str] = []
        if system_prompt:
            prompt_parts.append(system_prompt)

        for message in messages:
            if message.text:
                prompt_parts.append(message.text)
            images.extend(message.images)

        if not images:
            raise ValueError("没有传入图片")

        prompt = "\n".join(prompt_parts).strip()
        if not prompt:
            raise ValueError("没有传入图像编辑提示词")

        kwargs = self.__translate_edit_image(config) if config else {}
        res = self._retry(
            lambda: self.client.images.edit(
                image=[
                    self.__to_openai_file(image, f"image_{index}")
                    for index, image in enumerate(images)
                ],
                prompt=prompt,
                model=self.model,
                **kwargs,
            ),
        )

        if not res.data:
            raise ValueError("OpenAI 未返回图像编辑结果")

        output_format = kwargs.get("output_format") or "png"
        contents: list[TextContent | ImageContent] = []
        urls: list[str] = []
        for item in res.data:
            if item.b64_json:
                contents.append(
                    ImageContent(
                        source=base64.b64decode(item.b64_json),
                        mime_type=f"image/{output_format}",
                    )
                )
            elif item.url:
                urls.append(item.url)

        if contents:
            return Message(role="model", content=contents)
        if urls:
            return Message(role="model", content="\n".join(urls))
        raise ValueError("OpenAI 未返回可用的图像数据")

    def __translate_generate(self, input: GenerateConfig) -> dict[str, Any]:
        """将内部 GenerateConfig 转换为 OpenAI API 兼容的参数字典.

        Args:
            input: 内部生成配置对象.

        Returns:
            dict[str, Any]: 转换后的参数字典.
        """
        kwargs = input.model_dump(exclude_none=True)

        if "prediction" in kwargs:
            pred_str = kwargs.pop("prediction")
            kwargs["prediction"] = {"type": "content", "content": pred_str}

        return kwargs

    def __translate_edit_image(self, input: EditImageConfig) -> dict[str, Any]:
        """将内部 EditImageConfig 转换为 OpenAI API 兼容的参数字典.

        Args:
            input: 内部图像编辑配置对象.

        Returns:
            dict[str, Any]: 转换后的参数字典.
        """
        kwargs = input.model_dump(exclude_none=True)

        mask = kwargs.pop("mask", None)
        if mask is not None:
            kwargs["mask"] = self.__to_openai_file(mask, "mask")

        return kwargs

    @staticmethod
    def __to_openai_file(image: ImageContent, name: str) -> tuple[str, bytes, str]:
        """将 ImageContent 转换为 OpenAI API 要求的元组格式.

        Args:
            image: 图像内容对象.
            name: 文件基本名称.

        Returns:
            tuple[str, bytes, str]: 包含 (文件名, 内容, MIME 类型) 的元组.
        """
        extension = image.mime_type.removeprefix("image/").replace("jpeg", "jpg")
        filename = f"{name}.{extension or 'png'}"
        return filename, image.source, image.mime_type
