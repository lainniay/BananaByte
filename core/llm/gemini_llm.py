import base64
import os
from collections.abc import Callable
from typing import Annotated, Any, Literal, cast

from google import genai
from google.genai.errors import ServerError
from google.genai.types import (
    AutomaticFunctionCallingConfig,
    Content,
    ContentListUnionDict,
    FunctionDeclaration,
    HttpOptions,
    Part,
    Tool as GeminiTool,
)
from pydantic import BaseModel, Field

from core.llm.adapters import (
    message_to_gemini,
    tool_to_gemini,
)
from core.llm.common import _call_with_retry, _ensure_list
from core.llm.tool import Tool
from core.schemas import ImageContent, Message, TextContent

_GEMINI_RETRYABLE = (ServerError,)


class GeminiLLM:
    """Google Gemini LLM 实现.

    支持 Google Gemini API, 包括文本生成和图像编辑功能.

    Attributes:
        model: 模型名称.
        client: Gemini API 客户端实例.
        api_key: API 密钥.
        base_url: API 基础 URL.
    """

    class GenerateConfig(BaseModel):
        """Gemini 生成参数配置.

        对应 genai.types.GenerateContentConfig 中面向用户的字段.
        system_instruction 不在此处配置, 通过 generate() 的 system_prompt 参数传入.
        response_schema 不在此处配置, 通过 generate_struct() 的 schema 参数传入.

        Attributes:
            temperature: 采样温度, 范围 0.0 到 2.0.
            top_p: 核采样阈值, 范围 0.0 到 1.0.
            top_k: Top-K 采样参数.
            max_output_tokens: 最大生成 token 数量.
            candidate_count: 生成候选回复的数量.
            stop_sequences: 停止序列列表.
            frequency_penalty: 频率惩罚.
            presence_penalty: 存在惩罚.
            seed: 随机种子.
            think_level: 思考深度等级 (针对特定模型).
        """

        # 采样参数
        temperature: Annotated[float, Field(ge=0.0, le=2.0)] = 1.0
        top_p: Annotated[float | None, Field(ge=0.0, le=1.0)] = None
        top_k: Annotated[float | None, Field(ge=0.0)] = None
        max_output_tokens: Annotated[int | None, Field(ge=1)] = None
        candidate_count: Annotated[int | None, Field(ge=1)] = None
        stop_sequences: list[str] | None = None
        frequency_penalty: float | None = None
        presence_penalty: float | None = None
        seed: int | None = None

        think_level: Literal["HIGH", "LOW", "MEDIUM", "MINIMAL"] | None = None

    class EditImageConfig(BaseModel):
        """Gemini 图像编辑/生成参数配置.

        对应 genai.types.GenerateContentConfig 与 genai.types.ImageConfig 中
        与图像生成相关的字段.

        Attributes:
            temperature: 采样温度.
            seed: 随机种子.
            think_level: 思考深度等级.
            aspect_ratio: 输出图像宽高比, 如 "1:1", "16:9", "9:16" 等.
            image_size: 输出图像分辨率, 可选 "1K", "2K", "4K", 默认 "1K".
            output_mime_type: 输出图像 MIME 类型, 如 "image/png", "image/jpeg".
            output_compression_quality: 输出图像压缩质量 (0-100), 仅 JPEG/WebP 有效.
        """

        # 生成控制 (与 GenerateConfig 对齐的子集)
        temperature: Annotated[float, Field(ge=0.0, le=2.0)] = 1.0
        seed: int | None = None
        think_level: Literal["HIGH", "LOW", "MEDIUM", "MINIMAL"] | None = None

        # ImageConfig 字段
        aspect_ratio: str | None = None
        image_size: str | None = None
        output_mime_type: str | None = None
        output_compression_quality: Annotated[int | None, Field(ge=0, le=100)] = None

    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: int = 120 * 1000,
        max_retries: int = 3,
    ) -> None:
        """初始化 Gemini LLM 客户端.

        Args:
            model: 模型名称.
            api_key: API 密钥. 如果为 None, 从环境变量 GEMINI_API_KEY 读取.
            base_url: API 基础 URL. 如果为 None, 从环境变量 GEMINI_API_BASE 读取.
            timeout: 请求超时时间, 单位是毫秒.
            max_retries: 最大重试次数, 默认 3.

        Raises:
            ValueError: 当 API 密钥缺失时.
        """
        super().__init__()
        self.model = model
        self.max_retries = max_retries
        api_key = api_key or os.getenv("GEMINI_API_KEY", "")
        base_url = base_url or os.getenv("GEMINI_API_BASE", "")
        if not api_key:
            raise ValueError("GEMINI_API_KEY 环境变量缺失")
        self.client = genai.Client(
            api_key=api_key,
            http_options=HttpOptions(base_url=base_url, timeout=timeout),
        )

    _RETRYABLE = _GEMINI_RETRYABLE

    def _retry[T](self, fn: Callable[[], T]) -> T:
        """带重试的 API 调用包裹器."""
        return _call_with_retry(self.max_retries, self._RETRYABLE, fn)

    def _normalize_messages(
        self, messages: Message | list[Message]
    ) -> ContentListUnionDict:
        """规范化消息为 Gemini 格式.

        Args:
            messages: 单个消息或消息列表.

        Returns:
            ContentListUnionDict: Gemini 格式的消息列表.
        """
        messages = _ensure_list(messages)
        return cast(
            ContentListUnionDict,
            [message_to_gemini(msg) for msg in messages],
        )

    def generate(
        self,
        messages: Message | list[Message],
        system_prompt: str | None = None,
        config: GenerateConfig | None = None,
    ) -> Message:
        """使用 Gemini API 生成响应.

        Args:
            messages: 单个消息或消息列表, 包含对话历史.
            system_prompt: 系统指令, 用于设定 LLM 的行为模式.
            config: 生成参数配置. 如果为 None, 使用 temperature=1.0 的默认配置.

        Returns:
            Message: LLM 生成的响应消息, 角色为 "model".
        """
        gemini_mes = self._normalize_messages(messages)
        gen_config = self.__translate_generate(config, system_prompt=system_prompt)
        res = self._retry(
            lambda: self.client.models.generate_content(
                model=self.model,
                contents=gemini_mes,
                config=gen_config,
            ),
        )
        return Message(role="model", content=res.text or "")

    def generate_struct(
        self,
        messages: Message | list[Message],
        schema: type[BaseModel],
        system_prompt: str | None = None,
        config: GenerateConfig | None = None,
    ) -> Message:
        """使用 Gemini API 生成符合 Pydantic 模型的结构化响应.

        通过设置 response_mime_type="application/json" 与 response_schema=schema
        实现结构化输出. config 中的 response_mime_type 和 response_schema 字段
        会被忽略, 以 schema 参数为准.

        Args:
            messages: 单个消息或消息列表, 包含对话历史.
            schema: Pydantic BaseModel 子类, 定义期望的输出结构.
            system_prompt: 系统指令, 用于设定 LLM 的行为模式.
            config: 生成参数配置.

        Returns:
            Message: LLM 生成的响应消息, content 为 JSON 字符串.

        Raises:
            ValueError: 当模型未返回有效内容时.
        """
        gemini_mes = self._normalize_messages(messages)
        gen_config = self.__translate_generate(
            config,
            system_prompt=system_prompt,
            extra={"response_mime_type": "application/json", "response_schema": schema},
        )
        res = self._retry(
            lambda: self.client.models.generate_content(
                model=self.model,
                contents=gemini_mes,
                config=gen_config,
            ),
        )

        content = res.text
        if not content:
            raise ValueError("模型未返回可解析的结构化输出")
        return Message(role="model", content=content)

    def generate_with_tool(
        self,
        messages: Message | list[Message],
        tools: list[Tool[Any]],
        system_prompt: str | None = None,
        config: GenerateConfig | None = None,
        max_tool_rounds: int = 5,
    ) -> Message:
        """使用 Gemini API 生成响应, 并自动执行模型请求的工具调用.

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
        tool_map = self.__validate_tool_loop_inputs(tools, max_tool_rounds)
        contents = self.__prepare_tool_contents(messages)
        gen_config = self.__translate_generate(
            config,
            system_prompt=system_prompt,
            extra={
                "tools": [self.__to_gemini_tool(tools)],
                "automatic_function_calling": AutomaticFunctionCallingConfig(
                    disable=True,
                    maximum_remote_calls=None,
                ),
            },
        )

        for _ in range(max_tool_rounds):
            res = self._retry(
                lambda: self.client.models.generate_content(
                    model=self.model,
                    contents=cast(ContentListUnionDict, contents),
                    config=gen_config,
                ),
            )

            if not res.candidates:
                return Message(role="model", content="")

            function_calls = res.function_calls or []
            if not function_calls:
                return Message(role="model", content=res.text or "")

            self.__append_tool_results(contents, res, tool_map)

        raise RuntimeError("超过最大工具调用轮数")

    def generate_struct_with_tool(
        self,
        messages: Message | list[Message],
        schema: type[BaseModel],
        tools: list[Tool[Any]],
        system_prompt: str | None = None,
        config: GenerateConfig | None = None,
        max_tool_rounds: int = 5,
    ) -> Message:
        """使用 Gemini API 生成结构化响应, 并自动执行工具调用.

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
            ValueError: 当工具列表为空、工具名重复、最大轮数无效或模型未返回有效结构化输出时.
            RuntimeError: 当超过最大工具调用轮数时.
        """
        tool_map = self.__validate_tool_loop_inputs(tools, max_tool_rounds)
        contents = self.__prepare_tool_contents(messages)
        gen_config = self.__translate_generate(
            config,
            system_prompt=system_prompt,
            extra={
                "tools": [self.__to_gemini_tool(tools)],
                "automatic_function_calling": AutomaticFunctionCallingConfig(
                    disable=True,
                    maximum_remote_calls=None,
                ),
                "response_mime_type": "application/json",
                "response_schema": schema,
            },
        )

        for _ in range(max_tool_rounds):
            res = self._retry(
                lambda: self.client.models.generate_content(
                    model=self.model,
                    contents=cast(ContentListUnionDict, contents),
                    config=gen_config,
                ),
            )

            if not res.candidates:
                raise ValueError("模型未返回有效响应")

            function_calls = res.function_calls or []
            if not function_calls:
                content = res.text
                if not content:
                    raise ValueError("模型未返回可解析的结构化输出")
                return Message(role="model", content=content)

            self.__append_tool_results(contents, res, tool_map)

        raise RuntimeError("超过最大工具调用轮数")

    def edit_image(
        self,
        messages: Message | list[Message],
        system_prompt: str | None = None,
        config: EditImageConfig | None = None,
    ) -> Message:
        """使用 Gemini API 编辑/生成图像.

        Args:
            messages: 单个消息或消息列表, 必须包含至少一张图像.
            system_prompt: 可选的系统指令.
            config: 图像生成参数配置.

        Returns:
            Message: LLM 生成的图像.

        Raises:
            ValueError: 当消息中没有图像时.
        """
        messages = _ensure_list(messages)
        if not any(msg.images for msg in messages):
            raise ValueError("没有传入图片")

        gemini_mes = self._normalize_messages(messages)
        cfg = config or GeminiLLM.EditImageConfig()
        gen_config = self.__translate_edit_image(cfg, system_prompt)
        res = self._retry(
            lambda: self.client.models.generate_content(
                model=self.model,
                contents=gemini_mes,
                config=gen_config,
            ),
        )

        contents: list[TextContent | ImageContent] = []
        if res.parts:
            for part in res.parts:
                if hasattr(part, "inline_data") and part.inline_data:
                    img_data = part.inline_data.data
                    mime_type = part.inline_data.mime_type
                    if img_data and mime_type:
                        img_bytes = (
                            img_data
                            if isinstance(img_data, bytes)
                            else base64.b64decode(img_data)
                        )
                        contents.append(
                            ImageContent(source=img_bytes, mime_type=mime_type)
                        )

        if not contents:
            return Message(role="model", content="")
        return Message(role="model", content=contents)

    def __translate_generate(
        self,
        config: GenerateConfig | None,
        system_prompt: str | None = None,
        extra: dict[str, Any] | None = None,
    ) -> genai.types.GenerateContentConfig:
        """将 GenerateConfig 构造为 GenerateContentConfig.

        Args:
            config: 生成配置, 为 None 时使用默认 temperature=1.0.
            system_prompt: 系统指令.
            extra: 额外覆盖字段 (如 tools, response_mime_type, response_schema).

        Returns:
            genai.types.GenerateContentConfig: 构造好的生成配置对象.
        """
        cfg = config or GeminiLLM.GenerateConfig()
        extra = extra or {}

        thinking_config: genai.types.ThinkingConfig | None = None
        if cfg.think_level is not None:
            thinking_config = genai.types.ThinkingConfig(
                thinking_level=genai.types.ThinkingLevel(cfg.think_level),
            )

        return genai.types.GenerateContentConfig(
            system_instruction=system_prompt,
            temperature=cfg.temperature,
            top_p=cfg.top_p,
            top_k=cfg.top_k,
            max_output_tokens=cfg.max_output_tokens,
            candidate_count=cfg.candidate_count,
            stop_sequences=cfg.stop_sequences,
            frequency_penalty=cfg.frequency_penalty,
            presence_penalty=cfg.presence_penalty,
            seed=cfg.seed,
            thinking_config=thinking_config,
            tools=extra.get("tools"),
            automatic_function_calling=extra.get("automatic_function_calling"),
            response_mime_type=extra.get("response_mime_type"),
            response_schema=extra.get("response_schema"),
        )

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
    def __prepare_tool_contents(messages: Message | list[Message]) -> list[Content]:
        """将内部消息转换为 Gemini 工具调用循环使用的 Content 列表."""
        return [
            Content.model_validate(message_to_gemini(msg))
            for msg in _ensure_list(messages)
        ]

    @staticmethod
    def __to_gemini_tool(tools: list[Tool[Any]]) -> GeminiTool:
        """将内部 Tool 列表转换为 Gemini 函数声明工具."""
        declarations = [
            FunctionDeclaration(
                name=tool.name,
                description=tool.description,
                parameters_json_schema=tool_to_gemini(tool)["parameters"],
            )
            for tool in tools
        ]
        return GeminiTool(function_declarations=declarations)

    @staticmethod
    def __append_tool_results(
        contents: list[Content],
        response: Any,
        tool_map: dict[str, Tool[Any]],
    ) -> None:
        """追加 Gemini 模型工具调用轮次和对应工具返回值."""
        model_content = response.candidates[0].content
        if model_content is not None:
            contents.append(model_content)

        response_parts: list[Part] = []
        for function_call in response.function_calls or []:
            tool_name = function_call.name or ""
            tool_args = function_call.args or {}
            try:
                tool = tool_map.get(tool_name)
                if tool is None:
                    raise ValueError(f"未知工具: {tool_name}")
                tool_result = {"is_error": False, "result": tool(**tool_args)}
            except Exception as e:
                tool_result = {"is_error": True, "error": str(e)}

            response_parts.append(
                Part.from_function_response(name=tool_name, response=tool_result)
            )

        contents.append(Content(role="user", parts=response_parts))

    def __translate_edit_image(
        self,
        config: EditImageConfig,
        system_prompt: str | None = None,
    ) -> genai.types.GenerateContentConfig:
        """将 EditImageConfig 构造为 GenerateContentConfig.

        Args:
            config: 图像编辑配置对象.
            system_prompt: 系统指令.

        Returns:
            genai.types.GenerateContentConfig: 构造好的生成配置对象.
        """
        _image_fields = {
            "aspect_ratio",
            "image_size",
            "output_mime_type",
            "output_compression_quality",
        }
        image_config_kwargs: dict[str, Any] = config.model_dump(
            include=_image_fields, exclude_none=True
        )

        image_config = (
            genai.types.ImageConfig(**image_config_kwargs)
            if image_config_kwargs
            else None
        )

        thinking_config: genai.types.ThinkingConfig | None = None
        if config.think_level is not None:
            thinking_config = genai.types.ThinkingConfig(
                thinking_level=genai.types.ThinkingLevel(config.think_level),
            )

        return genai.types.GenerateContentConfig(
            system_instruction=system_prompt,
            temperature=config.temperature,
            seed=config.seed,
            thinking_config=thinking_config,
            response_modalities=["IMAGE"],
            image_config=image_config,
        )
