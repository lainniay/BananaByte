import base64
import logging
import os
from collections.abc import Callable
from typing import Annotated, Any, Literal, TypeVar, cast

from google import genai
from google.genai.errors import ServerError
from google.genai.types import ContentListUnionDict, HttpOptions
from openai import (
    APIConnectionError,
    APITimeoutError,
    InternalServerError,
    OpenAI,
    RateLimitError,
)
from openai.types.chat import ChatCompletionMessageParam
from pydantic import BaseModel, Field
from tenacity import (
    Retrying,
    before_sleep_log,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

from core.schemas import ImageContent, Message, TextContent

logger = logging.getLogger(__name__)


_OPENAI_RETRYABLE = (
    RateLimitError,
    InternalServerError,
    APITimeoutError,
    APIConnectionError,
)
_GEMINI_RETRYABLE = (ServerError,)


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


StructuredOutput = TypeVar("StructuredOutput", bound=BaseModel)


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
            tools: 模型可调用的工具列表.
            tool_choice: 工具选择策略.
            parallel_tool_calls: 是否允许并行工具调用.
            max_completion_tokens: 模型生成的最大 token 数量.
        """

        timeout: float | None = None  # 单位是秒
        temperature: Annotated[float, Field(ge=0.0, le=2.0)] = 1.0
        top_p: Annotated[float | None, Field(ge=0.0, le=1.0)] = None
        frequency_penalty: Annotated[float | None, Field(ge=-2.0, le=2.0)] = None
        presence_penalty: Annotated[float | None, Field(ge=-2.0, le=2.0)] = None
        n: Annotated[int, Field(ge=1)] = 1
        seed: int | None = None
        prediction: str | None = None
        verbosity: Literal["low", "medium", "high"] | None = None
        tools: list[dict] | None = None
        tool_choice: Literal["none", "auto", "required"] | dict | None = None
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
        openai_mes = [msg.to_openai_format() for msg in messages]
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

    def generate_struct(
        self,
        messages: Message | list[Message],
        schema: type[StructuredOutput],
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
            tools: 工具/函数调用定义列表.
            tool_config: 工具调用配置.
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

        # 工具/函数调用
        tools: list[Any] | None = None
        tool_config: genai.types.ToolConfig | None = None

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
            [msg.to_gemini_format() for msg in messages],
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
        schema: type[StructuredOutput],
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
            Message: LLM 生成的响应消息, 可能包含文本和/或图像.

        Raises:
            ValueError: 当消息中没有图像时.
        """
        messages = _ensure_list(messages)
        if not any(msg.images for msg in messages):
            raise ValueError("没有传入图片")

        gemini_mes = self._normalize_messages(messages)
        cfg = config or GeminiLLM.EditImageConfig()
        _, gen_config = self.__translate_edit_image(cfg, system_prompt)
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
            extra: 额外覆盖字段 (如 response_mime_type, response_schema).

        Returns:
            genai.types.GenerateContentConfig: 构造好的生成配置对象.
        """
        cfg = config or GeminiLLM.GenerateConfig()

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
            tools=cfg.tools,
            tool_config=cfg.tool_config,
            response_mime_type=(extra or {}).get("response_mime_type"),
            response_schema=(extra or {}).get("response_schema"),
        )

    def __translate_edit_image(
        self,
        config: EditImageConfig,
        system_prompt: str | None = None,
    ) -> tuple[dict[str, Any], genai.types.GenerateContentConfig]:
        """将 EditImageConfig 构造为 (image_config_kwargs, GenerateContentConfig).

        Args:
            config: 图像编辑配置对象.
            system_prompt: 系统指令.

        Returns:
            tuple[dict[str, Any], genai.types.GenerateContentConfig]: 包含以下内容的元组:
                - image_config_kwargs: 供调用方按需构造 genai.types.ImageConfig.
                - gen_config: 构造好的生成配置对象.
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

        gen_config = genai.types.GenerateContentConfig(
            system_instruction=system_prompt,
            temperature=config.temperature,
            seed=config.seed,
            thinking_config=thinking_config,
            response_modalities=["IMAGE"],
            image_config=image_config,
        )
        return image_config_kwargs, gen_config
