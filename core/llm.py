import os
from abc import ABC, abstractmethod
from typing import Any, Literal, cast, overload

from dotenv import load_dotenv
from google import genai
from google.genai.types import HttpOptions
from langsmith import traceable
from litellm import ModelResponse, completion

from core.schemas import ImageContent, Message, TextContent


def _process_llm_inputs(inputs: dict[str, Any]) -> dict[str, Any]:
    """将 trace 输入转换为 LangSmith 兼容的 OpenAI 消息格式."""
    messages = inputs.get("messages", [])
    system_prompt = inputs.get("system_prompt")

    if isinstance(messages, Message):
        messages = [messages]

    openai_messages: list[dict[str, Any]] = []
    if isinstance(system_prompt, str) and system_prompt:
        openai_messages.append({"role": "system", "content": system_prompt})

    if isinstance(messages, list):
        for message in messages:
            if isinstance(message, Message):
                openai_messages.append(cast(dict[str, Any], message.to_openai_format()))
            elif isinstance(message, dict):
                openai_messages.append(message)

    return {"messages": openai_messages}


def _process_llm_outputs(output: Any) -> dict[str, Any]:
    """将 trace 输出转换为 LangSmith 兼容的 OpenAI 响应格式."""
    if isinstance(output, Message):
        return cast(dict[str, Any], output.to_openai_format())
    return {"output": output}


class BaseLLM(ABC):
    """LLM 基类, 定义统一的生成接口.

    所有 LLM 实现都必须继承此类并实现 generate 方法.
    """

    @abstractmethod
    def generate(
        self,
        messages: list[Message],
        system_prompt: str | None = None,
        temperature: float = 1.0,
    ) -> Message:
        """生成 LLM 响应.

        Args:
            messages: 消息列表, 包含对话历史.
            system_prompt: 系统提示词, 用于设定 LLM 的行为模式.
            temperature: 温度参数, 控制输出的随机性, 范围 [0, 2].

        Returns:
            Message: LLM 生成的响应消息.
        """
        pass


class OpenAILLM(BaseLLM):
    """OpenAI 兼容的 LLM 实现.

    支持 OpenAI API 及兼容的第三方 API (如 Azure OpenAI, Anthropic 等).
    """

    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float = 60,
    ) -> None:
        """初始化 OpenAI LLM 客户端.

        Args:
            model: 模型名称, 如 "gpt-4", "gpt-3.5-turbo".
            api_key: API 密钥. 如果为 None, 从环境变量 OPENAI_API_KEY 读取.
            base_url: API 基础 URL. 如果为 None, 从环境变量 OPENAI_API_BASE 读取.
            timeout: 请求超时时间, 单位是秒.

        Raises:
            ValueError: 当 API 密钥缺失时.
        """
        super().__init__()
        self.model = model
        load_dotenv()
        self.api_key = api_key or os.getenv("OPENAI_API_KEY", "")
        self.base_url = base_url or os.getenv("OPENAI_API_BASE", "")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY 环境变量缺失")
        self.timeout = timeout

    def generate(
        self,
        messages: Message | list[Message],
        system_prompt: str | None = None,
        temperature: float = 1.0,
    ) -> Message:
        """使用 OpenAI API 生成响应.

        Args:
            messages: 单个消息或消息列表, 包含对话历史.
            system_prompt: 系统提示词. 如果提供, 会插入到消息列表开头.
            temperature: 温度参数, 控制输出的随机性, 范围 [0, 2].

        Returns:
            Message: LLM 生成的响应消息, 角色为 "model".
        """
        return self._generate_traced(messages, system_prompt, temperature)

    @traceable(
        run_type="llm",
        name="OpenAILLM.generate",
        process_inputs=_process_llm_inputs,
        process_outputs=_process_llm_outputs,
    )
    def _generate_traced(
        self,
        messages: Message | list[Message],
        system_prompt: str | None = None,
        temperature: float = 1.0,
    ) -> Message:
        if isinstance(messages, Message):
            messages = [messages]
        openai_mess = [cast(dict, msg.to_openai_format()) for msg in messages]
        if system_prompt:
            openai_mess.insert(0, {"role": "system", "content": system_prompt})

        res = completion(
            model=self.model,
            messages=openai_mess,
            temperature=temperature,
            api_key=self.api_key,
            api_base=self.base_url if self.base_url else None,
            timeout=self.timeout,
        )
        if isinstance(res, ModelResponse):
            content = res.choices[0].message.content or ""
        else:
            content = ""
        return Message(role="model", content=content)


class GeminiLLM(BaseLLM):
    """Google Gemini LLM 实现.

    支持 Google Gemini API, 包括文本生成和图像编辑功能.
    """

    ImageRatio = Literal[
        "1:1",
        "1:4",
        "1:8",
        "2:3",
        "3:2",
        "3:4",
        "4:1",
        "4:3",
        "4:5",
        "5:4",
        "8:1",
        "9:16",
        "16:9",
        "21:9",
    ]
    ImageResolution = Literal["512", "1k", "2k", "4k"]

    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: int = 120 * 1000,
    ) -> None:
        """初始化 Gemini LLM 客户端.

        Args:
            model: 模型名称.
            api_key: API 密钥. 如果为 None, 从环境变量 GEMINI_API_KEY 读取.
            base_url: API 基础 URL. 如果为 None, 从环境变量 GEMINI_API_BASE 读取.
            timeout: 请求超时时间, 单位是毫秒.

        Raises:
            ValueError: 当 API 密钥缺失时.
        """
        super().__init__()
        self.model = model
        load_dotenv()
        self.api_key = api_key or os.getenv("GEMINI_API_KEY", "")
        self.base_url = base_url or os.getenv("GEMINI_API_BASE", "")
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY 环境变量缺失")
        self.client = genai.Client(
            api_key=self.api_key,
            http_options=HttpOptions(base_url=self.base_url, timeout=timeout),
        )

    def generate(
        self,
        messages: Message | list[Message],
        system_prompt: str | None = None,
        temperature: float = 1.0,
    ) -> Message:
        """使用 Gemini API 生成响应.

        Args:
            messages: 单个消息或消息列表, 包含对话历史.
            system_prompt: 系统指令, 用于设定 LLM 的行为模式.
            temperature: 温度参数, 控制输出的随机性, 范围 [0, 2].

        Returns:
            Message: LLM 生成的响应消息, 角色为 "model".
        """
        return self._generate_traced(messages, system_prompt, temperature)

    @traceable(
        run_type="llm",
        name="GeminiLLM.generate",
        process_inputs=_process_llm_inputs,
        process_outputs=_process_llm_outputs,
    )
    def _generate_traced(
        self,
        messages: Message | list[Message],
        system_prompt: str | None = None,
        temperature: float = 1.0,
    ) -> Message:
        if isinstance(messages, Message):
            messages = [messages]

        gemini_mes = [msg.to_gemini_format() for msg in messages]

        res = self.client.models.generate_content(
            model=self.model,
            contents=gemini_mes,
            config=genai.types.GenerateContentConfig(
                system_instruction=system_prompt, temperature=temperature
            ),
        )
        return Message(role="model", content=res.text or "")

    def edit_image(
        self,
        messages: Message | list[Message],
        temperature: float = 1.0,
        ratio: None | ImageRatio = None,
        resolution: None | ImageResolution = None,
    ) -> Message:
        """使用 Gemini API 编辑图像.

        Args:
            messages: 单个消息或消息列表, 必须包含至少一张图像.
            temperature: 温度参数, 控制输出的随机性, 范围 [0, 2].
            ratio: 输出图像的宽高比, 如 "16:9", "1:1".
            resolution: 输出图像的分辨率, 如 "2k", "4k".

        Returns:
            Message: LLM 生成的响应消息, 可能包含文本和/或图像.

        Raises:
            ValueError: 当消息中没有图像时.
        """
        return self._edit_image_traced(messages, temperature, ratio, resolution)

    @traceable(
        run_type="llm",
        name="GeminiLLM.edit_image",
        process_inputs=_process_llm_inputs,
        process_outputs=_process_llm_outputs,
    )
    def _edit_image_traced(
        self,
        messages: Message | list[Message],
        temperature: float = 1.0,
        ratio: None | ImageRatio = None,
        resolution: None | ImageResolution = None,
    ) -> Message:
        if isinstance(messages, Message):
            messages = [messages]
        img_content: ImageContent | None = None
        for message in messages:
            if isinstance(message.content, list):
                for item in message.content:
                    if item.type == "image":
                        img_content = item

        if not img_content:
            raise ValueError("没有传入图片")

        gemini_mes = [item.to_gemini_format() for item in messages]

        config_kwargs: dict[str, Any] = {
            "temperature": temperature,
            "response_modalities": ["IMAGE"],
        }
        image_config_kwargs: dict[str, Any] = {}
        if ratio is not None:
            image_config_kwargs["aspect_ratio"] = ratio
        if resolution is not None:
            image_config_kwargs["image_size"] = resolution
        if image_config_kwargs:
            config_kwargs["image_config"] = genai.types.ImageConfig(
                **image_config_kwargs
            )

        res = self.client.models.generate_content(
            model=self.model,
            contents=gemini_mes,
            config=genai.types.GenerateContentConfig(**config_kwargs),
        )

        contents: list[TextContent | ImageContent] = []
        if res.parts:
            for part in res.parts:
                if hasattr(part, "inline_data") and part.inline_data:
                    img_data = part.inline_data.data
                    mime_type = part.inline_data.mime_type
                    if img_data and mime_type:
                        if isinstance(img_data, bytes):
                            import base64

                            img_data = base64.b64encode(img_data).decode()
                        contents.append(
                            ImageContent(source=img_data, mime_type=mime_type)
                        )

        if not contents:
            return Message(role="model", content="")

        return Message(role="model", content=contents)


@overload
def create_llm(
    model: str,
    provider: Literal["gemini"],
    api_key: str | None = None,
    base_url: str | None = None,
    timeout: float | int | None = None,
) -> GeminiLLM: ...


@overload
def create_llm(
    model: str,
    provider: Literal["openai"],
    api_key: str | None = None,
    base_url: str | None = None,
    timeout: float | int | None = None,
) -> OpenAILLM: ...


@overload
def create_llm(
    model: str,
    provider: None = None,
    api_key: str | None = None,
    base_url: str | None = None,
    timeout: float | int | None = None,
) -> GeminiLLM | OpenAILLM: ...


def create_llm(
    model: str,
    provider: Literal["openai", "gemini"] | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
    timeout: float | int | None = None,
) -> GeminiLLM | OpenAILLM:
    """统一的 LLM 工厂函数, 根据模型名称或指定的提供商创建相应的 LLM 实例.

    Args:
        model: 模型名称.
        provider: 显式指定提供商, 可选 "openai" 或 "gemini".
                 如果不指定, 将根据模型名称自动识别.
        api_key: API 密钥, 如果不提供则从环境变量读取.
        base_url: API 基础 URL, 如果不提供则使用默认值.
        timeout: 请求超时时间 (秒或毫秒, 取决于提供商).

    Returns:
        GeminiLLM | OpenAILLM: 相应的 LLM 实例 (OpenAILLM 或 GeminiLLM).
    """
    if provider is None:
        model_lower = model.lower()
        provider = "gemini" if "gemini" in model_lower else "openai"

    # 根据提供商创建相应的 LLM 实例
    if provider == "gemini":
        # Gemini 的 timeout 单位是毫秒
        gemini_timeout = int(timeout) if timeout is not None else 120 * 1000
        return GeminiLLM(
            model=model,
            api_key=api_key,
            base_url=base_url,
            timeout=gemini_timeout,
        )
    elif provider == "openai":
        # OpenAI 的 timeout 单位是秒
        openai_timeout = float(timeout) if timeout is not None else 10.0
        return OpenAILLM(
            model=model,
            api_key=api_key,
            base_url=base_url,
            timeout=openai_timeout,
        )
