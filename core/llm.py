from abc import ABC, abstractmethod
from typing import Literal, cast, overload
from google.genai.types import HttpOptions
from litellm import completion, ModelResponse
from google import genai
from dotenv import load_dotenv
import os

from core.schemas import Message, ImageContent, TextContent


class BaseLLM(ABC):
    @abstractmethod
    def generate(
        self,
        messages: list[Message],
        system_prompt: str | None = None,
        temperature: float = 0,
    ) -> Message:
        pass


class OpenAILLM(BaseLLM):
    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float = 60,
    ) -> None:
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
        temperature: float = 0,
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
        ratio: ImageRatio = "16:9",
        resolution: ImageResolution = "2k",
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

        res = self.client.models.generate_content(
            model=self.model,
            contents=gemini_mes,
            config=genai.types.GenerateContentConfig(
                temperature=temperature,
                response_modalities=["TEXT", "IMAGE"],
                image_config=genai.types.ImageConfig(
                    aspect_ratio=ratio, image_size=resolution
                ),
            ),
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
                elif hasattr(part, "text") and part.text:
                    contents.append(TextContent(text=part.text))

        if len(contents) == 1 and isinstance(contents[0], TextContent):
            return Message(role="model", content=contents[0].text)

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
    """
    统一的 LLM 工厂函数，根据模型名称或指定的提供商创建相应的 LLM 实例。

    Args:
        model: 模型名称（如 "gpt-4", "gemini-2.0-flash"）
        provider: 显式指定提供商，可选 "openai" 或 "gemini"。
                 如果不指定，将根据模型名称自动识别。
        api_key: API 密钥，如果不提供则从环境变量读取
        base_url: API 基础 URL，如果不提供则使用默认值
        timeout: 请求超时时间（秒或毫秒，取决于提供商）

    Returns:
        GeminiLLM | OpenAILLM: 相应的 LLM 实例（OpenAILLM 或 GeminiLLM）

    Examples:
        # 自动识别提供商
        llm1 = create_llm("gpt-4")
        llm2 = create_llm("gemini-2.0-flash")

        # 显式指定提供商
        llm3 = create_llm("gpt-4", provider="openai")
        llm4 = create_llm("gemini-2.0-flash", provider="gemini")

        # 自定义配置
        llm5 = create_llm(
            "gpt-4",
            api_key="your-api-key",
            base_url="https://api.custom.com",
            timeout=30
        )
    """
    # 如果没有显式指定提供商，根据模型名称自动识别
    if provider is None:
        model_lower = model.lower()
        if "gemini" in model_lower:
            provider = "gemini"
        else:
            # 默认使用 OpenAI（兼容大部分第三方 API）
            provider = "openai"

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
