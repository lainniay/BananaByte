import base64
import mimetypes
from pathlib import Path
from typing import Literal, Self

from pydantic import BaseModel


class ImageContent(BaseModel):
    """图像内容模型.

    使用 base64 编码存储图像数据.

    Attributes:
        type: 内容类型, 固定为 "image".
        source: base64 编码的图像数据.
        mime_type: MIME 类型, 如 "image/png", "image/jpeg".

    """

    type: Literal["image"] = "image"
    source: str
    mime_type: str = "image/png"

    @classmethod
    def from_base64(cls, data: str, mime_type: str = "image/png") -> Self:
        """从 base64 字符串创建图像内容.

        Args:
            data: base64 编码的图像数据.
            mime_type: MIME 类型.

        Returns:
            ImageContent: 图像内容实例.
        """
        return cls(source=data, mime_type=mime_type)

    @classmethod
    def from_file(cls, path: str | Path) -> Self:
        """从文件创建图像内容.

        Args:
            path: 图像文件路径.

        Returns:
            ImageContent: 图像内容实例.

        Raises:
            ValueError: 当无法识别文件类型时.
        """
        path = Path(path)
        mime_type, _ = mimetypes.guess_type(path)
        if not mime_type:
            raise ValueError(f"无法从{path}中读取文件类型")
        with open(path, "rb") as f:
            data = base64.b64encode(f.read()).decode()
        return cls(source=data, mime_type=mime_type)

    def save_to_file(self, path: str | Path) -> Path:
        """将图像内容保存为文件.

        Args:
            path: 输出文件路径.

        Returns:
            Path: 保存后的文件路径.

        Raises:
            ValueError: 当 base64 数据无效时.
        """
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            image_bytes = base64.b64decode(self.source)
        except Exception as exc:
            raise ValueError("图片 base64 数据无效") from exc
        with open(output_path, "wb") as f:
            f.write(image_bytes)
        return output_path


class TextContent(BaseModel):
    """文本内容模型.

    Attributes:
        type: 内容类型, 固定为 "text".
        text: 文本内容.
    """

    type: Literal["text"] = "text"
    text: str


class Message(BaseModel):
    """消息模型, 表示对话中的一条消息.

    支持纯文本消息和多模态消息 (文本 + 图像).

    Attributes:
        role: 消息角色, "user" 表示用户, "model" 表示 LLM.
        content: 消息内容, 可以是字符串或内容列表 (TextContent/ImageContent).
    """

    role: Literal["user", "model"] = "user"
    content: str | list[TextContent | ImageContent]

    @property
    def text(self) -> str:
        """提取消息中的所有文本内容.

        Returns:
            str: 拼接后的文本内容, 多个文本片段用空格分隔.
        """
        if isinstance(self.content, str):
            return self.content
        texts = [item.text for item in self.content if isinstance(item, TextContent)]
        return " ".join(texts)

    @property
    def images(self) -> list[ImageContent]:
        """提取消息中的所有图像内容.

        Returns:
            list[ImageContent]: 图像内容列表, 如果没有图像则返回空列表.
        """
        if isinstance(self.content, str):
            return []
        return [item for item in self.content if isinstance(item, ImageContent)]

    @property
    def parts(self) -> list[TextContent | ImageContent]:
        """获取消息的所有内容部分.

        Returns:
            list[TextContent | ImageContent]: 内容部分列表.
                如果 content 是字符串, 返回包含单个 TextContent 的列表.
        """
        if isinstance(self.content, str):
            return [TextContent(text=self.content)]
        return self.content

    def to_gemini_format(self) -> dict:
        """转换为 Gemini API 格式.

        Returns:
            dict: Gemini API 格式的消息字典, 包含 "role" 和 "parts" 字段.
        """
        if isinstance(self.content, str):
            return {"role": self.role, "parts": [{"text": self.content}]}

        parts = []
        for item in self.content:
            if item.type == "text":
                parts.append({"text": item.text})
            elif item.type == "image":
                parts.append(
                    {"inline_data": {"mime_type": item.mime_type, "data": item.source}}
                )
        return {"role": self.role, "parts": parts}

    def to_openai_format(self) -> dict:
        """转换为 OpenAI API 格式.

        Returns:
            dict: OpenAI API 格式的消息字典, 包含 "role" 和 "content" 字段.
                角色 "model" 会被转换为 "assistant".
        """
        role = "assistant" if self.role == "model" else self.role
        if isinstance(self.content, str):
            return {"role": role, "content": self.content}

        content = []
        for item in self.content:
            if item.type == "text":
                content.append({"type": "text", "text": item.text})
            elif item.type == "image":
                content.append(
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{item.mime_type};base64,{item.source}"
                        },
                    }
                )
        return {"role": role, "content": content}
