import base64
import mimetypes
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Self

from pydantic import BaseModel


@dataclass(slots=True)
class ImageContent:
    """图像内容模型.

    使用二进制存储图像数据.

    Attributes:
        type: 内容类型, 固定为 "image".
        source: 图片的二进制数据.
        mime_type: MIME 类型, 如 "image/png", "image/jpeg".

    """

    source: bytes
    mime_type: str = "image/png"
    type: Literal["image"] = "image"

    @classmethod
    def from_base64(cls, data: str, mime_type: str = "image/png") -> Self:
        """从 base64 字符串创建图像内容.

        Args:
            data: base64 编码的图像数据.
            mime_type: MIME 类型.

        Returns:
            ImageContent: 图像内容实例.
        """
        return cls(source=base64.b64decode(data), mime_type=mime_type)

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
            data = f.read()
        return cls(source=data, mime_type=mime_type)

    def save_to_file(self, path: str | Path) -> Path:
        """将图像内容保存为文件.

        Args:
            path: 输出文件路径.

        Returns:
            Path: 保存后的文件路径.
        """
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "wb") as f:
            f.write(self.source)
        return output_path


@dataclass(slots=True)
class TextContent:
    """文本内容模型.

    Attributes:
        type: 内容类型, 固定为 "text".
        text: 文本内容.
    """

    text: str
    type: Literal["text"] = "text"


@dataclass(slots=True)
class Message:
    """消息模型, 表示对话中的一条消息.

    支持纯文本消息和多模态消息 (文本 + 图像).

    Attributes:
        role: 消息角色, "user" 表示用户, "model" 表示 LLM.
        content: 消息内容, 可以是字符串或内容列表 (TextContent/ImageContent).
    """

    content: str | list[TextContent | ImageContent]
    role: Literal["user", "model"] = "user"

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

    def parse_as[T: BaseModel](self, schema: type[T]) -> T:
        """将消息文本解析为指定的 Pydantic 模型.

        Args:
            schema: 用于校验和解析消息文本的 Pydantic 模型类型.

        Returns:
            解析后的 Pydantic 模型实例.

        Raises:
            pydantic.ValidationError: 当消息文本不是合法 JSON 或不符合 schema 时.
        """
        return schema.model_validate_json(self.text)
