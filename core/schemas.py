from typing import Literal, Self
import mimetypes
import base64
from pydantic import BaseModel
from pathlib import Path


class ImageContent(BaseModel):
    type: Literal["image"] = "image"
    source: str
    mime_type: str = "image/png"

    @classmethod
    def from_base64(cls, data: str, mime_type: str = "image/png") -> Self:
        return cls(source=data, mime_type=mime_type)

    @classmethod
    def from_file(cls, path: str | Path) -> Self:
        path = Path(path)
        mime_type, _ = mimetypes.guess_type(path)
        if not mime_type:
            raise ValueError(f"无法从{path}中读取文件类型")
        with open(path, "rb") as f:
            data = base64.b64encode(f.read()).decode()
        return cls(source=data, mime_type=mime_type)


class TextContent(BaseModel):
    type: Literal["text"] = "text"
    text: str


class Message(BaseModel):
    role: Literal["user", "model"]
    content: str | list[TextContent | ImageContent]

    @property
    def text(self) -> str:
        if isinstance(self.content, str):
            return self.content
        texts = [item.text for item in self.content if isinstance(item, TextContent)]
        return " ".join(texts)

    @property
    def images(self) -> list[ImageContent]:
        if isinstance(self.content, str):
            return []
        return [item for item in self.content if isinstance(item, ImageContent)]

    @property
    def parts(self) -> list[TextContent | ImageContent]:
        if isinstance(self.content, str):
            return [TextContent(text=self.content)]
        return self.content

    def to_gemini_format(self) -> dict:
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
