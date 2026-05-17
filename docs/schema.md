# core.schemas 使用说明

`core.schemas` 定义业务代码传给模型, 以及从模型结果中读取内容时使用的数据对象. 这些对象是轻量 `dataclass`, 可以直接构造和传递.

业务使用时主要记住三类对象:

- `Message`: 一条用户消息或模型回复.
- `TextContent`: 多模态消息中的一段文本.
- `ImageContent`: 多模态消息中的一张图片.

如果只是纯文本请求, 只需要使用 `Message(content="...")`. 如果请求中包含图片, 再使用 `TextContent` 和 `ImageContent`.

## ImageContent

接口:

```python
ImageContent(
    source: bytes,
    mime_type: str = "image/png",
    type: Literal["image"] = "image",
)
```

说明:

- `source`: 图片二进制数据.
- `mime_type`: 图片 MIME 类型, 例如 `image/png` 或 `image/jpeg`.
- `type`: 固定为 `image`, 一般不需要手动设置.

### 从 base64 创建图片

接口:

```python
ImageContent.from_base64(data: str, mime_type: str = "image/png") -> ImageContent
```

说明:

- 将 base64 字符串解码为 `ImageContent`.
- 适合上游系统已经把图片转成 base64 字符串的场景.

示例:

```python
from core.schemas import ImageContent

image = ImageContent.from_base64(base64_text, mime_type="image/png")
```

### 从本地文件创建图片

接口:

```python
ImageContent.from_file(path: str | Path) -> ImageContent
```

说明:

- 从本地图片文件读取二进制数据.
- MIME 类型通过文件路径自动推断.
- 如果无法识别文件类型, 会抛出 `ValueError`.

示例:

```python
from core.schemas import ImageContent

image = ImageContent.from_file("workspace/input.png")
```

### 保存图片到本地文件

接口:

```python
save_to_file(path: str | Path) -> Path
```

说明:

- 将 `ImageContent.source` 写入目标路径.
- 如果父目录不存在, 会自动创建.
- 返回保存后的 `Path`.
- 常用于保存 `llm.edit_image()` 返回的图片.

示例:

```python
from core.schemas import ImageContent

image = ImageContent.from_file("workspace/input.png")
output_path = image.save_to_file("workspace/copy.png")
```

## TextContent

接口:

```python
TextContent(
    text: str,
    type: Literal["text"] = "text",
)
```

说明:

- `text`: 文本内容.
- `type`: 固定为 `text`, 一般不需要手动设置.
- 在多模态消息中, 文本需要用 `TextContent` 包装.

示例:

```python
from core.schemas import TextContent

text = TextContent(text="Describe this image")
```

## Message

接口:

```python
Message(
    content: str | list[TextContent | ImageContent],
    role: Literal["user", "model"] = "user",
)
```

说明:

- `content`: 消息内容. 可以是纯文本字符串, 也可以是 `TextContent` 和 `ImageContent` 的列表.
- `role`: 消息角色. 用户输入使用 `user`, 模型输出使用 `model`.

常见写法:

```python
from core.schemas import ImageContent, Message, TextContent

# 纯文本消息.
text_message = Message(content="分析这段文案")

# 图文混合消息.
image = ImageContent.from_file("workspace/input.png")
multimodal_message = Message(
    content=[
        TextContent(text="分析这张图片"),
        image,
    ]
)
```

### 读取消息文本

接口:

```python
message.text -> str
```

说明:

- 如果 `content` 是字符串, 直接返回该字符串.
- 如果 `content` 是列表, 会提取所有 `TextContent.text`, 并使用空格拼接.
- 常用于读取模型文本回复, 或从多模态消息中取出 prompt 文本.

示例:

```python
from core.schemas import Message, TextContent

message = Message(content=[TextContent(text="hello"), TextContent(text="world")])

print(message.text)
```

### 读取消息图片

接口:

```python
message.images -> list[ImageContent]
```

说明:

- 提取消息中的所有图片内容.
- 如果没有图片, 返回空列表.
- 常用于读取 `edit_image()` 返回的图片结果.

示例:

```python
from core.schemas import ImageContent, Message, TextContent

image = ImageContent.from_file("workspace/input.png")
message = Message(content=[TextContent(text="Edit this image"), image])

first_image = message.images[0]
```

### 读取所有消息片段

接口:

```python
message.parts -> list[TextContent | ImageContent]
```

说明:

- 返回消息的所有内容片段.
- 如果 `content` 是字符串, 会返回 `[TextContent(text=content)]`.
- 普通业务代码较少直接使用, 通常使用 `text` 或 `images` 即可.

示例:

```python
from core.schemas import Message

message = Message(content="hello")
parts = message.parts
```

### 将 JSON 文本解析为对象

接口:

```python
parse_as(schema: type[T]) -> T
```

说明:

- 将 `message.text` 当作 JSON 字符串解析.
- `schema` 必须是 Pydantic `BaseModel` 子类.
- 如果 JSON 不合法或字段不符合 schema, 会抛出 `pydantic.ValidationError`.
- 常和 `llm.generate_struct()` 搭配使用.

示例:

```python
from pydantic import BaseModel

from core.schemas import Message


class Answer(BaseModel):
    name: str
    score: int


message = Message(content='{"name":"banana","score":10}', role="model")
answer = message.parse_as(Answer)

print(answer.score)
```

## 纯文本消息示例

```python
from core.schemas import Message

message = Message(content="hello")
print(message.text)
```

## 多模态消息示例

```python
from core.schemas import ImageContent, Message, TextContent

image = ImageContent.from_file("workspace/input.png")
message = Message(
    content=[
        TextContent(text="Describe the image"),
        image,
    ]
)

print(message.text)
print(len(message.images))
```
