# core.llm 使用说明

`core.llm` 是业务代码调用大模型的入口. 面向业务使用者时, 只需要关心下面这些对象:

```python
from core.llm import GeminiLLM, OpenAILLM
```

一般不要导入 `core.llm` 目录下的子模块. 这些子模块用于框架内部实现, 业务代码只使用 `GeminiLLM` 和 `OpenAILLM` 即可.

## 适用场景

| 需求 | 推荐接口 |
| --- | --- |
| 普通文本问答 | `generate()` |
| 让模型返回 JSON | `generate_struct()` |
| 输入图片并让模型理解图片 | `generate()` + 多模态 `Message` |
| 编辑或生成图片 | `edit_image()` |

工具调用接口目前属于实验能力. 如果只是写普通业务流程, 不建议使用. 如果确实需要让模型调用 Python 函数, 阅读 [`core.llm.tool`](./tool.md).

## 初始化模型

### GeminiLLM

适合调用 Gemini API.

```python
GeminiLLM(
    model: str,
    api_key: str | None = None,
    base_url: str | None = None,
    timeout: int = 120 * 1000,
    max_retries: int = 3,
)
```

参数说明:

- `model`: 模型名称, 例如 `gemini-3-pro-preview`.
- `api_key`: API key. 不传时读取环境变量 `GEMINI_API_KEY`.
- `base_url`: API base URL. 不传时读取环境变量 `GEMINI_API_BASE`.
- `timeout`: 请求超时时间, 单位是毫秒.
- `max_retries`: 失败后的最大重试次数.

最小示例:

```python
from core.llm import GeminiLLM

llm = GeminiLLM(model="gemini-3-pro-preview")
```

### OpenAILLM

适合调用 OpenAI API 或 OpenAI 兼容 API.

```python
OpenAILLM(
    model: str,
    api_key: str | None = None,
    base_url: str | None = None,
    timeout: float = 120,
    max_retries: int = 3,
)
```

参数说明:

- `model`: 模型名称, 例如 `kimi-k2.5`.
- `api_key`: API key. 不传时读取环境变量 `OPENAI_API_KEY`.
- `base_url`: API base URL. 不传时读取环境变量 `OPENAI_API_BASE`.
- `timeout`: 请求超时时间, 单位是秒.
- `max_retries`: 失败后的最大重试次数.

最小示例:

```python
from core.llm import OpenAILLM

llm = OpenAILLM(model="kimi-k2.5")
```

如果项目使用第三方 OpenAI 兼容服务, 可以显式传入 `api_key` 和 `base_url`:

```python
import os

from core.llm import OpenAILLM

llm = OpenAILLM(
    model="kimi-k2.5",
    api_key=os.getenv("KIMI_API_KEY"),
    base_url=os.getenv("KIMI_API_BASE"),
)
```

## 普通文本生成

接口:

```python
generate(
    messages: Message | list[Message],
    system_prompt: str | None = None,
    config: GenerateConfig | None = None,
) -> Message
```

参数说明:

- `messages`: 一条消息或多条消息. 单轮对话传一个 `Message`, 多轮上下文传 `list[Message]`.
- `system_prompt`: 系统提示词, 用于约束模型身份, 输出风格或任务规则.
- `config`: 生成配置, 常用字段包括 `temperature`, `top_p`, `seed`.

返回值:

- 返回 `Message`.
- 文本结果通过 `res.text` 获取.

单轮示例:

```python
from core.llm import GeminiLLM
from core.schemas import Message

llm = GeminiLLM(model="gemini-3-pro-preview")

res = llm.generate(
    messages=Message(content="用一句话解释什么是图像分割"),
    system_prompt="你是一个简洁的技术助手",
)

print(res.text)
```

多轮示例:

```python
from core.llm import OpenAILLM
from core.schemas import Message

llm = OpenAILLM(model="kimi-k2.5")

messages = [
    Message(role="user", content="给这个角色起一个名字"),
    Message(role="model", content="可以叫 Nira"),
    Message(role="user", content="再给她写一句简介"),
]

res = llm.generate(messages=messages)
print(res.text)
```

## 结构化生成

当业务逻辑需要稳定读取字段时, 使用 `generate_struct()`.

接口:

```python
generate_struct(
    messages: Message | list[Message],
    schema: type[BaseModel],
    system_prompt: str | None = None,
    config: GenerateConfig | None = None,
) -> Message
```

参数说明:

- `schema`: Pydantic `BaseModel` 子类, 用来定义模型必须返回的 JSON 结构.
- 其他参数与 `generate()` 相同.

返回值:

- 返回 `Message`.
- `res.text` 是 JSON 字符串.
- 如果需要强类型对象, 使用 `res.parse_as(YourSchema)`.

示例:

```python
from pydantic import BaseModel

from core.llm import GeminiLLM
from core.schemas import Message


class ProductInfo(BaseModel):
    name: str
    category: str
    selling_points: list[str]


llm = GeminiLLM(model="gemini-3-pro-preview")

res = llm.generate_struct(
    messages=Message(content="为一款香蕉味能量饮料生成商品信息"),
    schema=ProductInfo,
    system_prompt="只返回符合 schema 的 JSON",
)

info = res.parse_as(ProductInfo)

print(res.text)
print(info.name)
print(info.selling_points)
```

适合使用结构化生成的场景:

- 后续代码需要读取固定字段.
- 需要把模型输出保存到数据库.
- 需要减少手写 `json.loads()` 和字段判断.

## 多模态输入

如果请求中包含图片, 使用 `ImageContent` 和 `TextContent` 组成多模态 `Message`.

示例:

```python
from core.llm import GeminiLLM
from core.schemas import ImageContent, Message, TextContent

llm = GeminiLLM(model="gemini-3-pro-preview")

image = ImageContent.from_file("workspace/input.png")
message = Message(
    content=[
        TextContent(text="描述这张图片中的主体, 背景和风格"),
        image,
    ]
)

res = llm.generate(message)
print(res.text)
```

注意事项:

- 纯文本消息可以直接使用 `Message(content="...")`.
- 图文混合消息需要使用 `TextContent(text="...")` 放文本, 使用 `ImageContent.from_file(...)` 放图片.

## 图像编辑

接口:

```python
edit_image(
    messages: Message | list[Message],
    system_prompt: str | None = None,
    config: EditImageConfig | None = None,
) -> Message
```

参数说明:

- `messages`: 必须包含至少一张图片.
- `system_prompt`: 可选的全局图像编辑要求.
- `config`: 图像生成或编辑配置.

返回值:

- 如果模型返回图片, 通过 `res.images` 获取.
- 如果 OpenAI 兼容接口返回 URL, URL 会放在 `res.text` 中.

Gemini 示例:

```python
from core.llm import GeminiLLM
from core.schemas import ImageContent, Message, TextContent

llm = GeminiLLM(model="gemini-3-pro-image-preview")

image = ImageContent.from_file("workspace/input.png")
config = GeminiLLM.EditImageConfig(
    aspect_ratio="1:1",
    image_size="1K",
    output_mime_type="image/png",
)

res = llm.edit_image(
    messages=Message(
        content=[
            TextContent(text="把图片改成手绘铅笔素描风格"),
            image,
        ]
    ),
    config=config,
)

if res.images:
    res.images[0].save_to_file("workspace/output.png")
```

OpenAI 示例:

```python
from core.llm import OpenAILLM
from core.schemas import ImageContent, Message, TextContent

llm = OpenAILLM(model="gpt-image-2")

image = ImageContent.from_file("workspace/input.png")
config = OpenAILLM.EditImageConfig(
    input_fidelity="high",
    output_format="png",
    quality="high",
)

res = llm.edit_image(
    messages=Message(
        content=[
            TextContent(text="移除背景, 保留主体"),
            image,
        ]
    ),
    config=config,
)

if res.images:
    res.images[0].save_to_file("workspace/output.png")
else:
    print(res.text)
```

## 常用生成配置

### GeminiLLM.GenerateConfig

常用字段:

- `temperature`: 输出随机性. 越低越稳定, 越高越发散.
- `top_p`: 采样范围. 通常和 `temperature` 搭配使用.
- `max_output_tokens`: 最大输出长度.
- `seed`: 随机种子. 用于尽量稳定结果.
- `think_level`: 模型思考深度, 只对支持该字段的模型有效.

示例:

```python
from core.llm import GeminiLLM
from core.schemas import Message

llm = GeminiLLM(model="gemini-3-pro-preview")
config = GeminiLLM.GenerateConfig(temperature=0, top_p=0.1, seed=39)

res = llm.generate(Message(content="生成一句稳定的商品标题"), config=config)
print(res.text)
```

### OpenAILLM.GenerateConfig

常用字段:

- `temperature`: 输出随机性.
- `top_p`: 采样范围.
- `seed`: 随机种子.
- `max_completion_tokens`: 最大输出 token 数.
- `verbosity`: 输出详细程度, 可选 `low`, `medium`, `high`.

示例:

```python
from core.llm import OpenAILLM
from core.schemas import Message

llm = OpenAILLM(model="kimi-k2.5")
config = OpenAILLM.GenerateConfig(temperature=0.2, top_p=0.9, seed=39)

res = llm.generate(Message(content="写一个简短广告语"), config=config)
print(res.text)
```

## 推荐使用方式

- 普通业务优先使用 `generate()` 和 `generate_struct()`.
- 需要读取字段时, 优先使用 `generate_struct()` 而不是让模型自由输出文本.
- prompt 较长或多人维护时, 把 prompt 放到 Markdown 文件中, 配合 `PromptLib` 使用.
- 不要在业务代码中依赖 `core.llm` 的内部实现模块.
