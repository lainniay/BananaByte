# core 包使用说明

`core` 包提供常用接口的顶层导出. 如果只是写业务流程, 可以从 `core` 直接导入常用对象. 如果希望代码来源更清晰, 也可以从具体模块导入.

导出对象:

```python
from core import (
    BaseState,
    GeminiLLM,
    ImageContent,
    Message,
    OpenAILLM,
    Prompt,
    PromptLib,
    TextContent,
    setup_rich_logging,
)
```

说明:

- `core` 在 import 时会调用 `load_dotenv()`, `.env` 中的环境变量会自动加载.
- `GeminiLLM` 和 `OpenAILLM` 来自 `core.llm`.
- `ImageContent`, `TextContent`, `Message` 来自 `core.schemas`.
- `Prompt` 和 `PromptLib` 来自 `core.prompt`.
- `BaseState` 来自 `core.state`.

推荐业务代码优先使用下面两种导入方式之一.

方式一, 从顶层导入常用对象:

```python
from core import GeminiLLM, ImageContent, Message, TextContent
```

方式二, 从具体模块导入, 适合较大的业务文件:

```python
from core.llm import GeminiLLM
from core.schemas import ImageContent, Message, TextContent
```

## setup_rich_logging

接口:

```python
setup_rich_logging(level: int = logging.WARNING) -> None
```

说明:

- 为 `core.llm` logger 添加 RichHandler.
- 用于美化 LLM 重试和错误日志.
- 建议在程序入口调用一次, 避免重复添加 handler.

示例:

```python
import logging

from core import OpenAILLM, setup_rich_logging
from core.schemas import Message

setup_rich_logging(level=logging.WARNING)

llm = OpenAILLM(model="kimi-k2.5")
res = llm.generate(Message(content="hello"))

print(res.text)
```

## 顶层导入示例

```python
from core import GeminiLLM, ImageContent, Message, TextContent

llm = GeminiLLM(model="gemini-3-pro-preview")
image = ImageContent.from_file("workspace/input.png")

res = llm.generate(
    Message(content=[TextContent(text="Describe this image"), image])
)

print(res.text)
```

## 何时使用具体模块导入

如果代码需要明确依赖来源, 推荐从具体模块导入:

```python
from core.llm import GeminiLLM
from core.prompt import PromptLib
from core.schemas import Message
from core.state import BaseState
```

这种方式在模块变多时更清晰.

不要从 `core.llm` 的内部实现文件导入对象. 业务代码只需要使用文档中列出的公开入口.
