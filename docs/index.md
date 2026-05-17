# Core 文档索引

本目录记录 `core` 内部框架的业务使用方式. 文档面向团队内部不会参与框架开发的成员, 重点说明如何调用接口完成业务任务.

## 阅读顺序

1. [`core.schemas`](./schema.md): 先了解 `Message`, `ImageContent`, `TextContent` 如何表示文本和图片.
2. [`core.llm`](./llm.md): 再了解如何调用模型完成文本生成, 结构化生成和图像编辑.
3. [`core.prompt`](./prompt.md): 如果 prompt 需要多人维护或复用, 阅读 prompt 文件管理方式.
4. [`core.state`](./state.md): 如果流程较长, 需要失败后恢复, 阅读状态保存方式.
5. [`core.llm.tool`](./tool.md): 如果需要让模型调用 Python 函数, 阅读工具调用方式.
6. [`core`](./core.md): 查看顶层导入和日志配置.

## 模块列表

- [`core.schemas`](./schema.md): 业务消息对象, 包括文本, 图片和多模态消息.
- [`core.llm`](./llm.md): 模型调用入口, 包括 `GeminiLLM` 和 `OpenAILLM`.
- [`core.llm.tool`](./tool.md): 工具调用说明, 包括 `tool` 和 `Tool`.
- [`core.prompt`](./prompt.md): Markdown prompt 加载和变量渲染.
- [`core.state`](./state.md): MessagePack 状态保存和恢复.
- [`core`](./core.md): 常用对象的顶层导出和 Rich 日志配置.

## 最小文本生成示例

```python
from core.llm import GeminiLLM
from core.schemas import Message

llm = GeminiLLM(model="gemini-3-pro-preview")
res = llm.generate(Message(content="hello"))

print(res.text)
```

## 最小结构化生成示例

```python
from pydantic import BaseModel

from core.llm import GeminiLLM
from core.schemas import Message


class Answer(BaseModel):
    title: str
    summary: str


llm = GeminiLLM(model="gemini-3-pro-preview")
res = llm.generate_struct(
    Message(content="为香蕉饮料写一个标题和简介"),
    schema=Answer,
)

answer = res.parse_as(Answer)
print(answer.title)
```
