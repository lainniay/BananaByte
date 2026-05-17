# core.state 使用说明

`core.state` 提供基于 MessagePack 的状态持久化能力. 它适合在较长业务流程中保存中间状态, 以便网络失败, LLM 调用失败或程序中断后恢复.

如果一个流程只包含一两步, 通常不需要使用 `BaseState`. 如果流程包含多次 LLM 调用, 图像生成, 外部接口请求或昂贵中间结果, 建议使用它保存 checkpoint.

## BaseState

接口:

```python
class BaseState(BaseModel):
    def save(self, path: str | Path) -> Path: ...

    @classmethod
    def load(cls, path: str | Path) -> Self: ...
```

说明:

- `BaseState` 继承自 Pydantic `BaseModel`.
- 子类可以定义自己的状态字段.
- `save()` 使用 `model_dump()` 得到 Python 原生数据, 然后写入 MessagePack 文件.
- `load()` 从 MessagePack 文件读取数据, 然后通过 `model_validate()` 恢复实例.
- MessagePack 原生支持 `bytes`, 适合保存图片二进制数据或其他中间产物.

使用方式是先定义一个继承 `BaseState` 的业务状态类, 然后在关键步骤前后调用 `save()`.

## save

接口:

```python
save(path: str | Path) -> Path
```

说明:

- 将当前状态保存到文件.
- 如果父目录不存在, 会自动创建.
- 建议使用 `.msgpack` 后缀.
- 文件无法写入时会抛出 `OSError`.

## load

接口:

```python
load(path: str | Path) -> Self
```

说明:

- 从 MessagePack 文件恢复状态.
- 文件不存在时会抛出 `FileNotFoundError`.
- 文件内容不是合法 MessagePack 时会抛出 `msgpack.UnpackException`.
- 文件数据不符合状态模型时会抛出 `pydantic.ValidationError`.

## 基础示例

```python
from core.state import BaseState


class AnalyzeState(BaseState):
    run_id: str
    step: str
    prompt: str
    result_json: str | None = None


state = AnalyzeState(
    run_id="run_001",
    step="analyze",
    prompt="Analyze this image",
)

path = state.save("workspace/checkpoints/run_001/analyze.msgpack")
restored = AnalyzeState.load(path)

print(restored.step)
```

## 保存 bytes 示例

```python
from core.schemas import ImageContent
from core.state import BaseState


class ImageState(BaseState):
    run_id: str
    image_bytes: bytes
    mime_type: str


image = ImageContent.from_file("workspace/input.png")

state = ImageState(
    run_id="run_001",
    image_bytes=image.source,
    mime_type=image.mime_type,
)

state.save("workspace/checkpoints/run_001/image.msgpack")
restored = ImageState.load("workspace/checkpoints/run_001/image.msgpack")

restored_image = ImageContent(
    source=restored.image_bytes,
    mime_type=restored.mime_type,
)
```

## 推荐保存时机

建议在容易失败或耗时较长的步骤前后保存状态:

- LLM 文本生成前后.
- LLM 图像编辑前后.
- 外部网络请求前后.
- 写入重要中间结果后.
- 每个耗时或不可重复步骤完成后.

示例:

```python
from core.llm import GeminiLLM
from core.schemas import Message
from core.state import BaseState


class RunState(BaseState):
    run_id: str
    prompt: str
    answer: str | None = None


state = RunState(run_id="run_001", prompt="hello")
state.save("workspace/checkpoints/run_001/before_llm.msgpack")

llm = GeminiLLM(model="gemini-3-pro-preview")
res = llm.generate(Message(content=state.prompt))

state.answer = res.text
state.save("workspace/checkpoints/run_001/after_llm.msgpack")
```

## 路径命名建议

建议按 `run_id` 和步骤名组织 checkpoint:

```text
workspace/checkpoints/
  run_001/
    01_before_analyze.msgpack
    02_after_analyze.msgpack
    03_after_edit_image.msgpack
```

这样恢复时可以清楚知道流程停在哪一步.
