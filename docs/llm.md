# llm.py 使用

llm.py 目前包含了两个 LLM 类:

1. OpenAILLM
2. GeminiLLM

> 目前 `creat_llm()` 已经移除, 需要直接使用 `OpenAILLM` 或者是 `GeminiLLM` 进行初始化.

OpenAILLM 默认读取 `OPENAI_API_KEY`, `OPENAI_API_BASE` 环境变量
GeminiLLM 默认使用 `GEMINI_API_KEY` 和 `GEMINI_API_BASE` 环境变量.

```python
# 使用默认环境变量
llm = OpenAILLM("kimi-k2.5")

# 使用自定义环境变量
llm = GeminiLLM("gemini-3.1-pro-preview", api_key=os.getenv("GEMINI_API"), base_url="os.getenv("GEMINI_BASE"))
```

可以通过 `max_retries: int` 参数来配置生成回复时重试的次数, 从而防止因为偶发的网络问题导致的 Agent 中断, `max_retries` 的默认参数为 3.

如果因为网络问题导致重试, 会打印出 WARNING 等级的日志, 可以使用 `core.setup_rich_logging` 来获得比较美观的打印输出, 只需要调用一次即可.

```python
# 使用 rich 库打印日志, 并将日志等级设置为 DEBUG
setup_rich_logging(level=logging.DEBUG)
```

## 输入输出接口

> 两种 LLM 的输入输出接口是一样的, 因此这里使用 OpenAILLM 作为例子.

目前有三种输入输出的接口:

1. `generate()` text/image -> text
2. `generate_struct()` text/image -> json
3. `edit_image()` text/image -> image

`generate()`大致的使用如下:

```python
import os

from core.llm import OpenAILLM
from core.schemas import Message

llm = OpenAILLM(
    model="kimi-k2.5",
    api_key=os.getenv("KIMI_API_KEY"),
    base_url=os.getenv("KIMI_API_BASE"),
)

res = llm.generate(Message(content="hello"), system_prompt="You are a cat girl")

print(res.text)
```

`generate_struct()` 需要使用 `pydantic.Basemodel` 来定义输出的 JSON 格式, 并没有给 JSON 提供专门的 Field
直接使用 `Message.text` 就可以得到 JSON 文本.

```python
import os

from pydantic import BaseModel
from rich import print_json

from core.llm import OpenAILLM
from core.schemas import Message

llm = OpenAILLM(
    model="kimi-k2.5",
    api_key=os.getenv("KIMI_API_KEY"),
    base_url=os.getenv("KIMI_API_BASE"),
)

# 定义模型回复的 JSON 格式
class ReturnSchema(BaseModel):
    name: str
    age: int
    words: str


res = llm.generate_struct(
    messages=Message(content="hello"),
    system_prompt=" you are a cat girl",
    schema=ReturnSchema,
)

# JSON 原始文本
json_text = res.text

# 使用 rich.print_json 可以获得比较美观的 json 打印
print_json(json_text)

# 使用 JSON 文本生成, ReturnSchema 对象
# 这一步会做检验, 检查生成的文本中是否缺少要求的 key.
answer: ReturnSchema = ReturnSchema.model_validate_json(res.text)

```

`edit_image` 的使用方法如下:

```python
from core.llm import OpenAILLM
from core.schemas import ImageContent, Message, TextContent

llm = OpenAILLM(model="gpt-image-2")


image = ImageContent.from_file("../workspace/test.jpg")
text = TextContent(text="修改图像的色调")


res = llm.edit_image(messages=Message(content=[image, text]))

# 提取第一张图片
res.images[0].save_to_file("../workspace/after.jpg")
```

## 参数配置

现在, 可以使用 `config` 参数来配置单次生成的参数, 参数的意义可以参考 `core/llm.py` 中的注释, 或者是自己查.

`generate` 和 `generate_struct` 使用相同的配置包 GenerateConfig, 而 `edit_image` 使用 EditImageConfig.

> 目前有些参数可能不够成熟, 比如说 tool 相关的参数, 以及用于控制生成回答数量的参数.
> 具体某些参数如何使用, 需要查询相关模型的 API 说明, 比如说 size 等.

下面是使用示例:

```python
from core.llm import OpenAILLM
from core.schemas import ImageContent, Message, TextContent

llm = OpenAILLM(model="gpt-image-2")


image = ImageContent.from_file("../workspace/test.jpg")
text = TextContent(text="修改图像的色调")


config = OpenAILLM.EditImageConfig(input_fidelity="high", background="transparent")


res = llm.edit_image(messages=Message(content=[image, text]), config=config)


res.images[0].save_to_file("../workspace/after.jpg")
```

```python

from pydantic import BaseModel
from rich import print_json

from core.llm import OpenAILLM
from core.schemas import Message

llm = OpenAILLM(model="gpt-image-2")


class ReturnSchema(BaseModel):
    name: str
    age: int
    words: str


config = llm.GenerateConfig(timeout=120, temperature=1.0, top_p=0.9, seed=100)


res = llm.generate_struct(
    messages=Message(content="hello"),
    system_prompt=" you are a cat girl",
    schema=ReturnSchema,
    config=config,
)

json_text = res.text

print_json(json_text)

answer: ReturnSchema = ReturnSchema.model_validate_json(res.text)
```
