# core.llm.tool 使用说明

`core.llm.tool` 用于把普通 Python 函数包装成 LLM 可以调用的工具. 业务代码通常不需要使用工具调用. 只有当模型需要在生成过程中主动调用业务函数时, 才需要阅读本文档.

推荐从 `core.llm` 导入公开对象:

```python
from core.llm import Tool, tool
```

不要从 `core.llm.tool` 中导入以下划线开头的内部函数.

## tool 装饰器

接口:

```python
tool(
    *,
    description: str,
    name: str | None = None,
) -> Callable[[Callable[..., ReturnType]], Tool[ReturnType]]
```

参数说明:

- `description`: 工具描述. 会暴露给模型, 用于告诉模型什么时候应该调用这个工具.
- `name`: 工具名称. 不传时使用函数名.

返回值:

- 返回一个装饰器.
- 被装饰的函数会变成 `Tool` 对象.

约束:

- `description` 不能为空.
- 每个参数都必须有类型标注.
- 每个参数都必须使用 `Annotated[..., Field(description="...")]`.
- 不支持 positional-only 参数.
- 不支持 `*args` 和 `**kwargs`.

## 最小示例

```python
from typing import Annotated

from pydantic import Field

from core.llm import tool


@tool(description="Get weather by city")
def get_weather(city: Annotated[str, Field(description="City name")]) -> str:
    return f"{city}: sunny"
```

这段代码会创建一个名为 `get_weather` 的工具. 模型看到工具描述和参数描述后, 可以决定是否调用它.

## 自定义工具名称

如果函数名不适合直接暴露给模型, 可以使用 `name` 指定工具名.

```python
from typing import Annotated

from pydantic import Field

from core.llm import tool


@tool(description="Get product price by product id", name="get_product_price")
def price(product_id: Annotated[str, Field(description="Product id")]) -> float:
    return 19.9
```

## 参数写法

每个工具参数都要写成 `Annotated[类型, Field(description="参数说明")]`.

正确示例:

```python
from typing import Annotated

from pydantic import Field

from core.llm import tool


@tool(description="Calculate discount price")
def calculate_discount(
    price: Annotated[float, Field(description="Original price")],
    discount: Annotated[float, Field(description="Discount ratio, from 0 to 1")],
) -> float:
    return price * discount
```

错误示例:

```python
from core.llm import tool


@tool(description="Calculate discount price")
def calculate_discount(price: float, discount: float) -> float:
    return price * discount
```

上面的写法会报错, 因为参数缺少 `Field(description=...)`.

## 可选参数

工具函数可以使用默认值表示可选参数.

```python
from typing import Annotated

from pydantic import Field

from core.llm import tool


@tool(description="Search products by keyword")
def search_products(
    keyword: Annotated[str, Field(description="Search keyword")],
    limit: Annotated[int, Field(description="Max result count")] = 5,
) -> list[str]:
    return [f"{keyword}-{index}" for index in range(limit)]
```

## 直接调用工具

`Tool` 对象也可以像普通函数一样调用. 调用时会先用 Pydantic 校验参数, 然后执行原函数.

```python
from typing import Annotated

from pydantic import Field

from core.llm import tool


@tool(description="Add two numbers")
def add(
    a: Annotated[int, Field(description="First number")],
    b: Annotated[int, Field(description="Second number")],
) -> int:
    return a + b


result = add(a=1, b=2)
print(result)
```

如果参数类型不符合要求, 会抛出 Pydantic 校验错误.

## 配合 generate_with_tool 使用

`generate_with_tool()` 会让模型在需要时调用工具, 然后根据工具结果生成最终回复.

```python
from typing import Annotated

from pydantic import Field

from core.llm import OpenAILLM, tool
from core.schemas import Message


@tool(description="Get weather by city")
def get_weather(city: Annotated[str, Field(description="City name")]) -> str:
    return f"{city}: sunny"


llm = OpenAILLM(model="kimi-k2.5")

res = llm.generate_with_tool(
    messages=Message(content="今天 Tokyo 天气怎么样"),
    tools=[get_weather],
    system_prompt="你可以在需要时调用工具回答问题",
)

print(res.text)
```

参数说明:

- `tools`: 传入 `Tool` 对象列表.
- `max_tool_rounds`: 最大工具调用轮数, 默认是 `5`, 用于避免无限循环.

注意事项:

- 工具名称不能重复.
- `tools` 不能为空.
- 工具执行异常会被包装成工具结果返回给模型, 模型可能继续生成解释.

## 配合 generate_struct_with_tool 使用

如果最终结果需要是 JSON, 使用 `generate_struct_with_tool()`.

```python
from typing import Annotated

from pydantic import BaseModel, Field

from core.llm import OpenAILLM, tool
from core.schemas import Message


class WeatherAnswer(BaseModel):
    city: str
    weather: str


@tool(description="Get weather by city")
def get_weather(city: Annotated[str, Field(description="City name")]) -> str:
    return f"{city}: sunny"


llm = OpenAILLM(model="kimi-k2.5")

res = llm.generate_struct_with_tool(
    messages=Message(content="查询 Tokyo 天气, 并返回 JSON"),
    schema=WeatherAnswer,
    tools=[get_weather],
)

answer = res.parse_as(WeatherAnswer)
print(answer.city)
print(answer.weather)
```

## Tool 对象

接口字段:

```python
Tool(
    name: str,
    description: str,
    parameters: type[BaseModel],
    fn: Callable[..., ReturnType],
)
```

字段说明:

- `name`: 工具名称.
- `description`: 工具描述.
- `parameters`: 根据函数参数自动生成的 Pydantic 模型.
- `fn`: 原始 Python 函数.

业务代码通常不需要手动创建 `Tool`, 使用 `@tool(...)` 即可.

## 常见错误

### description 为空

```python
@tool(description="")
def demo() -> str:
    return "ok"
```

会抛出 `ValueError`.

### 参数缺少 Annotated

```python
@tool(description="Demo")
def demo(name: str) -> str:
    return name
```

会抛出 `TypeError`.

### 参数缺少 Field description

```python
from typing import Annotated

from pydantic import Field


@tool(description="Demo")
def demo(name: Annotated[str, Field()]) -> str:
    return name
```

会抛出 `TypeError`.

### 使用 args 或 kwargs

```python
@tool(description="Demo")
def demo(*args: str) -> str:
    return "ok"
```

会抛出 `TypeError`.

## 使用建议

- 工具函数保持简单, 输入输出尽量明确.
- 工具描述要写清楚什么时候调用, 不要只写函数名.
- 参数描述要写清楚单位, 格式和取值范围.
