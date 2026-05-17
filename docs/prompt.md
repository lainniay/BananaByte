# core.prompt 使用说明

`core.prompt` 用于管理 Markdown prompt 文件, 并提供简单的 `{var}` 变量替换能力. 它适合把较长 prompt 从 Python 代码中拆出来, 方便多人维护和复用.

推荐用法:

1. 在 `prompts/` 目录中创建 `.md` 文件.
2. 用 `{变量名}` 标记运行时要填入的内容.
3. 在代码中用 `PromptLib` 加载目录.
4. 使用 `prompt_lib["name"].render(...)` 得到最终 prompt 文本.

## Prompt

接口:

```python
Prompt(
    name: str,
    description: str = "",
    version: str = "",
    template: str,
)
```

说明:

- `name`: prompt 名称.
- `description`: prompt 描述.
- `version`: prompt 版本.
- `template`: prompt 模板文本.

业务代码通常不需要手动构造 `Prompt`, 更常见的是通过 `Prompt.from_file()` 或 `PromptLib` 从文件加载.

### variables

接口:

```python
prompt.variables -> list[str]
```

说明:

- 返回模板中出现的变量名.
- 变量格式是 `{name}`.
- 返回结果按出现顺序去重.

示例:

```python
from core.prompt import Prompt

prompt = Prompt(
    name="intro",
    template="Write a short intro for {name} with style {style}",
)

print(prompt.variables)
```

### render

接口:

```python
render(**kwargs: str) -> str
```

说明:

- 使用传入变量渲染模板.
- 缺少必要变量时抛出 `ValueError`.
- 多余变量会被忽略.
- 变量替换只处理 `{name}` 这类简单占位符, 不支持复杂模板语法.

示例:

```python
from core.prompt import Prompt

prompt = Prompt(
    name="intro",
    template="Write a short intro for {name} with style {style}",
)

text = prompt.render(name="BananaByte", style="friendly")
print(text)
```

### from_file

接口:

```python
Prompt.from_file(path: str | Path) -> Prompt
```

说明:

- 从 Markdown 文件加载 prompt.
- 支持 YAML front matter.
- 如果没有 front matter, `name` 使用文件名 stem, `template` 使用完整文件内容.

Markdown 文件示例:

```markdown
---
name: analyze
description: Analyze an image
version: 1.0
---
Analyze this image with context: {context}
```

Python 使用示例:

```python
from core.prompt import Prompt

prompt = Prompt.from_file("prompts/analyze.md")
text = prompt.render(context="product photo")
```

## PromptLib

接口:

```python
PromptLib(directory: str | Path)
```

说明:

- 加载目录下所有 `.md` 文件.
- 每个文件会通过 `Prompt.from_file()` 解析.
- prompt 使用 `Prompt.name` 作为索引 key.
- 目录不存在或不是目录时抛出 `ValueError`.

建议每个业务流程使用一个 prompt 目录, 例如 `prompts/zjc/` 或 `prompts/analyze/`.

### get

接口:

```python
get(name: str) -> Prompt
```

说明:

- 按名称获取 prompt.
- 找不到时抛出 `KeyError`.

示例:

```python
from core.prompt import PromptLib

prompt_lib = PromptLib("prompts")
prompt = prompt_lib.get("analyze")
```

### list_prompts

接口:

```python
list_prompts() -> list[str]
```

说明:

- 返回当前目录中加载到的所有 prompt 名称.

示例:

```python
from core.prompt import PromptLib

prompt_lib = PromptLib("prompts")
print(prompt_lib.list_prompts())
```

### 下标访问

接口:

```python
prompt_lib[name] -> Prompt
```

说明:

- 等价于 `prompt_lib.get(name)`.

示例:

```python
from core.prompt import PromptLib

prompt_lib = PromptLib("prompts")

text = prompt_lib["analyze"].render(context="product photo")
```

## 完整示例

目录结构:

```text
prompts/
  analyze.md
  edit.md
```

代码:

```python
from core.llm import GeminiLLM
from core.prompt import PromptLib
from core.schemas import Message

prompt_lib = PromptLib("prompts")
llm = GeminiLLM(model="gemini-3-pro-preview")

prompt = prompt_lib["analyze"].render(context="product photo")
res = llm.generate(Message(content=prompt))

print(res.text)
```

## 文件组织建议

建议 prompt 文件名和 front matter 中的 `name` 保持一致:

```text
prompts/
  analyze.md
  rewrite.md
  edit_image.md
```

示例 `prompts/analyze.md`:

```markdown
---
name: analyze
description: Analyze product image
version: 1.0
---
请分析下面的商品图片.

业务背景: {context}
输出要求: {requirement}
```

对应 Python 代码:

```python
from core.prompt import PromptLib

prompt_lib = PromptLib("prompts")

prompt = prompt_lib["analyze"].render(
    context="电商商品主图",
    requirement="输出主体, 背景, 风格和可优化点",
)
```
