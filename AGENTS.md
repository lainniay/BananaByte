# AgentFramework - Developer Guidelines for AI Coding Agents

This document provides essential information for AI coding agents working in this Python repository.

## Project Overview

**AgentFramework** is a Python 3.13+ library providing unified LLM interfaces for OpenAI and Google Gemini APIs.

**Key modules:**
- `core/schemas.py` - Pydantic models (Message, TextContent, ImageContent) with format adapters
- `core/llm.py` - LLM implementations (BaseLLM, OpenAILLM, GeminiLLM) and factory function
- `core/__init__.py` - Package exports

## Build, Test, and Lint Commands

### Running the Test Suite
```bash
# Run the integration test
python test/test.py

# Run a single test (no test framework configured)
python test/test.py
```

**Note:** This project currently has no pytest or unittest framework. Tests are simple integration scripts.

### Type Checking
```bash
# Run Pyright type checker
pyright

# Check a specific file
pyright core/llm.py
```

### Linting and Formatting
```bash
# Run Ruff linter
ruff check .

# Run Ruff linter with auto-fix
ruff check --fix .

# Run Ruff formatter
ruff format .

# Check a specific file
ruff check core/llm.py
```

### Package Management
```bash
# Install dependencies using uv
uv sync

# Add a new dependency
uv add package-name

# Update dependencies
uv lock --upgrade
```

### Virtual Environment
```bash
# Virtual environment is at .venv (managed by uv)
source .venv/bin/activate  # On Unix/macOS
```

## Code Style Guidelines

### Import Organization
Follow PEP 8 import order with three distinct groups:

```python
# 1. Standard library imports
from abc import ABC, abstractmethod
from typing import Literal, Self
import base64
import os

# 2. Third-party imports
from pydantic import BaseModel
from google.genai.types import HttpOptions
from litellm import completion

# 3. Local imports
from core.schemas import Message, ImageContent, TextContent
```

### Type Annotations

**Required:** Comprehensive type annotations on all functions, methods, and properties.

```python
# Function signatures with return types
def from_file(cls, path: str | Path) -> Self:
    ...

def generate(
    self,
    messages: list[Message],
    system_prompt: str | None = None,
    temperature: float = 0,
) -> Message:
    ...

# Properties must have return types
@property
def text(self) -> str:
    ...

# Use modern Python 3.10+ union syntax (| not Union)
content: str | list[TextContent | ImageContent]
api_key: str | None = None

# Use Literal for constrained string values
role: Literal["user", "model"]
provider: Literal["openai", "gemini"] | None = None

# Generic collections
messages: list[Message]
parts: list[TextContent | ImageContent]
```

**Type System:**
- Python 3.13 features supported
- Use `|` for unions (not `Union` from typing)
- Use `str | None` (not `Optional[str]`)
- Use `Self` type for builder methods returning class instance
- Use `Literal` for string enums
- Always annotate return types (including `-> None`)

### Naming Conventions

```python
# Classes: PascalCase
class ImageContent(BaseModel):
class OpenAILLM(BaseLLM):

# Functions and methods: snake_case
def create_llm(model: str) -> BaseLLM:
def to_gemini_format(self) -> dict:

# Variables: snake_case
mime_type = "image/png"
base_url = None

# Constants: UPPER_SNAKE_CASE (when used)
DEFAULT_TIMEOUT = 60

# Type aliases: PascalCase
ImageRatio = Literal["1:1", "16:9", ...]
```

### Error Handling

```python
# Early validation with descriptive ValueError messages
if not mime_type:
    raise ValueError(f"无法从{path}中读取文件类型")

if not self.api_key:
    raise ValueError("OPENAI_API_KEY 环境变量缺失")

# Environment variable loading with fallbacks
load_dotenv()
self.api_key = api_key or os.getenv("OPENAI_API_KEY", "")

# Type checking and safe conversions
if isinstance(res, ModelResponse):
    content = res.choices[0].message.content or ""
else:
    content = ""

# Graceful None handling with or operator
return Message(role="model", content=res.text or "")
```

**Error conventions:**
- Use `ValueError` for validation failures
- Error messages may be in Chinese or English
- No try-except wrappers unless necessary
- Let errors propagate naturally
- No logging framework in use

### Design Patterns

**Abstract Base Classes:**
```python
from abc import ABC, abstractmethod

class BaseLLM(ABC):
    @abstractmethod
    def generate(self, messages: list[Message], ...) -> Message:
        pass
```

**Factory Pattern:**
```python
def create_llm(
    model: str,
    provider: Literal["openai", "gemini"] | None = None,
    ...
) -> BaseLLM:
    # Auto-detect provider from model name if not specified
    if provider is None:
        provider = "gemini" if "gemini" in model.lower() else "openai"
    
    if provider == "gemini":
        return GeminiLLM(...)
    return OpenAILLM(...)
```

**Builder Pattern with Pydantic:**
```python
class ImageContent(BaseModel):
    @classmethod
    def from_base64(cls, data: str, mime_type: str = "image/png") -> Self:
        return cls(source=data, mime_type=mime_type)
    
    @classmethod
    def from_file(cls, path: str | Path) -> Self:
        # ... implementation
        return cls(source=data, mime_type=mime_type)
```

**Adapter Pattern:**
```python
class Message(BaseModel):
    def to_gemini_format(self) -> dict:
        # Convert to Gemini API format
        ...
    
    def to_openai_format(self) -> dict:
        # Convert to OpenAI API format
        ...
```

### Pydantic Models

```python
# Use Pydantic for data validation
class Message(BaseModel):
    role: Literal["user", "model"]
    content: str | list[TextContent | ImageContent]
    
    # Properties for computed values
    @property
    def text(self) -> str:
        if isinstance(self.content, str):
            return self.content
        texts = [item.text for item in self.content if isinstance(item, TextContent)]
        return " ".join(texts)
```

### Documentation

```python
# Module docstrings (bilingual supported)
"""
Core module for LLM framework

提供统一的 LLM 接口和消息模型
"""

# Function docstrings for complex functions
def create_llm(
    model: str,
    provider: Literal["openai", "gemini"] | None = None,
    ...
) -> BaseLLM:
    """
    统一的 LLM 工厂函数，根据模型名称或指定的提供商创建相应的 LLM 实例。
    
    Args:
        model: 模型名称（如 "gpt-4", "gemini-2.0-flash"）
        provider: 显式指定提供商，可选 "openai" 或 "gemini"
        
    Returns:
        BaseLLM: 相应的 LLM 实例
        
    Examples:
        llm = create_llm("gpt-4")
        llm = create_llm("gemini-2.0-flash", timeout=30)
    """
```

## Important Notes for Agents

1. **Python Version:** This project requires Python 3.13+. Use modern syntax features.

2. **No Test Framework:** Currently no pytest. Tests are manual scripts in `test/`. If adding tests, consider using pytest.

3. **Environment Variables:** Use `.env` file for API keys:
   - `OPENAI_API_KEY` - For OpenAI-compatible APIs
   - `OPENAI_API_BASE` - For custom base URLs

4. **Dependencies:** Use `uv` package manager, not pip. Lock file is `uv.lock`.

5. **Type Safety:** Pyright is configured. All code must pass type checking.

6. **Language:** Code comments and error messages may be in Chinese or English.

7. **No Pre-commit Hooks:** No automated checks on commit. Run tools manually.

8. **File Changes:** When modifying code:
   - Always preserve existing type annotations
   - Follow the adapter pattern for API conversions
   - Use Pydantic models for data structures
   - Keep error messages consistent with existing style
