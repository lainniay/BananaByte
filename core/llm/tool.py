import inspect
from collections.abc import Callable
from dataclasses import dataclass
from typing import Annotated, Any, cast, get_args, get_origin, get_type_hints

from pydantic import BaseModel, create_model
from pydantic.fields import FieldInfo


def _ensure_annotated_field(param_name: str, annotation: Any) -> None:
    """确保工具参数使用 Annotated[..., Field(description=...)] 标注.

    Args:
        param_name: 参数名称, 用于生成错误信息.
        annotation: 参数类型标注.

    Raises:
        TypeError: 当参数没有使用 Annotated 或缺少 Field description 时.
    """
    if get_origin(annotation) is not Annotated:
        raise TypeError(
            f"tool 参数 '{param_name}' 必须使用 Annotated[..., Field(description=...)]"
        )

    metadata = get_args(annotation)[1:]
    field_infos = [item for item in metadata if isinstance(item, FieldInfo)]
    if not any(info.description and info.description.strip() for info in field_infos):
        raise TypeError(f"tool 参数 '{param_name}' 必须包含 Field(description=...)")


def _create_parameters_model(fn: Callable[..., Any]) -> type[BaseModel]:
    """根据函数签名创建 Pydantic 参数模型.

    Args:
        fn: 被转换为工具的 Python 函数.

    Returns:
        根据函数参数动态创建的 Pydantic 模型类型.

    Raises:
        TypeError: 当函数参数不适合作为 LLM 工具参数时.
    """
    signature = inspect.signature(fn)
    type_hints = get_type_hints(fn, include_extras=True)
    fields: dict[str, Any] = {}

    for param_name, param in signature.parameters.items():
        if param.kind is inspect.Parameter.POSITIONAL_ONLY:
            raise TypeError(f"tool 参数 '{param_name}' 不能是 positional-only")
        if param.kind in (
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        ):
            raise TypeError("tool 不支持 *args 或 **kwargs")

        annotation = type_hints.get(param_name)
        if annotation is None:
            raise TypeError(f"tool 参数 '{param_name}' 缺少类型标注")
        _ensure_annotated_field(param_name, annotation)

        default = ... if param.default is inspect.Parameter.empty else param.default
        fields[param_name] = (annotation, default)

    model_name = f"{fn.__name__.title().replace('_', '')}Parameters"
    return cast(
        type[BaseModel],
        create_model(model_name, __module__=fn.__module__, **fields),
    )


def tool[ReturnType](
    *,
    description: str,
    name: str | None = None,
) -> Callable[[Callable[..., ReturnType]], "Tool[ReturnType]"]:
    """创建将函数转换为 Tool 的装饰器.

    被装饰函数的每个参数都必须使用
    ``Annotated[..., Field(description=...)]`` 标注, 以保证生成的工具
    schema 包含清晰的参数描述.

    Args:
        description: 工具描述, 会暴露给 LLM 作为 function description.
        name: 工具名称. 如果为 None, 使用函数名.

    Returns:
        接收函数并返回 Tool 实例的装饰器.

    Raises:
        ValueError: 当 description 为空时.
    """
    if not description.strip():
        raise ValueError("tool description 不能为空")

    def decorator(fn: Callable[..., ReturnType]) -> Tool[ReturnType]:
        """将函数包装为 Tool 实例.

        Args:
            fn: 需要暴露给 LLM 调用的函数.

        Returns:
            根据函数签名和装饰器参数创建的 Tool 实例.
        """
        parameters = _create_parameters_model(fn)
        return Tool(
            name=name or fn.__name__,
            description=description,
            parameters=parameters,
            fn=fn,
        )

    return decorator


@dataclass
class Tool[ReturnType]:
    """LLM 可调用工具的统一描述.

    Attributes:
        name: 工具名称, 会暴露给模型作为 function name.
        description: 工具描述, 会暴露给模型作为 function description.
        parameters: 工具参数的 Pydantic 模型类型.
        fn: 实际被调用的 Python 函数.
    """

    name: str
    description: str
    parameters: type[BaseModel]
    fn: Callable[..., ReturnType]

    def __call__(self, **kwds: Any) -> ReturnType:
        """校验参数并调用底层函数.

        Args:
            **kwds: 模型传入的工具调用参数.

        Returns:
            底层函数的返回值.

        Raises:
            ValidationError: 当参数无法通过 Pydantic 校验时.
            Exception: 底层函数执行时抛出的任意异常.
        """
        validated = self.parameters(**kwds)
        return self.fn(**validated.model_dump(exclude_none=True))
