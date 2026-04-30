"""状态管理, 提供 YAML 文件持久化能力."""

from pathlib import Path
from typing import Self

import yaml
from pydantic import BaseModel


class State(BaseModel):
    """为 Pydantic v2 模型提供 YAML 文件持久化能力的 Mixin.

    继承此类即可获得 ``save`` 和 ``load`` 方法, 用于将模型状态读写到磁盘.

    Example:
        >>> class MyConfig(PersistableModel):
        ...     name: str
        ...     value: int
        >>> config = MyConfig(name="test", value=42)
        >>> config.save("/tmp/config.yaml")
        PosixPath('/tmp/config.yaml')
        >>> loaded = MyConfig.load("/tmp/config.yaml")
        >>> loaded == config
        True
    """

    def save(self, path: str | Path) -> Path:
        """将模型状态保存为 YAML 文件.

        使用 ``yaml.dump`` 序列化, 自动创建父目录.

        Args:
            path: 目标文件路径.

        Returns:
            Path: 保存后的文件路径.

        Raises:
            OSError: 当文件无法写入时.
        """
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        data = self.model_dump(mode="json")
        with open(output, "w", encoding="utf-8") as f:
            yaml.dump(
                data,
                f,
                allow_unicode=True,
                default_flow_style=False,
                sort_keys=False,
            )
        return output

    @classmethod
    def load(cls, path: str | Path) -> Self:
        """从 YAML 文件恢复模型实例.

        Args:
            path: YAML 文件路径.

        Returns:
            Self: 从文件内容恢复的模型实例.

        Raises:
            FileNotFoundError: 当文件不存在时.
            yaml.YAMLError: 当文件不是合法 YAML 时.
            pydantic.ValidationError: 当 YAML 内容不符合模型 schema 时.
        """
        source = Path(path)
        with open(source, encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return cls.model_validate(data)
