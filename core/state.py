"""状态管理, 提供 MessagePack 二进制持久化能力.

使用 MessagePack 代替 YAML/JSON, 原生支持 ``bytes`` 字段,
避免 base64 编解码带来的开销和数据膨胀.
"""

from pathlib import Path
from typing import Self

import msgpack
from pydantic import BaseModel


class BaseState(BaseModel):
    """为 Pydantic v2 模型提供 MessagePack 文件持久化能力的基类.

    继承此类即可获得 ``save`` 和 ``load`` 方法, 用于将模型状态
    以 MessagePack 二进制格式读写到磁盘. 原生支持 ``bytes`` 字段,
    适合包含 ``ImageContent.source`` 等二进制数据的模型.
    """

    def save(self, path: str | Path) -> Path:
        """将模型状态序列化为 MessagePack 文件.

        使用 ``model_dump()`` 获取 Python 原生数据, 再由 ``msgpack``
        打包为二进制格式写入磁盘. 自动创建目标路径上的父目录.

        Args:
            path: 目标文件路径, 建议使用 ``.msgpack`` 后缀.

        Returns:
            Path: 保存后的文件路径.

        Raises:
            OSError: 当文件无法写入时 (如权限不足或路径非法).
        """
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        data = self.model_dump()
        with open(output, "wb") as f:
            msgpack.pack(data, f)
        return output

    @classmethod
    def load(cls, path: str | Path) -> Self:
        """从 MessagePack 文件恢复模型实例.

        读取二进制文件, 由 ``msgpack`` 解包后调用 ``model_validate()``
        重建模型实例.

        Args:
            path: MessagePack 文件路径.

        Returns:
            Self: 从文件内容恢复的模型实例.

        Raises:
            FileNotFoundError: 当文件不存在时.
            msgpack.UnpackException: 当文件不是合法的 MessagePack 数据时.
            pydantic.ValidationError: 当数据不符合模型 schema 时.
        """
        source = Path(path)
        with open(source, "rb") as f:
            data = msgpack.unpack(f)
        return cls.model_validate(data)
