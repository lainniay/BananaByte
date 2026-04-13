import re
from pathlib import Path
from typing import Self

from pydantic import BaseModel

_VAR_RE = re.compile(r"\{([a-zA-Z_]\w*)\}")


class Prompt(BaseModel):
    """Prompt 模型."""

    name: str
    description: str = ""
    version: str = ""
    template: str

    @property
    def variables(self) -> list[str]:
        """提取模板中定义的变量名.

        Returns:
            list[str]: 按出现顺序去重后的变量名列表.
        """
        seen: set[str] = set()
        variables: list[str] = []
        for match in _VAR_RE.finditer(self.template):
            field_name = match.group(1)
            if field_name in seen:
                continue
            seen.add(field_name)
            variables.append(field_name)
        return variables

    def render(self, **kwargs: str) -> str:
        """渲染模板.

        Args:
            **kwargs: 模板变量.

        Returns:
            str: 渲染后的提示词文本.

        Raises:
            ValueError: 当缺少模板所需变量时.
        """
        required = self.variables
        required_set = set(required)
        provided_set = set(kwargs.keys())
        missing = [name for name in required if name not in provided_set]
        if missing:
            raise ValueError(
                f"缺少必要的模板变量: {missing}. 该模板需要变量: {required}"
            )

        payload = {key: value for key, value in kwargs.items() if key in required_set}

        def _replace(match: re.Match[str]) -> str:
            field_name = match.group(1)
            return payload.get(field_name, match.group(0))

        return _VAR_RE.sub(_replace, self.template)

    @classmethod
    def from_file(cls, path: str | Path) -> Self:
        """从 Markdown 文件加载 Prompt.

        支持 YAML front matter, 格式如下:
            ---
            name: demo
            description: demo prompt
            version: 1.0
            ---

        Args:
            path: Markdown 文件路径.

        Returns:
            Self: Prompt 实例.

        Raises:
            ValueError: 当 front matter 格式非法或缺少 pyyaml 时.
        """
        file_path = Path(path)
        content = file_path.read_text(encoding="utf-8")

        name = file_path.stem
        description = ""
        version = ""
        template = content

        match = re.match(r"^---\s*\n(.*?)\n---\s*\n?(.*)$", content, flags=re.DOTALL)
        if match:
            front_matter_text, body = match.groups()
            try:
                import yaml
            except ModuleNotFoundError as error:
                raise ValueError("缺少 pyyaml 依赖, 请先安装 pyyaml") from error

            metadata = yaml.safe_load(front_matter_text) or {}
            if not isinstance(metadata, dict):
                raise ValueError("front matter 必须是 YAML 对象")

            raw_name = metadata.get("name")
            if raw_name is not None:
                name = str(raw_name)
            raw_description = metadata.get("description")
            if raw_description is not None:
                description = str(raw_description)
            raw_version = metadata.get("version")
            if raw_version is not None:
                version = str(raw_version)
            template = body

        return cls(
            name=name,
            description=description,
            version=version,
            template=template,
        )

    def __str__(self) -> str:
        """返回模板原始文本."""
        return self.template


class PromptLib:
    """Prompt 库, 用于管理目录中的 Markdown 提示词文件."""

    def __init__(self, directory: str | Path) -> None:
        """初始化 Prompt 库并加载目录下所有 .md 文件.

        Args:
            directory: Prompt 文件目录路径.
        """
        self.directory = Path(directory)
        if not self.directory.exists():
            raise ValueError(f"目录不存在: {self.directory}")
        if not self.directory.is_dir():
            raise ValueError(f"路径不是目录: {self.directory}")

        self._prompts: dict[str, Prompt] = {}
        for file_path in sorted(self.directory.glob("*.md")):
            prompt = Prompt.from_file(file_path)
            self._prompts[prompt.name] = prompt

    def get(self, name: str) -> Prompt:
        """按名称获取 Prompt.

        Args:
            name: Prompt 名称.

        Returns:
            Prompt: 对应名称的 Prompt.

        Raises:
            KeyError: 当 Prompt 不存在时.
        """
        if name not in self._prompts:
            raise KeyError(f"未找到 Prompt: {name}")
        return self._prompts[name]

    def list_prompts(self) -> list[str]:
        """列出所有可用 Prompt 名称."""
        return list(self._prompts.keys())

    def __getitem__(self, name: str) -> Prompt:
        """支持通过下标访问 Prompt."""
        return self.get(name)
