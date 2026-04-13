from pathlib import Path
from tempfile import TemporaryDirectory

from core.prompt import Prompt, PromptLib


def test_variables_and_render() -> None:
    prompt = Prompt(
        name="demo",
        template="你好 {name}, 你有 {count} 条消息。再次确认 {name}。",
    )

    assert prompt.variables == ["name", "count"]
    assert (
        prompt.render(name="Alice", count="3")
        == "你好 Alice, 你有 3 条消息。再次确认 Alice。"
    )

    # 多余变量应被忽略
    assert (
        prompt.render(name="Bob", count="5", unused="x")
        == "你好 Bob, 你有 5 条消息。再次确认 Bob。"
    )

    # 缺少变量应报错并包含变量提示
    try:
        prompt.render(name="Alice")
        raise AssertionError("缺少变量时应抛出 ValueError")
    except ValueError as error:
        msg = str(error)
        assert "缺少必要的模板变量" in msg
        assert "count" in msg


def test_from_file_without_front_matter() -> None:
    with TemporaryDirectory() as tmp_dir:
        path = Path(tmp_dir) / "plain.md"
        path.write_text("这是一个纯模板 {x}", encoding="utf-8")

        prompt = Prompt.from_file(path)

        assert prompt.name == "plain"
        assert prompt.description == ""
        assert prompt.version == ""
        assert prompt.template == "这是一个纯模板 {x}"


def test_from_file_with_front_matter() -> None:
    with TemporaryDirectory() as tmp_dir:
        path = Path(tmp_dir) / "with_meta.md"
        path.write_text(
            """---
name: custom_name
description: 用于摘要
version: 1.2
---
请总结以下内容:\n{content}
""",
            encoding="utf-8",
        )

        prompt = Prompt.from_file(path)

        assert prompt.name == "custom_name"
        assert prompt.description == "用于摘要"
        assert prompt.version == "1.2"
        assert prompt.template == "请总结以下内容:\n{content}\n"
        assert str(prompt) == prompt.template


def test_prompt_lib() -> None:
    with TemporaryDirectory() as tmp_dir:
        root = Path(tmp_dir)
        (root / "first.md").write_text("hello {name}", encoding="utf-8")
        (root / "second.md").write_text(
            """---
name: second_custom
description: second
version: 0.1
---
world {name}
""",
            encoding="utf-8",
        )

        lib = PromptLib(root)

        names = lib.list_prompts()
        assert "first" in names
        assert "second_custom" in names

        assert lib.get("first").render(name="A") == "hello A"
        assert lib["second_custom"].render(name="B") == "world B\n"

        try:
            lib.get("missing")
            raise AssertionError("不存在的 prompt 应抛出 KeyError")
        except KeyError:
            pass


def test_prompt_lib_invalid_directory() -> None:
    with TemporaryDirectory() as tmp_dir:
        missing = Path(tmp_dir) / "not-exists"
        try:
            PromptLib(missing)
            raise AssertionError("不存在目录应抛出 ValueError")
        except ValueError as error:
            assert "目录不存在" in str(error)


def test_json_content_not_template_variable() -> None:
    prompt = Prompt(
        name="json_demo",
        template='请输出 JSON: {"type": "summary", "user": "{name}", "count": 2}',
    )

    assert prompt.variables == ["name"]
    assert (
        prompt.render(name="Alice")
        == '请输出 JSON: {"type": "summary", "user": "Alice", "count": 2}'
    )


if __name__ == "__main__":
    test_variables_and_render()
    test_from_file_without_front_matter()
    test_from_file_with_front_matter()
    test_prompt_lib()
    test_prompt_lib_invalid_directory()
    test_json_content_not_template_variable()
    print("testPrompt passed")
