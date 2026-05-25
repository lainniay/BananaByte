import subprocess
from enum import Enum
from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict

from agents.zjc.v3.tool import adjust_rgb_blance, get_color_mean, mime_to_ext
from core import BaseState, GeminiLLM, ImageContent, Message, PromptLib, TextContent


class State(Enum):  # noqa : D101
    ANALYZE = "analyze"
    EDIT_COLOR = "edit_color"
    EDIT_SHAPE = "edit_shape"
    EVALUATE = "evaluate"
    DONE = "done"


@runtime_checkable
class Analyzer(Protocol):  # noqa : D101
    def generate_struct(  # noqa : D102
        self,
        messages: Message | list[Message],
        schema: type[BaseModel],
        system_prompt: str | None = None,
        config: Any | None = None,
    ) -> Message: ...


@runtime_checkable
class Editor(Protocol):  # noqa : D101
    def edit_image(  # noqa : D102
        self,
        messages: Message | list[Message],
        system_prompt: str | None = None,
        config: Any | None = None,
    ) -> Message: ...


class Content(BaseState):  # noqa : D101
    model_config = ConfigDict(arbitrary_types_allowed=True)

    origin_img: ImageContent | None = None
    cur_img: ImageContent | None = None

    cur_state: State = State.ANALYZE
    cur_round: int = 1

    prompt_lib: PromptLib

    analyzer: Analyzer
    editor: Editor

    output_dir: str
    input_path: str


def run(ctx: Content) -> None:  # noqa : D103
    handlers = {
        State.ANALYZE: handle_analyze,
        State.EDIT_SHAPE: handle_edit_shape,
        State.EDIT_COLOR: handle_edit_color,
        State.EVALUATE: handle_evaluate,
    }
    ctx.origin_img = ImageContent.from_file(ctx.input_path)

    while ctx.cur_state is not State.DONE:
        handler = handlers.get(ctx.cur_state)
        if handler is None:
            raise ValueError(f"unknown state: {ctx.cur_state}")
        ctx.cur_state = handler(ctx)

    handle_done(ctx)


def handle_analyze(ctx: Content) -> State:  # noqa : D103
    if ctx.origin_img is None and ctx.cur_img is None:
        raise ValueError("no image for analyze stage")

    if ctx.origin_img is None:
        ctx.origin_img = ctx.cur_img
    if ctx.cur_img is None:
        ctx.cur_img = ctx.origin_img

    return State.EDIT_COLOR


def handle_edit_shape(ctx: Content) -> State:  # noqa : D103
    if ctx.cur_img is None:
        raise ValueError("no image for edit_shape stage")

    edit_input = Message(
        content=[TextContent(text=ctx.prompt_lib["blur"].render()), ctx.cur_img],
    )
    res = ctx.editor.edit_image(edit_input)
    if not res.images:
        raise ValueError("edit_shape stage returned no image")

    ctx.cur_img = res.images[0]
    return State.EVALUATE


def handle_edit_color(ctx: Content) -> State:  # noqa : D103
    if ctx.cur_img is None:
        raise ValueError("no image for edit_color stage")

    ext = _image_ext(ctx.cur_img)
    preview_path = f"{ctx.output_dir}/round_{ctx.cur_round}_color_input{ext}"
    ctx.cur_img.save_to_file(preview_path)
    print(f"[edit_color] current image saved: {preview_path}")
    subprocess.run(["open", str(preview_path)], check=False)

    b_mean, g_mean, r_mean = get_color_mean(ctx.cur_img)
    print(
        f"[edit_color] color mean (R, G, B): {r_mean:.2f}, {g_mean:.2f}, {b_mean:.2f}",
    )

    raw = input(
        "[edit_color] input gains in R G B order: r_gain g_gain b_gain "
        "(default: 1.0 1.0 1.0): ",
    ).strip()
    if raw:
        parts = raw.replace(",", " ").split()
        if len(parts) != 3:
            raise ValueError("color gains must be three numbers: r_gain g_gain b_gain")
        r_gain, g_gain, b_gain = (float(part) for part in parts)
    else:
        r_gain = g_gain = b_gain = 1.0

    fixed = adjust_rgb_blance(
        ctx.cur_img,
        r_gain=r_gain,
        g_gain=g_gain,
        b_gain=b_gain,
    )
    if fixed is None:
        raise ValueError("edit_color stage failed to adjust image color")

    ctx.cur_img = fixed
    b_mean, g_mean, r_mean = get_color_mean(ctx.cur_img)
    print(
        "[edit_color] adjusted color mean (R, G, B): "
        f"{r_mean:.2f}, {g_mean:.2f}, {b_mean:.2f}",
    )

    return State.EDIT_SHAPE


def handle_evaluate(ctx: Content) -> State:  # noqa : D103
    if ctx.cur_img is None:
        raise ValueError("no image for evaluate stage")

    ext = _image_ext(ctx.cur_img)
    eval_path = f"{ctx.output_dir}/round_{ctx.cur_round}_eval{ext}"
    ctx.cur_img.save_to_file(eval_path)
    print(f"[evaluate] round {ctx.cur_round} image saved: {eval_path}")
    subprocess.run(["open", str(eval_path)], check=False)

    raw = input("[evaluate] continue editing? [y/N]: ").strip().lower()
    if raw in {"y", "yes"}:
        ctx.cur_round += 1
        return State.EDIT_COLOR
    return State.DONE


def handle_done(ctx: Content) -> None:  # noqa : D103
    if ctx.cur_img is not None:
        ctx.cur_img.save_to_file(ctx.output_dir + "out.jpg")
    ctx.cur_state = State.DONE


def _image_ext(image: ImageContent) -> str:
    return mime_to_ext.get(image.mime_type, ".png")


def main() -> None:  # noqa: D103
    ctx = Content(
        analyzer=GeminiLLM("gemini-3.5-flash"),
        editor=GeminiLLM("gemini-3.1-flash-image-preview"),
        prompt_lib=PromptLib("./prompt/"),
        input_path="../../../workspace/Severe/2/out_lab.png",
        output_dir="../../../workspace/Severe/2/",
    )

    run(ctx)


if __name__ == "__main__":
    main()
