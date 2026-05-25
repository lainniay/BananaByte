from typing import Any, Protocol, runtime_checkable

import cv2
import numpy as np
from pydantic import ConfigDict

from agents.zjc.v3.spilt_lab import lab_to_bgr, resize_channel
from agents.zjc.v3.tool import mime_to_ext
from core import BaseState, GeminiLLM, ImageContent, Message, PromptLib, TextContent


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
    deblur_img: ImageContent | None = None
    cur_img: ImageContent | None = None

    prompt_lib: PromptLib
    editor: Editor

    output_path: str
    deblur_output_path: str | None = None


def run(ctx: Content) -> None:  # noqa : D103
    handle_analyze(ctx)
    handle_deblur(ctx)
    handle_merge_lab(ctx)
    handle_done(ctx)


def handle_analyze(ctx: Content) -> None:  # noqa : D103
    if ctx.origin_img is None and ctx.cur_img is None:
        raise ValueError("no image for analyze stage")

    if ctx.origin_img is None:
        ctx.origin_img = ctx.cur_img
    if ctx.cur_img is None:
        ctx.cur_img = ctx.origin_img


def handle_deblur(ctx: Content) -> None:  # noqa : D103
    if ctx.cur_img is None:
        raise ValueError("no image for deblur stage")

    edit_input = Message(
        content=[TextContent(text=ctx.prompt_lib["blur"].render()), ctx.cur_img],
    )
    res = ctx.editor.edit_image(edit_input)
    if not res.images:
        raise ValueError("deblur stage returned no image")

    ctx.deblur_img = res.images[0]
    ctx.cur_img = ctx.deblur_img
    if ctx.deblur_output_path is not None:
        deblur_path = ctx.deblur_img.save_to_file(ctx.deblur_output_path)
        print(f"[deblur] image saved: {deblur_path}")


def handle_merge_lab(ctx: Content) -> None:  # noqa : D103
    if ctx.origin_img is None:
        raise ValueError("no original image for lab merge stage")
    if ctx.deblur_img is None:
        raise ValueError("no deblur image for lab merge stage")

    ctx.cur_img = merge_deblur_l_with_original_ab(ctx.origin_img, ctx.deblur_img)


def handle_done(ctx: Content) -> None:  # noqa : D103
    if ctx.cur_img is not None:
        output_path = ctx.cur_img.save_to_file(ctx.output_path)
        print(f"[done] final image saved: {output_path}")


def merge_deblur_l_with_original_ab(
    original: ImageContent,
    deblurred: ImageContent,
) -> ImageContent:
    """Use deblurred LAB-L with original LAB-a/b to create a new image."""
    original_bgr = _decode_bgr(original)
    deblurred_bgr = _decode_bgr(deblurred)

    original_lab = cv2.cvtColor(original_bgr, cv2.COLOR_BGR2LAB)
    deblurred_lab = cv2.cvtColor(deblurred_bgr, cv2.COLOR_BGR2LAB)

    l_channel = deblurred_lab[:, :, 0]
    height, width = l_channel.shape[:2]
    size = (width, height)

    a_channel = resize_channel(original_lab[:, :, 1], size)
    b_channel = resize_channel(original_lab[:, :, 2], size)
    merged_bgr = lab_to_bgr(l_channel, a_channel, b_channel)

    mime_type = original.mime_type if original.mime_type in mime_to_ext else "image/png"
    ext = mime_to_ext[mime_type]
    success, encoded_img = cv2.imencode(ext, merged_bgr)
    if not success:
        raise ValueError("lab merge image encode failed")

    return ImageContent(source=encoded_img.tobytes(), mime_type=mime_type)


def _decode_bgr(image: ImageContent) -> np.ndarray:
    img_array = np.frombuffer(image.source, dtype=np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("image decode failed")
    return img


def main() -> None:  # noqa: D103
    ctx = Content(
        origin_img=ImageContent.from_file("../../../workspace/Severe/2/in.png"),
        output_path="../../../workspace/Severe/2/out_lab.png",
        deblur_output_path="../../../workspace/Severe/2/deblur.png",
        prompt_lib=PromptLib("./prompt/"),
        editor=GeminiLLM("gemini-3.1-flash-image-preview"),
    )

    run(ctx)


if __name__ == "__main__":
    main()
