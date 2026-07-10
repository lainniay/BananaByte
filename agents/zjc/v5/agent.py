from enum import StrEnum
from typing import Any, Protocol, runtime_checkable

import cv2
import numpy as np
from pydantic import ConfigDict, Field

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


class State(StrEnum):  # noqa : D101
    INITIAL = "initial"
    PHYSICAL_RESTORE = "physical_restore"
    GENERATE_CANDIDATES = "generate_candidates"
    SELECT_CANDIDATE = "select_candidate"
    FUSE = "fuse"
    DONE = "done"


class Context(BaseState):  # noqa : D101
    model_config = ConfigDict(arbitrary_types_allowed=True)

    origin_img: ImageContent | None = None
    physical_img: ImageContent | None = None
    candidates: list[ImageContent] = Field(default_factory=list)
    selected_img: ImageContent | None = None
    cur_img: ImageContent | None = None

    prompt_lib: PromptLib | None = None
    editor: Editor

    input_path: str
    output_dir: str
    cur_state: State = State.INITIAL
    cur_round: int = 1
    candidate_count: int = 3


def run(ctx: Context) -> None:  # noqa : D103
    handlers = {
        State.INITIAL: handle_initial,
        State.PHYSICAL_RESTORE: handle_physical_restore,
        State.GENERATE_CANDIDATES: handle_generate_candidates,
        State.SELECT_CANDIDATE: handle_select_candidate,
        State.FUSE: handle_fuse,
    }

    if ctx.origin_img is None:
        ctx.origin_img = ImageContent.from_file(ctx.input_path)

    while ctx.cur_state is not State.DONE:
        handler = handlers.get(ctx.cur_state)
        if handler is None:
            raise ValueError(f"unknown state: {ctx.cur_state}")
        try:
            ctx.cur_state = handler(ctx)
        except Exception:
            ctx.save(
                f"{ctx.output_dir}/context.msgpack", exclude={"prompt_lib", "editor"}
            )
            raise

    handle_done(ctx)


def handle_initial(ctx: Context) -> State:  # noqa : D103
    if ctx.origin_img is None:
        raise ValueError("no image for initial stage")
    return State.PHYSICAL_RESTORE


def handle_physical_restore(ctx: Context) -> State:  # noqa : D103
    if ctx.origin_img is None:
        raise ValueError("no image for physical_restore stage")

    restored = conservative_physical_restore(_decode_bgr(ctx.origin_img))
    ctx.physical_img = _encode_bgr(restored, ctx.origin_img.mime_type)
    path = (
        f"{ctx.output_dir}/round_{ctx.cur_round}_physical{_image_ext(ctx.physical_img)}"
    )
    ctx.physical_img.save_to_file(path)
    print(f"[physical_restore] result saved: {path}")
    return State.GENERATE_CANDIDATES


def handle_generate_candidates(ctx: Context) -> State:  # noqa : D103
    if ctx.origin_img is None or ctx.physical_img is None:
        raise ValueError("no image for generate_candidates stage")
    if ctx.prompt_lib is None:
        raise ValueError("no prompt lib for generate_candidates stage")

    prompt = ctx.prompt_lib["generate_candidate"].render()
    ctx.candidates = []
    for index in range(1, ctx.candidate_count + 1):
        edit_input = Message(
            content=[
                TextContent(text=prompt),
                TextContent(text="Input 1: original underwater image."),
                ctx.origin_img,
                TextContent(
                    text="Input 2: conservative physical restoration reference."
                ),
                ctx.physical_img,
            ],
        )
        res = ctx.editor.edit_image(edit_input)
        if not res.images:
            raise ValueError("generate_candidates stage returned no image")

        candidate = res.images[0]
        ctx.candidates.append(candidate)
        path = f"{ctx.output_dir}/round_{ctx.cur_round}_candidate_{index}{_image_ext(candidate)}"
        candidate.save_to_file(path)
        print(f"[generate_candidates] candidate {index} saved: {path}")

    return State.SELECT_CANDIDATE


def handle_select_candidate(ctx: Context) -> State:  # noqa : D103
    if (
        ctx.origin_img is None
        or ctx.physical_img is None
        or len(ctx.candidates) != ctx.candidate_count
    ):
        raise ValueError(
            "candidate selection needs original, physical image and candidates"
        )

    origin = _decode_bgr(ctx.origin_img)
    physical = _decode_bgr(ctx.physical_img)
    scores = [
        candidate_difference_score(_decode_bgr(candidate), origin, physical)
        for candidate in ctx.candidates
    ]
    best_index = min(range(len(scores)), key=scores.__getitem__)
    ctx.selected_img = ctx.candidates[best_index]
    path = f"{ctx.output_dir}/round_{ctx.cur_round}_selected_candidate_{best_index + 1}{_image_ext(ctx.selected_img)}"
    ctx.selected_img.save_to_file(path)
    print(f"[select_candidate] scores: {[round(score, 4) for score in scores]}")
    print(f"[select_candidate] selected saved: {path}")
    return State.FUSE


def handle_fuse(ctx: Context) -> State:  # noqa : D103
    if ctx.origin_img is None or ctx.physical_img is None or ctx.selected_img is None:
        raise ValueError("no image for fuse stage")

    origin = _decode_bgr(ctx.origin_img)
    physical = _decode_bgr(ctx.physical_img)
    selected = resize_like(_decode_bgr(ctx.selected_img), physical)
    weight = compute_reliability_map(origin, physical, selected)
    fused = lab_haar_fuse(physical, selected, weight)
    ctx.cur_img = _encode_bgr(fused, ctx.origin_img.mime_type)
    path = f"{ctx.output_dir}/round_{ctx.cur_round}_fused{_image_ext(ctx.cur_img)}"
    ctx.cur_img.save_to_file(path)
    print(f"[fuse] result saved: {path}")
    return State.DONE


def handle_done(ctx: Context) -> None:  # noqa : D103
    if ctx.cur_img is not None:
        path = f"{ctx.output_dir}/out_v5{_image_ext(ctx.cur_img)}"
        ctx.cur_img.save_to_file(path)
        print(f"[done] final saved: {path}")
    ctx.cur_state = State.DONE


def conservative_physical_restore(bgr: np.ndarray) -> np.ndarray:  # noqa : D103
    img = bgr.astype(np.float32)
    means = np.maximum(img.reshape(-1, 3).mean(axis=0), 1.0)
    gain = np.clip(float(means.mean()) / means, 0.8, 1.2)
    img *= 1 + (gain - 1) * 0.45

    b_channel, g_channel, r_channel = cv2.split(img)
    red_missing = np.maximum(g_channel, b_channel) - r_channel
    r_channel += np.clip(red_missing, 0, 255) * 0.18
    img = cv2.merge([b_channel, g_channel, r_channel])

    lab = cv2.cvtColor(np.clip(img, 0, 255).astype(np.uint8), cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=1.4, tileGridSize=(8, 8))
    l_channel = cv2.addWeighted(l_channel, 0.82, clahe.apply(l_channel), 0.18, 0)
    return cv2.cvtColor(cv2.merge([l_channel, a_channel, b_channel]), cv2.COLOR_LAB2BGR)


def compute_reliability_map(  # noqa : D103
    origin_bgr: np.ndarray,
    physical_bgr: np.ndarray,
    generated_bgr: np.ndarray,
) -> np.ndarray:
    origin = resize_like(origin_bgr, physical_bgr)
    generated = resize_like(generated_bgr, physical_bgr)
    b_channel, g_channel, r_channel = cv2.split(origin.astype(np.float32) / 255.0)

    red_loss = _normalize(np.maximum(g_channel + b_channel - 2 * r_channel, 0))
    y_origin = _luminance(origin)
    contrast_loss = 1 - _normalize(cv2.Laplacian(y_origin, cv2.CV_32F, ksize=3))
    saturation = (
        cv2.cvtColor(origin, cv2.COLOR_BGR2HSV)[:, :, 1].astype(np.float32) / 255.0
    )
    haze = 1 - _normalize(saturation + _normalize(cv2.Laplacian(y_origin, cv2.CV_32F)))
    degradation = np.clip(0.45 * red_loss + 0.30 * contrast_loss + 0.25 * haze, 0, 1)

    phy_grad = _gradient(_luminance(physical_bgr))
    gen_grad = _gradient(_luminance(generated))
    consistency = np.exp(-np.abs(gen_grad - phy_grad) / 0.18)
    edge_protect = _normalize(_gradient(y_origin))

    weight = np.clip(degradation * consistency * (1 - edge_protect), 0, 1)
    return cv2.bilateralFilter(weight.astype(np.float32), 9, 0.1, 7)


def structure_difference(candidate_bgr: np.ndarray, physical_bgr: np.ndarray) -> float:  # noqa : D103
    candidate = resize_like(candidate_bgr, physical_bgr)
    return float(
        np.mean(
            np.abs(
                _gradient(_luminance(candidate)) - _gradient(_luminance(physical_bgr))
            )
        )
    )


def candidate_difference_score(  # noqa : D103
    candidate_bgr: np.ndarray,
    origin_bgr: np.ndarray,
    physical_bgr: np.ndarray,
) -> float:
    candidate = resize_like(candidate_bgr, physical_bgr)
    origin = resize_like(origin_bgr, physical_bgr)
    structure_penalty = structure_difference(candidate, physical_bgr)
    luminance_penalty = float(
        np.mean(np.abs(_luminance(candidate) - _luminance(physical_bgr))) / 255.0
    )
    color_gain = lab_chroma_difference(candidate, physical_bgr)
    origin_gain = lab_chroma_difference(candidate, origin)
    return (
        structure_penalty
        + 0.35 * luminance_penalty
        - 0.25 * max(color_gain, origin_gain)
    )


def lab_chroma_difference(  # noqa : D103
    left_bgr: np.ndarray, right_bgr: np.ndarray
) -> float:
    right = resize_like(right_bgr, left_bgr)
    left_lab = cv2.cvtColor(left_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    right_lab = cv2.cvtColor(right, cv2.COLOR_BGR2LAB).astype(np.float32)
    return float(np.mean(np.abs(left_lab[:, :, 1:] - right_lab[:, :, 1:])) / 255.0)


def lab_haar_fuse(  # noqa : D103
    physical_bgr: np.ndarray, generated_bgr: np.ndarray, weight: np.ndarray
) -> np.ndarray:
    generated = resize_like(generated_bgr, physical_bgr)
    physical_lab = cv2.cvtColor(physical_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    generated_lab = cv2.cvtColor(generated, cv2.COLOR_BGR2LAB).astype(np.float32)
    fused_channels = []

    for channel_index in range(3):
        phy_bands, shape = haar_split(physical_lab[:, :, channel_index])
        gen_bands, _ = haar_split(generated_lab[:, :, channel_index])
        scales = (
            (0.25, 0.06, 0.06, 0.0) if channel_index == 0 else (0.9, 0.35, 0.35, 0.08)
        )
        fused_bands = []
        for phy_band, gen_band, scale in zip(phy_bands, gen_bands, scales, strict=True):
            band_weight = (
                cv2.resize(weight, (phy_band.shape[1], phy_band.shape[0])) * scale
            )
            fused_bands.append(phy_band * (1 - band_weight) + gen_band * band_weight)
        fused_channels.append(haar_merge(tuple(fused_bands), shape))

    fused_lab = np.clip(cv2.merge(fused_channels), 0, 255).astype(np.uint8)
    return cv2.cvtColor(fused_lab, cv2.COLOR_LAB2BGR)


def haar_split(  # noqa : D103
    channel: np.ndarray,
) -> tuple[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray], tuple[int, int]]:
    height, width = channel.shape[:2]
    padded = cv2.copyMakeBorder(
        channel.astype(np.float32),
        0,
        height % 2,
        0,
        width % 2,
        cv2.BORDER_REFLECT_101,
    )
    even = padded[0::2, 0::2]
    odd_x = padded[0::2, 1::2]
    odd_y = padded[1::2, 0::2]
    odd_xy = padded[1::2, 1::2]
    ll = (even + odd_x + odd_y + odd_xy) / 4
    lh = (even - odd_x + odd_y - odd_xy) / 4
    hl = (even + odd_x - odd_y - odd_xy) / 4
    hh = (even - odd_x - odd_y + odd_xy) / 4
    return (ll, lh, hl, hh), (height, width)


def haar_merge(  # noqa : D103
    bands: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray], shape: tuple[int, int]
) -> np.ndarray:
    ll, lh, hl, hh = bands
    height, width = shape
    merged = np.empty((ll.shape[0] * 2, ll.shape[1] * 2), dtype=np.float32)
    merged[0::2, 0::2] = ll + lh + hl + hh
    merged[0::2, 1::2] = ll - lh + hl - hh
    merged[1::2, 0::2] = ll + lh - hl - hh
    merged[1::2, 1::2] = ll - lh - hl + hh
    return merged[:height, :width]


def resize_like(img: np.ndarray, reference: np.ndarray) -> np.ndarray:  # noqa : D103
    height, width = reference.shape[:2]
    if img.shape[:2] == (height, width):
        return img
    return cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)


def _gradient(gray: np.ndarray) -> np.ndarray:
    grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    return cv2.magnitude(grad_x, grad_y) / 255.0


def _luminance(bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)[:, :, 0].astype(np.float32)


def _normalize(data: np.ndarray) -> np.ndarray:
    data = data.astype(np.float32)
    low, high = np.percentile(data, [1, 99])
    if high <= low:
        return np.zeros(data.shape, dtype=np.float32)
    return np.clip((data - low) / (high - low), 0, 1)


def _image_ext(image: ImageContent) -> str:
    return mime_to_ext.get(image.mime_type, ".png")


def _decode_bgr(image: ImageContent) -> np.ndarray:
    img_array = np.frombuffer(image.source, dtype=np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("image decode failed")
    return img


def _encode_bgr(img: np.ndarray, mime_type: str) -> ImageContent:
    ext = mime_to_ext.get(mime_type, ".png")
    output_mime_type = mime_type if mime_type in mime_to_ext else "image/png"
    success, encoded_img = cv2.imencode(ext, img)
    if not success:
        raise ValueError("image encode failed")
    return ImageContent(source=encoded_img.tobytes(), mime_type=output_mime_type)


def main() -> None:  # noqa: D103
    ctx = Context(
        input_path="../../../workspace/Slight/1/in.jpg",
        output_dir="../../../workspace/Slight/1",
        prompt_lib=PromptLib("./prompts/"),
        editor=GeminiLLM("gemini-3.1-flash-image-preview"),
    )
    run(ctx)


if __name__ == "__main__":
    main()
