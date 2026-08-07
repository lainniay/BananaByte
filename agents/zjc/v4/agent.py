import subprocess
from enum import StrEnum
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import cv2
import numpy as np
from pydantic import ConfigDict

from agents.zjc.v3.spilt_lab import lab_to_bgr, resize_channel
from agents.zjc.v3.tool import mime_to_ext
from agents.zjc.v4.lut import (
    _ask_channel_gains,
    _print_rgb_means,
    _show_rgb_channels,
    apply_channel_luts,
)
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
    COMPLETE_COLOR = "complete_color"
    EDIT_COLOR = "edit_color"
    RICH_COLOR = "rich_color"
    REPAIR = "repair"
    DONE = "done"


class Context(BaseState):  # noqa : D101
    model_config = ConfigDict(arbitrary_types_allowed=True)

    origin_img: ImageContent | None = None
    cur_img: ImageContent | None = None

    prompt_lib: PromptLib | None = None
    editor: Editor

    input_path: str
    output_dir: str
    cur_state: State = State.INITIAL
    cur_round: int = 1

    channel_completion_max_weight: float = 0.5
    channel_completion_weight_gamma: float = 0.65
    channel_completion_candidate_gain: float = 1.15
    channel_completion_missing_threshold: float = 0.25
    channel_completion_low_frequency_sigma: float = 0.0
    channel_completion_candidate_min_value: float = 1.0
    channel_completion_candidate_max_value: float = 253.0
    chroma_base_ab_weight: float = 0.65
    chroma_target_ab_gain: float = 1.05

    record_used: bool = False


def run(ctx: Context) -> None:  # noqa : D103
    handlers = {
        State.INITIAL: handle_initial,
        State.COMPLETE_COLOR: handle_complete_color,
        State.EDIT_COLOR: handle_edit_color,
        State.RICH_COLOR: handle_rich_color,
        State.REPAIR: handle_repair,
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
            ctx.save("./context.msgpack", exclude={"prompt_lib", "editor"})
            raise

    handle_done(ctx)


def handle_initial(ctx: Context) -> State:  # noqa : D103
    if ctx.origin_img is None and ctx.cur_img is None:
        raise ValueError("no image for initial stage")

    if ctx.origin_img is None:
        ctx.origin_img = ctx.cur_img
    if ctx.cur_img is None:
        ctx.cur_img = ctx.origin_img

    return State.COMPLETE_COLOR


def handle_complete_color(ctx: Context) -> State:  # noqa : D103
    if ctx.cur_img is None:
        raise ValueError("no image for complete_color stage")
    if ctx.prompt_lib is None:
        raise ValueError("no prompt lib for complete_color stage")

    channel_paths = save_rgb_channel_previews(
        ctx.cur_img, ctx.output_dir, ctx.cur_round
    )
    print("[complete_color] channel previews saved:")
    for channel, path in channel_paths.items():
        print(f"[complete_color] {channel.upper()}: {path}")
        subprocess.run(["open", str(path)], check=False)

    raw_channels = (
        input(
            "[complete_color] choose channels to complete [r/g/b], "
            "or press Enter to skip: ",
        )
        .strip()
        .lower()
    )
    selected_channels = parse_complete_color_channels(raw_channels)
    if not selected_channels:
        return State.EDIT_COLOR

    prompt_text = ctx.prompt_lib["complete_color"].render()
    for raw in selected_channels:
        channel_paths = save_rgb_channel_previews(
            ctx.cur_img, ctx.output_dir, ctx.cur_round
        )
        complete_color_channel(ctx, raw, channel_paths[raw], prompt_text)

    return State.EDIT_COLOR


def parse_complete_color_channels(raw: str) -> list[str]:  # noqa : D103
    if raw in {"", "skip", "s", "n", "no"}:
        return []

    compact = raw.replace(",", " ").replace("/", " ")
    parts = compact.split()
    if len(parts) == 1:
        parts = list(parts[0])

    selected_channels: list[str] = []
    for channel in parts:
        if channel not in {"r", "g", "b"}:
            raise ValueError("channels must only contain: r, g, b")
        if channel not in selected_channels:
            selected_channels.append(channel)

    return selected_channels


def complete_color_channel(  # noqa : D103
    ctx: Context,
    raw: str,
    channel_path: Path,
    prompt_text: str,
) -> None:
    if ctx.cur_img is None:
        raise ValueError("no image for complete_color stage")

    edit_input = Message(
        content=[
            TextContent(text=prompt_text),
            TextContent(text="Input 2: grayscale image to complete."),
            ImageContent.from_file(channel_path),
            TextContent(text="Input 3: luminance structure guide image."),
            create_luminance_image(ctx.cur_img),
        ],
    )
    res = ctx.editor.edit_image(edit_input)
    if not res.images:
        raise ValueError("complete_color stage returned no image")

    completion_img = res.images[0]
    raw_path = (
        f"{ctx.output_dir}/round_{ctx.cur_round}_{raw}_channel_completion_raw"
        f"{_image_ext(completion_img)}"
    )
    completion_img.save_to_file(raw_path)
    print(f"[complete_color] raw completion saved: {raw_path}")

    ctx.cur_img = blend_completed_rgb_channel(
        ctx.cur_img,
        completion_img,
        channel=raw,
        max_weight=ctx.channel_completion_max_weight,
        weight_gamma=ctx.channel_completion_weight_gamma,
        candidate_gain=ctx.channel_completion_candidate_gain,
        missing_threshold=ctx.channel_completion_missing_threshold,
        low_frequency_sigma=ctx.channel_completion_low_frequency_sigma,
        candidate_min_value=ctx.channel_completion_candidate_min_value,
        candidate_max_value=ctx.channel_completion_candidate_max_value,
    )
    completed_path = (
        f"{ctx.output_dir}/round_{ctx.cur_round}_{raw}_channel_completed"
        f"{_image_ext(ctx.cur_img)}"
    )
    ctx.cur_img.save_to_file(completed_path)
    print(f"[complete_color] completed image saved: {completed_path}")


def handle_edit_color(ctx: Context) -> State:  # noqa : D103
    if ctx.cur_img is None:
        raise ValueError("no image for edit_color stage")

    bgr = _decode_bgr(ctx.cur_img)
    _show_rgb_channels(bgr)
    _print_rgb_means(bgr)

    enhance_gains, reduce_gains = _ask_channel_gains()
    adjusted_bgr = apply_channel_luts(
        bgr,
        enhance_gains=enhance_gains,
        reduce_gains=reduce_gains,
    )
    _show_rgb_channels(adjusted_bgr)

    ctx.cur_img = _encode_bgr(adjusted_bgr, ctx.cur_img.mime_type)
    output_path = Path(ctx.output_dir) / (
        f"round_{ctx.cur_round}_edit_color{_image_ext(ctx.cur_img)}"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ctx.cur_img.save_to_file(str(output_path))

    print(f"[edit_color] enhance: {enhance_gains or '无'}")
    print(f"[edit_color] reduce: {reduce_gains or '无'}")
    print(f"[edit_color] result saved: {output_path}")

    return State.RICH_COLOR


def handle_rich_color(ctx: Context) -> State:  # noqa : D103
    if ctx.cur_img is None:
        raise ValueError("no image for rich_color stage")
    if ctx.prompt_lib is None:
        raise ValueError("no prompt lib for rich_color stage")

    edit_input = Message(
        content=[
            TextContent(text=ctx.prompt_lib["rich_color"].render()),
            TextContent(text="Input 1: current image to enrich colors."),
            ctx.cur_img,
            TextContent(text="Input 2: luminance structure guide image."),
            create_luminance_image(ctx.cur_img),
        ],
    )
    res = ctx.editor.edit_image(edit_input)
    if not res.images:
        raise ValueError("rich_color stage returned no image")

    color_candidate_img = res.images[0]
    candidate_path = (
        f"{ctx.output_dir}/round_{ctx.cur_round}_rich_candidate"
        f"{_image_ext(color_candidate_img)}"
    )
    color_candidate_img.save_to_file(candidate_path)
    print(f"[rich_color] color candidate saved: {candidate_path}")

    ctx.cur_img = merge_l_with_blended_ab(
        l_source=ctx.cur_img,
        base_ab=ctx.cur_img,
        target_ab=color_candidate_img,
        base_ab_weight=ctx.chroma_base_ab_weight,
        target_ab_gain=ctx.chroma_target_ab_gain,
    )
    rich_path = f"{ctx.output_dir}/round_{ctx.cur_round}_rich{_image_ext(ctx.cur_img)}"
    ctx.cur_img.save_to_file(rich_path)
    print(f"[rich_color] LAB AB-merged rich image saved: {rich_path}")

    return State.REPAIR


def handle_repair(ctx: Context) -> State:  # noqa : D103
    if ctx.cur_img is None:
        raise ValueError("no image for repair stage")
    if ctx.prompt_lib is None:
        raise ValueError("no prompt lib for repair stage")

    edit_input = Message(
        content=[
            TextContent(text=ctx.prompt_lib["repair"].render()),
            TextContent(text="Input 1: image to naturalize and repair."),
            ctx.cur_img,
        ],
    )
    res = ctx.editor.edit_image(edit_input)
    if not res.images:
        raise ValueError("repair stage returned no image")

    ctx.cur_img = res.images[0]
    repaired_path = (
        f"{ctx.output_dir}/round_{ctx.cur_round}_repaired{_image_ext(ctx.cur_img)}"
    )
    ctx.cur_img.save_to_file(repaired_path)
    print(f"[repair] repaired image saved: {repaired_path}")

    return State.DONE


def handle_done(ctx: Context) -> None:  # noqa : D103
    ctx.cur_state = State.DONE


def _image_ext(image: ImageContent) -> str:
    return mime_to_ext.get(image.mime_type, ".png")


def save_rgb_channel_previews(  # noqa : D103
    image: ImageContent,
    output_dir: str,
    cur_round: int,
) -> dict[str, Path]:
    bgr = _decode_bgr(image)
    b_channel, g_channel, r_channel = cv2.split(bgr)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    channel_paths = {
        "r": output_path / f"round_{cur_round}_r_channel.png",
        "g": output_path / f"round_{cur_round}_g_channel.png",
        "b": output_path / f"round_{cur_round}_b_channel.png",
    }
    channels = {"r": r_channel, "g": g_channel, "b": b_channel}

    for channel, path in channel_paths.items():
        if not cv2.imwrite(str(path), channels[channel]):
            raise ValueError(f"channel preview save failed: {path}")

    return channel_paths


def create_luminance_image(image: ImageContent) -> ImageContent:  # noqa : D103
    bgr = _decode_bgr(image)
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    return _encode_gray(lab[:, :, 0])


def blend_completed_rgb_channel(  # noqa : D103
    image: ImageContent,
    completed_channel: ImageContent,
    *,
    channel: str,
    max_weight: float,
    weight_gamma: float,
    candidate_gain: float,
    missing_threshold: float,
    low_frequency_sigma: float,
    candidate_min_value: float,
    candidate_max_value: float,
) -> ImageContent:
    if channel not in {"r", "g", "b"}:
        raise ValueError("channel must be one of: r, g, b")
    if not 0 <= max_weight <= 1:
        raise ValueError("max_weight must be between 0 and 1")
    if weight_gamma <= 0:
        raise ValueError("weight_gamma must be positive")
    if candidate_gain <= 0:
        raise ValueError("candidate_gain must be positive")
    if not 0 <= missing_threshold <= 1:
        raise ValueError("missing_threshold must be between 0 and 1")
    if low_frequency_sigma < 0:
        raise ValueError("low_frequency_sigma must be non-negative")
    if not 0 <= candidate_min_value <= candidate_max_value <= 255:
        raise ValueError(
            "candidate values must satisfy 0 <= min <= max <= 255",
        )

    bgr = _decode_bgr(image)
    completed = _decode_gray(completed_channel)
    height, width = bgr.shape[:2]
    if completed.shape[:2] != (height, width):
        completed = cv2.resize(completed, (width, height), interpolation=cv2.INTER_AREA)

    channel_index = {"b": 0, "g": 1, "r": 2}[channel]
    original = bgr[:, :, channel_index].astype(np.float32)
    completed_low_frequency = smooth_completed_channel(
        completed,
        sigma=low_frequency_sigma,
    )
    candidate = completed_low_frequency.astype(np.float32) * candidate_gain

    weight = create_channel_completion_weight(
        bgr,
        channel_index=channel_index,
        max_weight=max_weight,
        gamma=weight_gamma,
        missing_threshold=missing_threshold,
    )
    candidate_mask = (completed_low_frequency >= candidate_min_value) & (
        completed_low_frequency <= candidate_max_value
    )
    weight = np.where(candidate_mask, weight, 0.0)

    blended = original * (1 - weight) + candidate * weight
    bgr[:, :, channel_index] = np.clip(blended, 0, 255).astype(np.uint8)

    return _encode_bgr(bgr, image.mime_type)


def smooth_completed_channel(  # noqa : D103
    completed: np.ndarray, *, sigma: float
) -> np.ndarray:
    if sigma == 0:
        return completed
    return cv2.GaussianBlur(completed, (0, 0), sigma)


def create_channel_completion_weight(  # noqa : D103
    bgr: np.ndarray,
    *,
    channel_index: int,
    max_weight: float,
    gamma: float,
    missing_threshold: float,
) -> np.ndarray:
    if gamma <= 0:
        raise ValueError("gamma must be positive")
    if not 0 <= missing_threshold <= 1:
        raise ValueError("missing_threshold must be between 0 and 1")

    bgr_float = bgr.astype(np.float32)
    selected = bgr_float[:, :, channel_index]
    other_channels = [
        bgr_float[:, :, index] for index in range(3) if index != channel_index
    ]
    other_max = np.maximum(other_channels[0], other_channels[1])
    luminance = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)[:, :, 0].astype(np.float32)

    reference = np.maximum(other_max, luminance)
    reference = np.maximum(reference, 1.0)
    relative_missing = 1 - np.clip(selected / reference, 0, 1)

    selected_scale = max(float(np.percentile(selected, 99)), 1.0)
    absolute_missing = 1 - np.clip(selected / selected_scale, 0, 1)

    missing = np.maximum(relative_missing, 0.5 * absolute_missing)
    trusted_missing = missing >= missing_threshold
    missing = np.power(np.clip(missing, 0, 1), gamma)
    missing = cv2.GaussianBlur(missing, (0, 0), 2)
    weight = max_weight * np.clip(missing, 0, 1)
    return np.where(trusted_missing, weight, 0.0)


def merge_l_with_blended_ab(  # noqa : D103
    l_source: ImageContent,
    base_ab: ImageContent,
    target_ab: ImageContent,
    *,
    base_ab_weight: float,
    target_ab_gain: float = 1.0,
) -> ImageContent:
    if not 0 <= base_ab_weight <= 1:
        raise ValueError("base_ab_weight must be between 0 and 1")
    if target_ab_gain <= 0:
        raise ValueError("target_ab_gain must be positive")

    target_ab_weight = 1 - base_ab_weight
    l_source_bgr = _decode_bgr(l_source)
    base_ab_bgr = _decode_bgr(base_ab)
    target_ab_bgr = _decode_bgr(target_ab)

    l_source_lab = cv2.cvtColor(l_source_bgr, cv2.COLOR_BGR2LAB)
    base_ab_lab = cv2.cvtColor(base_ab_bgr, cv2.COLOR_BGR2LAB)
    target_ab_lab = cv2.cvtColor(target_ab_bgr, cv2.COLOR_BGR2LAB)

    l_channel = l_source_lab[:, :, 0]
    height, width = l_channel.shape[:2]
    size = (width, height)

    base_a = resize_channel(base_ab_lab[:, :, 1], size).astype(np.float32)
    base_b = resize_channel(base_ab_lab[:, :, 2], size).astype(np.float32)
    target_a = resize_channel(target_ab_lab[:, :, 1], size).astype(np.float32)
    target_b = resize_channel(target_ab_lab[:, :, 2], size).astype(np.float32)

    if target_ab_gain != 1.0:
        target_a = boost_lab_chroma_channel(target_a, target_ab_gain)
        target_b = boost_lab_chroma_channel(target_b, target_ab_gain)

    a_channel = np.clip(
        base_a * base_ab_weight + target_a * target_ab_weight,
        0,
        255,
    ).astype(np.uint8)
    b_channel = np.clip(
        base_b * base_ab_weight + target_b * target_ab_weight,
        0,
        255,
    ).astype(np.uint8)
    merged_bgr = lab_to_bgr(l_channel, a_channel, b_channel)

    return _encode_bgr(merged_bgr, l_source.mime_type)


def boost_lab_chroma_channel(  # noqa : D103
    channel: np.ndarray, gain: float
) -> np.ndarray:
    return np.clip(128.0 + (channel - 128.0) * gain, 0, 255)


def _decode_bgr(image: ImageContent) -> np.ndarray:
    img_array = np.frombuffer(image.source, dtype=np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("image decode failed")
    return img


def _decode_gray(image: ImageContent) -> np.ndarray:
    img_array = np.frombuffer(image.source, dtype=np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_GRAYSCALE)
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


def _encode_gray(img: np.ndarray) -> ImageContent:
    success, encoded_img = cv2.imencode(".png", img)
    if not success:
        raise ValueError("image encode failed")
    return ImageContent(source=encoded_img.tobytes(), mime_type="image/png")


ctx = Context(
    input_path="../../../workspace/Slight/6/in.png",
    output_dir="../../../workspace/Slight/6",
    prompt_lib=PromptLib("./prompts/"),
    editor=GeminiLLM("gemini-3.1-flash-image-preview"),
)

run(ctx)
