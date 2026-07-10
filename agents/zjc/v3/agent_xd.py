import subprocess
from enum import Enum
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import cv2
import numpy as np
from pydantic import ConfigDict

from agents.zjc.v3.spilt_lab import lab_to_bgr, resize_channel
from agents.zjc.v3.tool import adjust_rgb_blance, get_color_mean, mime_to_ext
from core import BaseState, GeminiLLM, ImageContent, Message, PromptLib, TextContent


class State(Enum):  # noqa : D101
    ANALYZE = "analyze"
    COMPLETE_COLOR = "complete_color"
    EDIT_COLOR = "edit_color"
    REPAIR_CHROMA = "repair_chroma"
    DEHAZE_L = "dehaze_l"
    EDIT_TEXTURE = "edit_texture"
    ENHANCE_COLOR = "enhance_color"
    EVALUATE = "evaluate"
    DONE = "done"


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
    texture_base_img: ImageContent | None = None
    color_candidate_img: ImageContent | None = None
    texture_img: ImageContent | None = None
    deblur_img: ImageContent | None = None
    enhanced_img: ImageContent | None = None

    cur_state: State = State.ANALYZE
    cur_round: int = 1

    prompt_lib: PromptLib
    editor: Editor

    output_dir: str
    input_path: str
    channel_completion_max_weight: float = 0.5
    channel_completion_weight_gamma: float = 0.65
    channel_completion_candidate_gain: float = 1.15
    channel_completion_missing_threshold: float = 0.35
    channel_completion_low_frequency_sigma: float = 12.0
    channel_completion_candidate_min_value: float = 2.0
    channel_completion_candidate_max_value: float = 253.0
    chroma_base_ab_weight: float = 0.6
    chroma_target_ab_gain: float = 1.05
    dehaze_l_weight: float = 0.2
    dehaze_clip_limit: float = 1.2
    base_ab_weight: float = 0.8
    texture_ab_weight: float = 0.3


def run(ctx: Content) -> None:  # noqa : D103
    handlers = {
        State.ANALYZE: handle_analyze,
        State.COMPLETE_COLOR: handle_complete_color,
        State.EDIT_COLOR: handle_edit_color,
        State.REPAIR_CHROMA: handle_repair_chroma,
        State.DEHAZE_L: handle_dehaze_l,
        State.EDIT_TEXTURE: handle_edit_texture,
        State.ENHANCE_COLOR: handle_enhance_color,
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

    return State.COMPLETE_COLOR


def handle_complete_color(ctx: Content) -> State:  # noqa : D103
    if ctx.cur_img is None:
        raise ValueError("no image for complete_color stage")

    channel_paths = save_rgb_channel_previews(
        ctx.cur_img, ctx.output_dir, ctx.cur_round
    )
    print("[complete_color] channel previews saved:")
    for channel, path in channel_paths.items():
        print(f"[complete_color] {channel.upper()}: {path}")
        subprocess.run(["open", str(path)], check=False)

    raw = (
        input(
            "[complete_color] choose channel to complete [r/g/b], "
            "or press Enter to skip: ",
        )
        .strip()
        .lower()
    )
    if raw in {"", "skip", "s", "n", "no"}:
        return State.EDIT_COLOR
    if raw not in {"r", "g", "b"}:
        raise ValueError("channel must be one of: r, g, b")

    channel_name = {"r": "red", "g": "green", "b": "blue"}[raw]
    edit_input = Message(
        content=[
            TextContent(
                text=(
                    f"Complete the missing information in the {channel_name} color "
                    "channel. Return a single grayscale image representing the "
                    f"repaired {channel_name} channel. Make the restored channel "
                    "clearly visible and well exposed instead of faint or "
                    "conservative, especially where this channel is missing, weak, "
                    "or absorbed. Preserve the exact object shapes, edges, texture "
                    "positions, composition, and luminance structure. Do not crop, "
                    "rotate, repaint scene content, or add new objects."
                ),
            ),
            TextContent(text="Input 1: full RGB reference image."),
            ctx.cur_img,
            TextContent(text=f"Input 2: current {channel_name} channel image."),
            ImageContent.from_file(channel_paths[raw]),
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

    return State.EDIT_COLOR


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
    ctx.texture_base_img = fixed

    color_path = f"{ctx.output_dir}/round_{ctx.cur_round}_color{ext}"
    ctx.cur_img.save_to_file(color_path)
    print(f"[edit_color] adjusted image saved: {color_path}")

    b_mean, g_mean, r_mean = get_color_mean(ctx.cur_img)
    print(
        "[edit_color] adjusted color mean (R, G, B): "
        f"{r_mean:.2f}, {g_mean:.2f}, {b_mean:.2f}",
    )

    return State.REPAIR_CHROMA


def handle_repair_chroma(ctx: Content) -> State:  # noqa : D103
    if ctx.cur_img is None:
        raise ValueError("no image for repair_chroma stage")

    edit_input = Message(
        content=[
            TextContent(
                text=(
                    "Restore plausible natural colors for the image. Focus on "
                    "missing, faded, biased, or locally inconsistent chroma. Keep "
                    "the recovery restrained and plausible; avoid pushing color too "
                    "strongly or making warm tones look artificial. Return one "
                    "complete RGB image with improved color only. "
                    "Preserve the exact composition, object shapes, edges, texture "
                    "positions, brightness structure, and scene content. Do not crop, "
                    "rotate, add objects, remove objects, repaint details, sharpen, "
                    "blur, or change lighting/contrast aggressively."
                ),
            ),
            TextContent(text="Input 1: current image to color-repair."),
            ctx.cur_img,
            TextContent(text="Input 2: luminance structure guide image."),
            create_luminance_image(ctx.cur_img),
        ],
    )
    res = ctx.editor.edit_image(edit_input)
    if not res.images:
        raise ValueError("repair_chroma stage returned no image")

    ctx.color_candidate_img = res.images[0]
    candidate_path = (
        f"{ctx.output_dir}/round_{ctx.cur_round}_chroma_candidate"
        f"{_image_ext(ctx.color_candidate_img)}"
    )
    ctx.color_candidate_img.save_to_file(candidate_path)
    print(f"[repair_chroma] color candidate saved: {candidate_path}")

    ctx.cur_img = merge_l_with_blended_ab(
        l_source=ctx.cur_img,
        base_ab=ctx.cur_img,
        target_ab=ctx.color_candidate_img,
        base_ab_weight=ctx.chroma_base_ab_weight,
        target_ab_gain=ctx.chroma_target_ab_gain,
    )
    ctx.texture_base_img = ctx.cur_img

    repaired_path = (
        f"{ctx.output_dir}/round_{ctx.cur_round}_chroma_repaired"
        f"{_image_ext(ctx.cur_img)}"
    )
    ctx.cur_img.save_to_file(repaired_path)
    print(f"[repair_chroma] LAB chroma-repaired image saved: {repaired_path}")

    return State.EDIT_TEXTURE


def handle_dehaze_l(ctx: Content) -> State:  # noqa : D103
    if ctx.cur_img is None:
        raise ValueError("no image for dehaze_l stage")

    ctx.cur_img = dehaze_l_channel(
        ctx.cur_img,
        weight=ctx.dehaze_l_weight,
        clip_limit=ctx.dehaze_clip_limit,
    )
    ctx.texture_base_img = ctx.cur_img

    dehaze_path = (
        f"{ctx.output_dir}/round_{ctx.cur_round}_dehaze_l{_image_ext(ctx.cur_img)}"
    )
    ctx.cur_img.save_to_file(dehaze_path)
    print(f"[dehaze_l] L-channel dehazed image saved: {dehaze_path}")

    return State.EDIT_TEXTURE


def handle_edit_texture(ctx: Content) -> State:  # noqa : D103
    if ctx.cur_img is None:
        raise ValueError("no image for edit_texture stage")

    edit_input = Message(
        content=[TextContent(text=ctx.prompt_lib["blur"].render()), ctx.cur_img],
    )
    res = ctx.editor.edit_image(edit_input)
    if not res.images:
        raise ValueError("edit_texture stage returned no image")

    ctx.deblur_img = res.images[0]
    ext = _image_ext(ctx.deblur_img)
    deblur_path = f"{ctx.output_dir}/round_{ctx.cur_round}_deblur{ext}"
    ctx.deblur_img.save_to_file(deblur_path)
    print(f"[edit_texture] deblur image saved: {deblur_path}")

    ctx.cur_img = ctx.deblur_img
    texture_path = f"{ctx.output_dir}/round_{ctx.cur_round}_texture{ext}"
    ctx.cur_img.save_to_file(texture_path)
    print(f"[edit_texture] direct texture image saved: {texture_path}")
    ctx.texture_img = ctx.cur_img

    return State.ENHANCE_COLOR


def handle_enhance_color(ctx: Content) -> State:  # noqa : D103
    if ctx.cur_img is None:
        raise ValueError("no image for enhance_color stage")
    if ctx.texture_img is None:
        raise ValueError("no texture image for enhance_color stage")

    edit_input = Message(
        content=[
            TextContent(
                text=(
                    "Apply a subtle whole-image harmonization pass. Reduce unnatural "
                    "color transitions, channel-fusion artifacts, haze residues, and "
                    "local inconsistency caused by prior processing. Keep the change "
                    "strength low: preserve the exact composition, object shapes, "
                    "edges, texture details, brightness structure, and overall color "
                    "intent. Do not strongly recolor the image, do not add saturation "
                    "aggressively, do not blur, crop, repaint, or change scene content."
                ),
            ),
            ctx.cur_img,
        ],
    )
    res = ctx.editor.edit_image(edit_input)
    if not res.images:
        raise ValueError("enhance_color stage returned no image")

    ctx.enhanced_img = res.images[0]
    enhanced_path = (
        f"{ctx.output_dir}/round_{ctx.cur_round}_enhance_raw"
        f"{_image_ext(ctx.enhanced_img)}"
    )
    ctx.enhanced_img.save_to_file(enhanced_path)
    print(f"[enhance_color] enhanced raw image saved: {enhanced_path}")

    ctx.cur_img = ctx.enhanced_img
    enhance_path = (
        f"{ctx.output_dir}/round_{ctx.cur_round}_enhance{_image_ext(ctx.cur_img)}"
    )
    ctx.cur_img.save_to_file(enhance_path)
    print(f"[enhance_color] enhanced color image saved: {enhance_path}")

    return State.EVALUATE


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
        return State.COMPLETE_COLOR
    return State.DONE


def handle_done(ctx: Content) -> None:  # noqa : D103
    if ctx.cur_img is not None:
        output_path = ctx.cur_img.save_to_file(
            f"{ctx.output_dir}/out_xd{_image_ext(ctx.cur_img)}"
        )
        print(f"[done] final image saved: {output_path}")
    ctx.cur_state = State.DONE


def _image_ext(image: ImageContent) -> str:
    return mime_to_ext.get(image.mime_type, ".png")


def save_rgb_channel_previews(
    image: ImageContent,
    output_dir: str,
    cur_round: int,
) -> dict[str, Path]:
    """Save current RGB channels as grayscale preview images."""
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


def create_luminance_image(image: ImageContent) -> ImageContent:
    """Create a grayscale LAB-L image for structure guidance."""
    bgr = _decode_bgr(image)
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    return _encode_gray(lab[:, :, 0])


def dehaze_l_channel(
    image: ImageContent,
    *,
    weight: float,
    clip_limit: float,
) -> ImageContent:
    """Enhance LAB-L with CLAHE and blend it back for light dehazing."""
    if not 0 <= weight <= 1:
        raise ValueError("weight must be between 0 and 1")
    if clip_limit <= 0:
        raise ValueError("clip_limit must be positive")

    bgr = _decode_bgr(image)
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
    enhanced_l = clahe.apply(l_channel)
    blended_l = cv2.addWeighted(l_channel, 1 - weight, enhanced_l, weight, 0)

    merged_lab = cv2.merge([blended_l, a_channel, b_channel])
    merged_bgr = cv2.cvtColor(merged_lab, cv2.COLOR_LAB2BGR)
    return _encode_bgr(merged_bgr, image.mime_type)


def blend_completed_rgb_channel(
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
    """Blend a completed RGB channel back with a missing-information mask."""
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


def smooth_completed_channel(completed: np.ndarray, *, sigma: float) -> np.ndarray:
    """Keep only low-frequency color prior from a generated channel."""
    if sigma == 0:
        return completed
    return cv2.GaussianBlur(completed, (0, 0), sigma)


def create_channel_completion_weight(
    bgr: np.ndarray,
    *,
    channel_index: int,
    max_weight: float,
    gamma: float,
    missing_threshold: float,
) -> np.ndarray:
    """Create a stronger blend mask where the selected channel is relatively weak."""
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


def merge_l_with_blended_ab(
    l_source: ImageContent,
    base_ab: ImageContent,
    target_ab: ImageContent,
    *,
    base_ab_weight: float,
    target_ab_gain: float = 1.0,
) -> ImageContent:
    """Use source LAB-L and blended LAB-a/b to create a new image."""
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

    mime_type = l_source.mime_type if l_source.mime_type in mime_to_ext else "image/png"
    ext = mime_to_ext[mime_type]
    success, encoded_img = cv2.imencode(ext, merged_bgr)
    if not success:
        raise ValueError("lab merge image encode failed")

    return ImageContent(source=encoded_img.tobytes(), mime_type=mime_type)


def boost_lab_chroma_channel(channel: np.ndarray, gain: float) -> np.ndarray:
    """Increase LAB-a/b distance from neutral gray."""
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


def main() -> None:  # noqa: D103
    ctx = Content(
        editor=GeminiLLM("gemini-3-pro-image-preview"),
        prompt_lib=PromptLib("./prompt/"),
        input_path="../../../workspace/Slight/13/in.png",
        output_dir="../../../workspace/Slight/13",
    )

    run(ctx)


if __name__ == "__main__":
    main()
