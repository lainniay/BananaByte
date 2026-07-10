from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

DEFAULT_IMAGE_PATH = Path(__file__).with_name("out.jpg")
DEFAULT_OUTPUT_PATH = Path(__file__).with_name("out_lab_lut.jpg")
_LAB_CHANNELS = ("a", "b")
_CHANNEL_INDEX = {"a": 1, "b": 2}


def _smoothstep(edge0: float, edge1: float, value: np.ndarray) -> np.ndarray:
    if edge0 >= edge1:
        raise ValueError("edge0 must be smaller than edge1")
    t = np.clip((value - edge0) / (edge1 - edge0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _build_enhance_soft_knee_lut(
    gain: float,
    *,
    knee_start: float = 0.15,
    knee_end: float = 0.85,
) -> np.ndarray:
    if gain < 1.0:
        raise ValueError("enhance gain must be >= 1.0")
    values = np.arange(256, dtype=np.float32)
    delta = values - 128.0
    strength = np.abs(delta) / 127.0
    protect = _smoothstep(knee_start, knee_end, strength)
    dynamic_gain = gain - (gain - 1.0) * protect
    return np.clip(128.0 + delta * dynamic_gain, 0, 255).astype(np.uint8)


def _build_reduce_reverse_soft_knee_lut(
    gain: float,
    *,
    knee_start: float = 0.2,
    knee_end: float = 0.9,
) -> np.ndarray:
    if not 0.0 < gain <= 1.0:
        raise ValueError("reduce gain must satisfy 0 < gain <= 1")
    values = np.arange(256, dtype=np.float32)
    delta = values - 128.0
    chroma_strength = np.abs(delta) / 127.0
    reduce_strength = _smoothstep(knee_start, knee_end, chroma_strength)
    dynamic_gain = 1.0 - (1.0 - gain) * reduce_strength
    return np.clip(128.0 + delta * dynamic_gain, 0, 255).astype(np.uint8)


def apply_lab_luts(
    bgr: np.ndarray,
    *,
    channel_offsets: dict[str, float],
    enhance_gains: dict[str, float],
    reduce_gains: dict[str, float],
) -> np.ndarray:
    """Apply interactive LAB A/B offset and LUT settings to an OpenCV BGR image."""
    overlap = set(enhance_gains) & set(reduce_gains)
    if overlap:
        raise ValueError(f"channels cannot be both enhanced and reduced: {sorted(overlap)}")

    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    channels = list(cv2.split(lab))

    for channel, offset in channel_offsets.items():
        index = _CHANNEL_INDEX[channel]
        shifted = channels[index].astype(np.float32) + offset
        channels[index] = np.clip(shifted, 0, 255).astype(np.uint8)
    for channel, gain in enhance_gains.items():
        index = _CHANNEL_INDEX[channel]
        channels[index] = cv2.LUT(channels[index], _build_enhance_soft_knee_lut(gain))
    for channel, gain in reduce_gains.items():
        index = _CHANNEL_INDEX[channel]
        channels[index] = cv2.LUT(channels[index], _build_reduce_reverse_soft_knee_lut(gain))

    adjusted_lab = cv2.merge(channels)
    return cv2.cvtColor(adjusted_lab, cv2.COLOR_LAB2BGR)


def _ask_float(prompt: str, default: float) -> float:
    raw = input(f"{prompt} (default: {default}): ").strip()
    if not raw:
        return default
    return float(raw)


def _ask_channel_settings() -> tuple[dict[str, float], dict[str, float], dict[str, float]]:
    channel_offsets: dict[str, float] = {}
    enhance_gains: dict[str, float] = {}
    reduce_gains: dict[str, float] = {}
    for channel in _LAB_CHANNELS:
        print(f"\n{channel.upper()} 通道")
        offset = _ask_float("offset, 可正可负", 0.0)
        if offset != 0.0:
            channel_offsets[channel] = offset
        gain = _ask_float("调整系数, 1.0 不变, <1.0 减弱, >1.0 增强", 1.0)
        if gain < 1.0:
            reduce_gains[channel] = gain
        elif gain > 1.0:
            enhance_gains[channel] = gain
    return channel_offsets, enhance_gains, reduce_gains


def _load_lab(image_path: Path) -> tuple[np.ndarray, np.ndarray]:
    bgr = cv2.imread(str(image_path))
    if bgr is None:
        raise ValueError(f"图片读取失败: {image_path}")
    return bgr, cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)


def _show_lab_channels(lab: np.ndarray, prefix: str) -> None:
    _, a_channel, b_channel = cv2.split(lab)
    cv2.imshow(f"{prefix} A Channel", a_channel)
    cv2.imshow(f"{prefix} B Channel", b_channel)
    cv2.waitKey(1)


def _print_ab_means(lab: np.ndarray) -> None:
    _, a_mean, b_mean, _ = cv2.mean(lab)
    print(f"A 均值: {a_mean:.2f}")
    print(f"B 均值: {b_mean:.2f}")


def _run_interactive(image_path: Path, output_path: Path) -> None:
    bgr, lab = _load_lab(image_path)
    _show_lab_channels(lab, "Before")
    _print_ab_means(lab)
    channel_offsets, enhance_gains, reduce_gains = _ask_channel_settings()

    adjusted = apply_lab_luts(
        bgr,
        channel_offsets=channel_offsets,
        enhance_gains=enhance_gains,
        reduce_gains=reduce_gains,
    )
    adjusted_lab = cv2.cvtColor(adjusted, cv2.COLOR_BGR2LAB)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), adjusted):
        raise ValueError(f"图片保存失败: {output_path}")

    print(f"offset: {channel_offsets or '无'}")
    print(f"增强通道: {enhance_gains or '无'}")
    print(f"减弱通道: {reduce_gains or '无'}")
    print(f"结果已保存: {output_path}")
    cv2.imshow("Before", bgr)
    cv2.imshow("After LAB LUT", adjusted)
    _show_lab_channels(adjusted_lab, "After")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Interactively apply soft-knee LUT adjustments on LAB A/B channels.",
    )
    parser.add_argument(
        "image",
        nargs="?",
        type=Path,
        default=DEFAULT_IMAGE_PATH,
        help=f"input image path, default: {DEFAULT_IMAGE_PATH}",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help=f"output image path, default: {DEFAULT_OUTPUT_PATH}",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    _run_interactive(args.image, args.output)
