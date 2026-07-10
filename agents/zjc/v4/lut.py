from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

DEFAULT_IMAGE_PATH = Path(__file__).with_name("out.jpg")
DEFAULT_OUTPUT_PATH = Path(__file__).with_name("out_lut.jpg")
_CHANNEL_INDEX = {"b": 0, "g": 1, "r": 2}
_RGB_CHANNELS = ("r", "g", "b")


def _smoothstep(edge0: float, edge1: float, value: np.ndarray) -> np.ndarray:
    if edge0 >= edge1:
        raise ValueError("edge0 must be smaller than edge1")
    t = np.clip((value - edge0) / (edge1 - edge0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _build_enhance_soft_knee_lut(
    gain: float,
    *,
    knee_start: float = 0.05,
    knee_end: float = 0.85,
) -> np.ndarray:
    if gain < 1.0:
        raise ValueError("enhance gain must be >= 1.0")
    values = np.arange(256, dtype=np.float32)
    t = values / 255.0
    protect = _smoothstep(knee_start, knee_end, t)
    dynamic_gain = gain - (gain - 1.0) * protect
    return np.clip(values * dynamic_gain, 0, 255).astype(np.uint8)


def _build_reduce_reverse_soft_knee_lut(
    gain: float,
    *,
    knee_start: float = 0.35,
    knee_end: float = 0.9,
) -> np.ndarray:
    if not 0.0 < gain <= 1.0:
        raise ValueError("reduce gain must satisfy 0 < gain <= 1")
    values = np.arange(256, dtype=np.float32)
    t = values / 255.0
    strength = _smoothstep(knee_start, knee_end, t)
    dynamic_gain = 1.0 - (1.0 - gain) * strength
    return np.clip(values * dynamic_gain, 0, 255).astype(np.uint8)


def apply_channel_luts(
    bgr: np.ndarray,
    *,
    enhance_gains: dict[str, float],
    reduce_gains: dict[str, float],
) -> np.ndarray:
    """Apply interactive edit-color LUT settings to an OpenCV BGR image."""
    overlap = set(enhance_gains) & set(reduce_gains)
    if overlap:
        raise ValueError(f"channels cannot be both enhanced and reduced: {sorted(overlap)}")

    channels = list(cv2.split(bgr))

    for channel, gain in enhance_gains.items():
        index = _CHANNEL_INDEX[channel]
        channels[index] = cv2.LUT(channels[index], _build_enhance_soft_knee_lut(gain))
    for channel, gain in reduce_gains.items():
        index = _CHANNEL_INDEX[channel]
        channels[index] = cv2.LUT(channels[index], _build_reduce_reverse_soft_knee_lut(gain))

    return cv2.merge(channels)


def _ask_float(prompt: str, default: float) -> float:
    raw = input(f"{prompt} (default: {default}): ").strip()
    if not raw:
        return default
    return float(raw)


def _ask_yes(prompt: str) -> bool:
    raw = input(f"{prompt} [y/N]: ").strip().lower()
    return raw in {"y", "yes"}


def _ask_channel_gains() -> tuple[dict[str, float], dict[str, float]]:
    enhance_gains: dict[str, float] = {}
    reduce_gains: dict[str, float] = {}
    for channel in _RGB_CHANNELS:
        print(f"\n{channel.upper()} 通道")
        if _ask_yes("是否减弱"):
            reduce_gains[channel] = _ask_float("减弱系数, 0~1", 0.8)
            continue
        if _ask_yes("是否增强"):
            enhance_gains[channel] = _ask_float("增强系数, >= 1", 1.2)
    return enhance_gains, reduce_gains


def _make_single_channel_image(bgr: np.ndarray, channel: str) -> np.ndarray:
    split_channels = cv2.split(bgr)
    empty = np.zeros_like(split_channels[0])
    output_channels = [empty, empty, empty]
    output_channels[_CHANNEL_INDEX[channel]] = split_channels[_CHANNEL_INDEX[channel]]
    return cv2.merge(output_channels)


def _show_rgb_channels(bgr: np.ndarray) -> None:
    for channel in _RGB_CHANNELS:
        cv2.imshow(f"{channel.upper()} Channel", _make_single_channel_image(bgr, channel))
    cv2.waitKey(1)


def _print_rgb_means(bgr: np.ndarray) -> None:
    b_mean, g_mean, r_mean, _ = cv2.mean(bgr)
    print(f"R 均值: {r_mean:.2f}")
    print(f"G 均值: {g_mean:.2f}")
    print(f"B 均值: {b_mean:.2f}")


def _run_interactive(image_path: Path, output_path: Path) -> None:
    bgr = cv2.imread(str(image_path))
    if bgr is None:
        raise ValueError(f"图片读取失败: {image_path}")

    _show_rgb_channels(bgr)
    _print_rgb_means(bgr)
    enhance_gains, reduce_gains = _ask_channel_gains()

    adjusted = apply_channel_luts(
        bgr,
        enhance_gains=enhance_gains,
        reduce_gains=reduce_gains,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), adjusted):
        raise ValueError(f"图片保存失败: {output_path}")

    print(f"增强通道: {enhance_gains or '无'}")
    print(f"减弱通道: {reduce_gains or '无'}")
    print(f"结果已保存: {output_path}")
    cv2.imshow("Before", bgr)
    cv2.imshow("After LUT", adjusted)
    for channel in _RGB_CHANNELS:
        cv2.imshow(
            f"After {channel.upper()} Channel",
            _make_single_channel_image(adjusted, channel),
        )
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Interactively apply soft-knee LUT color adjustments.",
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
