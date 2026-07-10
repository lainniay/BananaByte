from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from agents.zjc.v4.lut import _smoothstep

DEFAULT_OUTPUT_PATH = Path(__file__).with_name("edit_color_curves.png")


def plot_edit_color_curves(  # noqa : D103
    *,
    enhance_gains: list[float],
    reduce_gains: list[float],
    output_path: Path,
    show: bool,
) -> Path:
    values = np.linspace(0, 255, 2048)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(values, values, color="black", linestyle="--", linewidth=1, label="identity")

    for gain in enhance_gains:
        curve = build_enhance_curve(values, gain)
        ax.plot(values, curve, linewidth=2, label=f"enhance {gain:g}")
    for gain in reduce_gains:
        curve = build_reduce_curve(values, gain)
        ax.plot(values, curve, linewidth=2, label=f"reduce {gain:g}")

    ax.set_title("edit_color RGB channel LUT curves")
    ax.set_xlabel("input channel value")
    ax.set_ylabel("output channel value")
    ax.set_xlim(0, 255)
    ax.set_ylim(0, 255)
    ax.grid(True, alpha=0.25)
    ax.legend()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    if show:
        plt.show()
    plt.close(fig)
    return output_path


def build_enhance_curve(values: np.ndarray, gain: float) -> np.ndarray:  # noqa : D103
    if gain < 1.0:
        raise ValueError("enhance gain must be >= 1.0")
    t = values / 255.0
    protect = _smoothstep(0.05, 0.85, t)
    dynamic_gain = gain - (gain - 1.0) * protect
    return np.clip(values * dynamic_gain, 0, 255)


def build_reduce_curve(values: np.ndarray, gain: float) -> np.ndarray:  # noqa : D103
    if not 0.0 < gain <= 1.0:
        raise ValueError("reduce gain must satisfy 0 < gain <= 1")
    t = values / 255.0
    strength = _smoothstep(0.35, 0.9, t)
    dynamic_gain = 1.0 - (1.0 - gain) * strength
    return np.clip(values * dynamic_gain, 0, 255)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot the curve LUTs used by agents.zjc.v4.agent edit_color.",
    )
    parser.add_argument(
        "--enhance",
        type=float,
        nargs="*",
        default=[1.1, 1.2, 1.5],
        help="enhance gains to plot, each must be >= 1.0",
    )
    parser.add_argument(
        "--reduce",
        type=float,
        nargs="*",
        default=[0.9, 0.8],
        help="reduce gains to plot, each must satisfy 0 < gain <= 1.0",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help=f"output image path, default: {DEFAULT_OUTPUT_PATH}",
    )
    parser.add_argument("--show", action="store_true", help="show the plot window")
    return parser.parse_args()


def main() -> None:  # noqa : D103
    args = _parse_args()
    output_path = plot_edit_color_curves(
        enhance_gains=args.enhance,
        reduce_gains=args.reduce,
        output_path=args.output,
        show=args.show,
    )
    print(f"curve image saved: {output_path}")


if __name__ == "__main__":
    main()
