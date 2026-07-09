"""Scheme 2: traditional SR + generative SR fusion with a text mask.

This script is independent from the agent state machine. It builds a
conservative traditional SR image from LR, then fuses it with a creative SR
result globally and inside the text mask.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageFilter


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def default_base_dir() -> Path:
    return repo_root() / "workspace" / "SR" / "AllCharac"


def latest_round_image(directory: Path) -> Path:
    candidates = sorted(directory.glob("round_*_out.png"))
    if not candidates:
        raise FileNotFoundError(f"No round_*_out.png found in: {directory}")

    def round_number(path: Path) -> int:
        try:
            return int(path.stem.split("_")[1])
        except (IndexError, ValueError):
            return -1

    return max(candidates, key=round_number)


def load_rgb(path: Path) -> Image.Image:
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")
    return Image.open(path).convert("RGB")


def load_mask(path: Path, size: tuple[int, int], feather_radius: float) -> Image.Image:
    if not path.exists():
        raise FileNotFoundError(f"Mask not found: {path}")
    mask = Image.open(path).convert("L").resize(size, Image.Resampling.NEAREST)
    if feather_radius > 0:
        mask = mask.filter(ImageFilter.GaussianBlur(radius=feather_radius))
    return mask


def make_traditional_sr(
    lr_image: Image.Image,
    size: tuple[int, int],
    method: str,
) -> Image.Image:
    resampling = {
        "lanczos": Image.Resampling.LANCZOS,
        "bicubic": Image.Resampling.BICUBIC,
        "nearest": Image.Resampling.NEAREST,
    }[method]
    return lr_image.resize(size, resampling)


def blend_images(
    creative: Image.Image,
    traditional: Image.Image,
    alpha: float,
) -> Image.Image:
    return Image.blend(traditional, creative, alpha)


def masked_blend(
    creative: Image.Image,
    traditional: Image.Image,
    mask: Image.Image,
    alpha: float,
) -> Image.Image:
    text_region = blend_images(creative, traditional, alpha)
    return Image.composite(text_region, creative, mask)


def save_config(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    base = default_base_dir()
    stem = "DSC_1326_x1"
    parser = argparse.ArgumentParser(
        description="Fuse conservative traditional SR with creative SR using a text mask.",
    )
    parser.add_argument("--stem", default=stem)
    parser.add_argument("--base-dir", type=Path, default=base)
    parser.add_argument("--alpha", type=float, default=0.2, help="Creative SR weight. Default: 0.2.")
    parser.add_argument(
        "--traditional-method",
        choices=["lanczos", "bicubic", "nearest"],
        default="lanczos",
    )
    parser.add_argument("--creative-dir-name", default="masked_baseline")
    parser.add_argument("--output-dir-name", default="scheme2_fusion")
    parser.add_argument("--feather-radius", type=float, default=0.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not 0.0 <= args.alpha <= 1.0:
        raise ValueError("--alpha must be between 0 and 1")

    lr_path = args.base_dir / "LR" / f"{args.stem}.png"
    mask_lr_path = args.base_dir / "craft_text_regions" / args.stem / "mask.png"
    creative_dir = args.base_dir / "outputs" / args.stem / args.creative_dir_name
    creative_path = latest_round_image(creative_dir)
    output_dir = args.base_dir / "outputs" / args.stem / args.output_dir_name
    output_dir.mkdir(parents=True, exist_ok=True)

    lr_image = load_rgb(lr_path)
    creative = load_rgb(creative_path)
    traditional = make_traditional_sr(lr_image, creative.size, args.traditional_method)
    mask_lr = Image.open(mask_lr_path).convert("L")
    mask_sr = load_mask(mask_lr_path, creative.size, args.feather_radius)

    global_fused = blend_images(creative, traditional, args.alpha)
    onlymask_fused = masked_blend(creative, traditional, mask_sr, args.alpha)

    traditional.save(output_dir / "traditional_sr.png")
    shutil.copy2(creative_path, output_dir / "creative_sr.png")
    mask_lr.save(output_dir / "text_mask_lr.png")
    mask_sr.save(output_dir / "text_mask_sr.png")
    global_fused.save(output_dir / f"global_alpha_{args.alpha:.2f}.png")
    onlymask_fused.save(output_dir / f"onlymask_alpha_{args.alpha:.2f}.png")

    save_config(
        output_dir / f"fusion_config_alpha_{args.alpha:.2f}.json",
        {
            "stem": args.stem,
            "alpha": args.alpha,
            "alpha_meaning": "creative_sr weight; traditional_sr weight is 1 - alpha",
            "traditional_method": args.traditional_method,
            "traditional_method_note": (
                "Lanczos is a deterministic windowed-sinc interpolation method. "
                "It upsamples by weighted averaging of neighboring pixels and does "
                "not hallucinate new text strokes."
            ),
            "lr_path": str(lr_path),
            "creative_sr_path": str(creative_path),
            "mask_lr_path": str(mask_lr_path),
            "output_size": list(creative.size),
            "feather_radius": args.feather_radius,
            "outputs": {
                "traditional_sr": "traditional_sr.png",
                "creative_sr": "creative_sr.png",
                "text_mask_lr": "text_mask_lr.png",
                "text_mask_sr": "text_mask_sr.png",
                "global_fusion": f"global_alpha_{args.alpha:.2f}.png",
                "onlymask_fusion": f"onlymask_alpha_{args.alpha:.2f}.png",
            },
        },
    )

    print(f"Saved scheme2 fusion outputs to: {output_dir}")
    print(f"creative_sr={creative_path}")
    print(f"alpha={args.alpha} traditional_method={args.traditional_method}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
