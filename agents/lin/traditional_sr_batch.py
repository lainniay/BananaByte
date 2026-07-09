"""Batch traditional SR baselines for AllCharac LR images.

This script is independent from the agent state machine. It uses deterministic
PIL interpolation only, so it does not call any model API.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

from PIL import Image


IMAGE_EXTS = {".png", ".jpg", ".jpeg"}


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def default_base_dir() -> Path:
    return repo_root() / "workspace" / "SR" / "AllCharac"


def resampling_method(name: str) -> Image.Resampling:
    methods = {
        "lanczos": Image.Resampling.LANCZOS,
        "bicubic": Image.Resampling.BICUBIC,
        "nearest": Image.Resampling.NEAREST,
    }
    if name not in methods:
        raise ValueError(f"unsupported method: {name}")
    return methods[name]


def save_config(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def target_size_for(lr_path: Path, hr_dir: Path, scale: int) -> tuple[int, int]:
    hr_path = hr_dir / lr_path.name
    if hr_path.exists():
        with Image.open(hr_path) as hr:
            return hr.size

    with Image.open(lr_path) as lr:
        width, height = lr.size
    return width * scale, height * scale


def run_one(
    lr_path: Path,
    base_dir: Path,
    method: str,
    scale: int,
    output_dir_name: str,
) -> Path:
    output_dir = base_dir / "outputs" / lr_path.stem / output_dir_name
    output_dir.mkdir(parents=True, exist_ok=True)

    with Image.open(lr_path) as lr:
        lr_image = lr.convert("RGB")
    target_size = target_size_for(lr_path, base_dir / "HR", scale)
    traditional = lr_image.resize(target_size, resampling_method(method))

    original_copy = output_dir / "original_lr.png"
    traditional_path = output_dir / "traditional_sr.png"
    config_path = output_dir / "traditional_sr_config.json"

    shutil.copy2(lr_path, original_copy)
    traditional.save(traditional_path)
    save_config(
        config_path,
        {
            "image": lr_path.name,
            "source": "LR",
            "method": method,
            "method_note": (
                "Deterministic interpolation baseline. It upsamples existing "
                "pixels only and does not hallucinate new text strokes."
            ),
            "lr_path": str(lr_path),
            "output_size": list(target_size),
            "scale_used_when_hr_missing": scale,
            "outputs": {
                "original_lr": original_copy.name,
                "traditional_sr": traditional_path.name,
            },
        },
    )
    return traditional_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate traditional SR baselines for AllCharac LR images.",
    )
    parser.add_argument("--base-dir", type=Path, default=default_base_dir())
    parser.add_argument(
        "--method",
        choices=["lanczos", "bicubic", "nearest"],
        default="lanczos",
    )
    parser.add_argument("--scale", type=int, default=4)
    parser.add_argument("--output-dir-name", default="traditional_sr")
    parser.add_argument(
        "--exclude",
        action="append",
        default=["Canon_049_LR4"],
        help="Image stem to skip. Can be passed multiple times.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    lr_dir = args.base_dir / "LR"
    if not lr_dir.exists():
        raise FileNotFoundError(f"LR directory not found: {lr_dir}")

    excluded = set(args.exclude or [])
    lr_paths = sorted(p for p in lr_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS)
    processed: list[str] = []
    skipped: list[str] = []

    for lr_path in lr_paths:
        if lr_path.stem in excluded:
            skipped.append(lr_path.stem)
            continue
        out = run_one(
            lr_path=lr_path,
            base_dir=args.base_dir,
            method=args.method,
            scale=args.scale,
            output_dir_name=args.output_dir_name,
        )
        processed.append(str(out))
        print(f"saved: {out}")

    print(f"processed={len(processed)} skipped={skipped}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
