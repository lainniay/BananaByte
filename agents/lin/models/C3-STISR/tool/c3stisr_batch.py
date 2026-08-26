"""Run the complete C3-STISR experiment for every LR image with CRAFT regions."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def repo_root() -> Path:
    return Path(__file__).resolve().parents[5]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch C3-STISR over AllCharac LR images.")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("workspace/SR/C3-STISR-RecLing-Final/model_best_0.pth"),
    )
    parser.add_argument(
        "--recognizer-checkpoint",
        type=Path,
        default=Path("workspace/SR/C3-STISR-RecLing-Final/recognizer_best_0.pth"),
    )
    parser.add_argument("--min-aspect", type=float, default=0.0)
    parser.add_argument("--feather", type=int, default=3)
    parser.add_argument(
        "--paste-mode",
        choices=("quad", "text-mask", "text-mask-color-match", "text-mask-luma", "detail-compare"),
        default="detail-compare",
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--clean", action="store_true", help="Remove each old color-compare folder before running.")
    return parser.parse_args()


def clean_c3_dir(path: Path, image_output_dir: Path) -> None:
    resolved = path.resolve()
    expected = (image_output_dir / "c3stisr_color_compare").resolve()
    if resolved != expected or resolved.name != "c3stisr_color_compare":
        raise ValueError(f"Refusing to clean unexpected path: {resolved}")
    if resolved.is_dir():
        shutil.rmtree(resolved)


def save_overview(outputs_dir: Path) -> Path | None:
    rows = []
    for image_output in sorted(path for path in outputs_dir.iterdir() if path.is_dir()):
        comparison = image_output / "c3stisr_color_compare" / "comparison.png"
        if comparison.is_file():
            rows.append((image_output.name, Image.open(comparison).convert("RGB")))
    if not rows:
        return None
    label_height = 18
    width = max(image.width for _, image in rows)
    height = sum(image.height + label_height for _, image in rows)
    canvas = Image.new("RGB", (width, height), "white")
    painter = ImageDraw.Draw(canvas)
    y = 0
    for stem, image in rows:
        painter.text((3, y + 2), stem, fill="black")
        y += label_height
        canvas.paste(image, (0, y))
        y += image.height
    output = outputs_dir / "c3stisr_color_compare_overview.png"
    canvas.save(output)
    return output


def save_debug_diffs(outputs_dir: Path) -> int:
    written = 0
    for image_output in sorted(path for path in outputs_dir.iterdir() if path.is_dir()):
        experiment = image_output / "c3stisr_color_compare"
        base_path = experiment / "debug" / "bicubic_2x.png"
        result_paths = {
            "post_y": experiment / "result_post_y.png",
            "post_v": experiment / "result_post_v.png",
            "pre_y": experiment / "result_pre_y.png",
            "pre_v": experiment / "result_pre_v.png",
        }
        if not base_path.is_file() or not all(path.is_file() for path in result_paths.values()):
            continue
        base = np.asarray(Image.open(base_path).convert("RGB"), dtype=np.int16)
        results = {
            name: np.asarray(Image.open(path).convert("RGB"), dtype=np.int16)
            for name, path in result_paths.items()
        }
        debug_dir = experiment / "debug"
        for name, first, second, gain in (
            ("diff_post_y_vs_bicubic_x4.png", results["post_y"], base, 4),
            ("diff_post_v_vs_bicubic_x4.png", results["post_v"], base, 4),
            ("diff_pre_y_vs_bicubic_x4.png", results["pre_y"], base, 4),
            ("diff_pre_v_vs_bicubic_x4.png", results["pre_v"], base, 4),
            ("diff_pre_vs_post_y_x8.png", results["pre_y"], results["post_y"], 8),
            ("diff_pre_vs_post_v_x8.png", results["pre_v"], results["post_v"], 8),
        ):
            difference = np.clip(np.abs(first - second) * gain, 0, 255).astype(np.uint8)
            Image.fromarray(difference, mode="RGB").save(debug_dir / name)
        written += 1
    return written


def main() -> int:
    args = parse_args()
    root = repo_root()
    dataset = root / "workspace" / "SR" / "AllCharac"
    lr_dir = dataset / "LR"
    outputs_dir = dataset / "outputs"
    pipeline = root / "agents" / "lin" / "c3stisr_one_image_pipeline.py"
    output_dirs = {path.name.lower(): path for path in outputs_dir.iterdir() if path.is_dir()}

    summary: list[dict[str, object]] = []
    for image_path in sorted(path for path in lr_dir.iterdir() if path.is_file()):
        image_output_dir = output_dirs.get(image_path.stem.lower())
        regions_path = image_output_dir / "text_regions" / "union" / "regions.json" if image_output_dir else None
        if regions_path is None or not regions_path.is_file():
            summary.append({"stem": image_path.stem, "status": "skipped", "reason": "missing_regions"})
            continue

        c3_dir = image_output_dir / "c3stisr_color_compare"
        if args.clean:
            clean_c3_dir(c3_dir, image_output_dir)

        command = [
            sys.executable,
            str(pipeline),
            "--image",
            str(image_path),
            "--stem",
            image_output_dir.name,
            "--regions",
            str(regions_path),
            "--output-name",
            "c3stisr_color_compare",
            "--checkpoint",
            str((root / args.checkpoint).resolve() if not args.checkpoint.is_absolute() else args.checkpoint),
            "--recognizer-checkpoint",
            str(
                (root / args.recognizer_checkpoint).resolve()
                if not args.recognizer_checkpoint.is_absolute()
                else args.recognizer_checkpoint
            ),
            "--prior-mode",
            "rec-ling",
            "--paste-mode",
            args.paste_mode,
            "--save-mode",
            "clamp",
            "--min-aspect",
            str(args.min_aspect),
            "--min-width",
            "1",
            "--min-height",
            "1",
            "--max-aspect",
            "100",
            "--max-regions",
            "100",
            "--feather",
            str(args.feather),
            "--device",
            args.device,
        ]
        print(f"\n=== {image_output_dir.name} ===", flush=True)
        try:
            subprocess.run(command, cwd=root, check=True)
            run_record = json.loads((c3_dir / "run.json").read_text(encoding="utf-8"))
            summary.append(
                {
                    "stem": image_output_dir.name,
                    "status": "ok",
                    "processed_region_count": run_record["processed_region_count"],
                    "post_y": str(c3_dir / "result_post_y.png"),
                    "post_v": str(c3_dir / "result_post_v.png"),
                    "pre_y": str(c3_dir / "result_pre_y.png"),
                    "pre_v": str(c3_dir / "result_pre_v.png"),
                    "comparison": str(c3_dir / "comparison.png"),
                }
            )
        except Exception as exc:
            summary.append({"stem": image_output_dir.name, "status": "error", "reason": str(exc)})

    summary_path = outputs_dir / "c3stisr_color_compare_batch_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    save_debug_diffs(outputs_dir)
    overview_path = save_overview(outputs_dir)
    ok_count = sum(item["status"] == "ok" for item in summary)
    print(f"\nCompleted: {ok_count}/{len(summary)}")
    print(f"Summary: {summary_path}")
    print(f"Overview: {overview_path}")
    return 0 if ok_count == len(summary) else 1


if __name__ == "__main__":
    raise SystemExit(main())
