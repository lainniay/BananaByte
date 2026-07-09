"""Detect text-like regions with EasyOCR's CRAFT detector.

This is an independent experiment module under agents/lin. It does not touch
the existing agent flow. The goal is text-region localization, not reliable OCR
transcription; recognized text is saved as debug-only metadata.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


@dataclass
class TextRegion:
    box: list[list[float]]
    bbox_xyxy: list[int]
    confidence: float | None = None
    debug_text: str = ""


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def default_input_dir() -> Path:
    return repo_root() / "workspace" / "SR" / "AllCharac" / "LR"


def default_output_dir() -> Path:
    return repo_root() / "workspace" / "SR" / "AllCharac" / "craft_text_regions"


def check_runtime_dependencies() -> None:
    try:
        import easyocr  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "Missing CRAFT detector dependency: easyocr. "
            "Install it first if you want to run this experiment, for example: "
            "pip install easyocr"
        ) from exc


def list_images(input_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in input_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def bbox_from_box(box: list[list[float]], width: int, height: int) -> list[int]:
    xs = [point[0] for point in box]
    ys = [point[1] for point in box]
    return [
        int(max(0, min(xs))),
        int(max(0, min(ys))),
        int(min(width - 1, max(xs))),
        int(min(height - 1, max(ys))),
    ]


def normalize_box(value: Any) -> list[list[float]] | None:
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if not isinstance(value, (list, tuple)) or len(value) < 4:
        return None

    points: list[list[float]] = []
    for point in value[:4]:
        if not isinstance(point, (list, tuple, np.ndarray)) or len(point) < 2:
            return None
        try:
            points.append([float(point[0]), float(point[1])])
        except (TypeError, ValueError):
            return None
    return points


def resize_for_detection(image: np.ndarray, scale: float) -> tuple[np.ndarray, float, float]:
    if scale <= 1.0:
        return image, 1.0, 1.0
    height, width = image.shape[:2]
    new_width = max(1, int(round(width * scale)))
    new_height = max(1, int(round(height * scale)))
    resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
    return resized, new_width / width, new_height / height


def scale_box_to_original(
    box: list[list[float]],
    scale_x: float,
    scale_y: float,
) -> list[list[float]]:
    return [[point[0] / scale_x, point[1] / scale_y] for point in box]


def build_reader(languages: list[str], gpu: bool) -> Any:
    import easyocr

    return easyocr.Reader(languages, gpu=gpu)


def detect_regions(
    reader: Any,
    image: np.ndarray,
    det_scale: float,
    mag_ratio: float,
    canvas_size: int,
    text_threshold: float,
    low_text: float,
    link_threshold: float,
) -> list[TextRegion]:
    original_height, original_width = image.shape[:2]
    detection_image, scale_x, scale_y = resize_for_detection(image, det_scale)
    rgb_image = cv2.cvtColor(detection_image, cv2.COLOR_BGR2RGB)

    results = reader.readtext(
        rgb_image,
        detail=1,
        paragraph=False,
        decoder="greedy",
        mag_ratio=mag_ratio,
        canvas_size=canvas_size,
        text_threshold=text_threshold,
        low_text=low_text,
        link_threshold=link_threshold,
    )

    regions: list[TextRegion] = []
    for result in results:
        if not isinstance(result, (list, tuple)) or len(result) < 1:
            continue
        box = normalize_box(result[0])
        if box is None:
            continue
        box = scale_box_to_original(box, scale_x, scale_y)
        debug_text = str(result[1]) if len(result) > 1 else ""
        confidence = None
        if len(result) > 2:
            try:
                confidence = float(result[2])
            except (TypeError, ValueError):
                confidence = None
        regions.append(
            TextRegion(
                box=box,
                bbox_xyxy=bbox_from_box(box, original_width, original_height),
                confidence=confidence,
                debug_text=debug_text,
            )
        )
    return regions


def should_keep_region(
    region: TextRegion,
    image_shape: tuple[int, int, int],
    args: argparse.Namespace,
) -> bool:
    height, width = image_shape[:2]
    x_min, y_min, x_max, y_max = region.bbox_xyxy
    box_width = x_max - x_min + 1
    box_height = y_max - y_min + 1
    area = box_width * box_height
    image_area = width * height

    if region.confidence is not None and region.confidence < args.min_confidence:
        return False
    if box_width < args.min_width or box_height < args.min_height:
        return False
    if area < args.min_area:
        return False
    if image_area > 0 and area / image_area > args.max_area_ratio:
        return False
    if height > 0 and box_height / height > args.max_height_ratio:
        return False

    aspect_ratio = box_width / max(1, box_height)
    return args.min_aspect_ratio <= aspect_ratio <= args.max_aspect_ratio


def filter_regions(
    regions: list[TextRegion],
    image_shape: tuple[int, int, int],
    args: argparse.Namespace,
) -> list[TextRegion]:
    if not args.filter:
        return regions
    return [region for region in regions if should_keep_region(region, image_shape, args)]


def draw_regions(image: np.ndarray, regions: list[TextRegion]) -> np.ndarray:
    output = image.copy()
    for index, region in enumerate(regions, start=1):
        points = np.array(region.box, dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(output, [points], isClosed=True, color=(0, 255, 0), thickness=2)
        x_min, y_min, _, _ = region.bbox_xyxy
        cv2.putText(
            output,
            str(index),
            (x_min, max(0, y_min - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 0, 255),
            1,
            cv2.LINE_AA,
        )
    return output


def make_mask(image_shape: tuple[int, int, int], regions: list[TextRegion]) -> np.ndarray:
    height, width = image_shape[:2]
    mask = np.zeros((height, width), dtype=np.uint8)
    for region in regions:
        points = np.array(region.box, dtype=np.int32).reshape((-1, 1, 2))
        cv2.fillPoly(mask, [points], color=255)
    return mask


def region_to_json(region: TextRegion) -> dict[str, Any]:
    return {
        "box": region.box,
        "bbox_xyxy": region.bbox_xyxy,
        "confidence": region.confidence,
        "debug_text": region.debug_text,
    }


def save_outputs(
    image_path: Path,
    output_root: Path,
    image: np.ndarray,
    regions: list[TextRegion],
    raw_region_count: int,
    args: argparse.Namespace,
) -> None:
    image_output_dir = output_root / image_path.stem
    image_output_dir.mkdir(parents=True, exist_ok=True)

    original_path = image_output_dir / "original.png"
    boxes_path = image_output_dir / "boxes.png"
    mask_path = image_output_dir / "mask.png"
    json_path = image_output_dir / "regions.json"

    if image_path.suffix.lower() == ".png":
        shutil.copy2(image_path, original_path)
    else:
        cv2.imwrite(str(original_path), image)
    cv2.imwrite(str(boxes_path), draw_regions(image, regions))
    cv2.imwrite(str(mask_path), make_mask(image.shape, regions))

    payload = {
        "image": image_path.name,
        "source": "LR",
        "detector": "easyocr_craft",
        "text_is_debug_only": True,
        "coordinates": "original_lr_image",
        "detection_scale": args.det_scale,
        "raw_region_count": raw_region_count,
        "kept_region_count": len(regions),
        "easyocr_languages": args.languages,
        "easyocr_params": {
            "mag_ratio": args.mag_ratio,
            "canvas_size": args.canvas_size,
            "text_threshold": args.text_threshold,
            "low_text": args.low_text,
            "link_threshold": args.link_threshold,
        },
        "post_filter": {
            "enabled": args.filter,
            "min_confidence": args.min_confidence,
            "min_width": args.min_width,
            "min_height": args.min_height,
            "min_area": args.min_area,
            "max_area_ratio": args.max_area_ratio,
            "max_height_ratio": args.max_height_ratio,
            "min_aspect_ratio": args.min_aspect_ratio,
            "max_aspect_ratio": args.max_aspect_ratio,
        },
        "regions": [region_to_json(region) for region in regions],
    }
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def process_images(args: argparse.Namespace) -> tuple[int, int]:
    if not args.input_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {args.input_dir}")
    images = list_images(args.input_dir)
    if not images:
        raise FileNotFoundError(f"No supported images found in: {args.input_dir}")

    check_runtime_dependencies()
    reader = build_reader(args.languages, args.gpu)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    processed = 0
    failed = 0
    selected = set(args.only or [])
    for image_path in images:
        if selected and image_path.name not in selected and image_path.stem not in selected:
            continue
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            failed += 1
            print(f"[WARN] Failed to read image: {image_path}", file=sys.stderr)
            continue
        try:
            raw_regions = detect_regions(
                reader,
                image,
                args.det_scale,
                args.mag_ratio,
                args.canvas_size,
                args.text_threshold,
                args.low_text,
                args.link_threshold,
            )
            regions = filter_regions(raw_regions, image.shape, args)
            save_outputs(image_path, args.output_dir, image, regions, len(raw_regions), args)
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"[WARN] Failed to process {image_path.name}: {exc}", file=sys.stderr)
            continue
        processed += 1
        print(f"[OK] {image_path.name}: {len(regions)} kept / {len(raw_regions)} raw region(s)")
    return processed, failed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Detect text regions in AllCharac LR images with EasyOCR CRAFT.",
    )
    parser.add_argument("--input-dir", type=Path, default=default_input_dir())
    parser.add_argument("--output-dir", type=Path, default=default_output_dir())
    parser.add_argument("--languages", nargs="+", default=["en"])
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument("--det-scale", type=float, default=4.0)
    parser.add_argument("--mag-ratio", type=float, default=2.0)
    parser.add_argument("--canvas-size", type=int, default=2048)
    parser.add_argument("--text-threshold", type=float, default=0.4)
    parser.add_argument("--low-text", type=float, default=0.3)
    parser.add_argument("--link-threshold", type=float, default=0.3)
    parser.add_argument("--only", nargs="*", default=None, help="Optional image names or stems to process.")
    parser.add_argument("--filter", action="store_true", help="Enable geometry/confidence post-filtering.")
    parser.add_argument("--min-confidence", type=float, default=0.2)
    parser.add_argument("--min-width", type=int, default=4)
    parser.add_argument("--min-height", type=int, default=4)
    parser.add_argument("--min-area", type=int, default=60)
    parser.add_argument("--max-area-ratio", type=float, default=0.18)
    parser.add_argument("--max-height-ratio", type=float, default=0.75)
    parser.add_argument("--min-aspect-ratio", type=float, default=0.35)
    parser.add_argument("--max-aspect-ratio", type=float, default=12.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        processed, failed = process_images(args)
    except Exception as exc:  # noqa: BLE001
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1
    print(f"Done. processed={processed}, failed={failed}, output={args.output_dir}")
    return 0 if failed == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
