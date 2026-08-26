"""Compare local PaddleOCR detectors with the existing CRAFT result.

The script is detector-only: it does not run text recognition and it never
downloads a model. Each backend writes the same four artifacts used by CRAFT,
so its regions.json can be passed directly to c3stisr_one_image_pipeline.py.
"""

from __future__ import annotations

import argparse
import json
import shutil
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw

from craft_text_region_detect import TextRegion, bbox_from_box, draw_regions, make_mask, region_to_json


DEFAULT_MODELS = ("PP-OCRv5_server_det",)


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare local PaddleOCR text detectors with CRAFT.")
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--stem", help="Image stem under workspace/SR/AllCharac/LR.")
    selection.add_argument("--all", action="store_true", help="Process every LR image.")
    parser.add_argument("--models", nargs="+", default=list(DEFAULT_MODELS))
    parser.add_argument(
        "--model-root",
        type=Path,
        default=Path.home() / ".paddlex" / "official_models",
        help="Existing local PaddleX model cache. Missing models are not downloaded.",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--det-scale", type=float, default=4.0, help="Upscale tiny LR input before detection.")
    parser.add_argument("--limit-side-len", type=int, default=960)
    parser.add_argument("--thresh", type=float, default=0.2)
    parser.add_argument("--box-thresh", type=float, default=0.3)
    parser.add_argument("--unclip-ratio", type=float, default=1.5)
    parser.add_argument("--min-confidence", type=float, default=0.0)
    return parser.parse_args()


def find_image(stem: str) -> Path:
    lr_dir = repo_root() / "workspace" / "SR" / "AllCharac" / "LR"
    for suffix in (".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"):
        candidate = lr_dir / f"{stem}{suffix}"
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"No LR image found for {stem!r} under {lr_dir}")


def result_payload(result: Any) -> Mapping[str, Any]:
    candidates = [result]
    for attr in ("json", "res"):
        value = getattr(result, attr, None)
        candidates.append(value() if callable(value) else value)
    for candidate in candidates:
        if isinstance(candidate, Mapping):
            nested = candidate.get("res")
            return nested if isinstance(nested, Mapping) else candidate
    raise TypeError(f"Unsupported PaddleOCR result type: {type(result).__name__}")


def build_detector(
    model_name: str,
    model_dir: Path,
    args: argparse.Namespace,
) -> Any:
    from paddleocr import TextDetection

    return TextDetection(
        model_name=model_name,
        model_dir=str(model_dir),
        device=args.device,
        enable_mkldnn=False,
        limit_side_len=args.limit_side_len,
        limit_type="max",
        thresh=args.thresh,
        box_thresh=args.box_thresh,
        unclip_ratio=args.unclip_ratio,
    )


def detect(image_bgr: np.ndarray, model: Any, args: argparse.Namespace) -> list[TextRegion]:
    height, width = image_bgr.shape[:2]
    scale = max(1.0, args.det_scale)
    scaled = cv2.resize(image_bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    outputs = model.predict(input=scaled, batch_size=1)
    if not outputs:
        return []

    payload = result_payload(outputs[0])
    polygons = payload.get("dt_polys", [])
    scores = payload.get("dt_scores", [])
    if hasattr(polygons, "tolist"):
        polygons = polygons.tolist()
    if hasattr(scores, "tolist"):
        scores = scores.tolist()

    regions: list[TextRegion] = []
    for index, polygon in enumerate(polygons):
        points = np.asarray(polygon, dtype=np.float32).reshape(-1, 2)
        if len(points) < 4:
            continue
        if len(points) != 4:
            points = cv2.boxPoints(cv2.minAreaRect(points))
        points /= scale
        points[:, 0] = np.clip(points[:, 0], 0, width - 1)
        points[:, 1] = np.clip(points[:, 1], 0, height - 1)
        box = points.tolist()
        confidence = float(scores[index]) if index < len(scores) else None
        if confidence is not None and confidence < args.min_confidence:
            continue
        regions.append(
            TextRegion(
                box=box,
                bbox_xyxy=bbox_from_box(box, width, height),
                confidence=confidence,
            )
        )
    return regions


def save_backend(
    image_path: Path,
    image_bgr: np.ndarray,
    model_name: str,
    regions: list[TextRegion],
    output_dir: Path,
    args: argparse.Namespace,
) -> Path:
    backend_dir = output_dir
    backend_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(image_path, backend_dir / "original.png")
    cv2.imwrite(str(backend_dir / "boxes.png"), draw_regions(image_bgr, regions))
    cv2.imwrite(str(backend_dir / "mask.png"), make_mask(image_bgr.shape, regions))
    payload = {
        "image": image_path.name,
        "source": "LR",
        "detector": "paddleocr_text_detection",
        "model": model_name,
        "coordinates": "original_lr_image",
        "raw_region_count": len(regions),
        "kept_region_count": len(regions),
        "detector_params": {
            "detection_scale": args.det_scale,
            "limit_side_len": args.limit_side_len,
            "thresh": args.thresh,
            "box_thresh": args.box_thresh,
            "unclip_ratio": args.unclip_ratio,
            "min_confidence": args.min_confidence,
        },
        "regions": [region_to_json(region) for region in regions],
    }
    path = backend_dir / "regions.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def save_comparison(stem: str, detector_dir: Path, panels: list[tuple[str, Path]]) -> None:
    loaded: list[tuple[str, Image.Image]] = []
    for label, path in panels:
        if path.is_file():
            loaded.append((label, Image.open(path).convert("RGB")))
    if not loaded:
        return
    label_height = 22
    gap = 4
    panel_size = loaded[0][1].size
    canvas = Image.new(
        "RGB",
        (panel_size[0] * len(loaded) + gap * (len(loaded) - 1), panel_size[1] + label_height),
        "white",
    )
    draw = ImageDraw.Draw(canvas)
    x = 0
    for label, image in loaded:
        canvas.paste(image.resize(panel_size, Image.Resampling.NEAREST), (x, label_height))
        draw.text((x + 3, 3), label, fill="black")
        x += panel_size[0] + gap
    canvas.save(detector_dir / f"{stem}_comparison.png")


def main() -> int:
    args = parse_args()
    lr_dir = repo_root() / "workspace" / "SR" / "AllCharac" / "LR"
    image_paths = (
        sorted(path for path in lr_dir.iterdir() if path.is_file())
        if args.all
        else [find_image(args.stem)]
    )
    summaries: dict[str, list[dict[str, Any]]] = {path.stem: [] for path in image_paths}
    for model_name in args.models:
        model_dir = args.model_root / model_name
        if not model_dir.is_dir():
            print(f"[SKIP] {model_name}: local model missing at {model_dir}")
            for path in image_paths:
                summaries[path.stem].append(
                    {"model": model_name, "status": "missing_local_model", "model_dir": str(model_dir)}
                )
            continue
        model = build_detector(model_name, model_dir, args)
        try:
            for image_path in image_paths:
                image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
                if image_bgr is None:
                    raise RuntimeError(f"Failed to read {image_path}")
                detector_dir = (
                    repo_root()
                    / "workspace"
                    / "SR"
                    / "AllCharac"
                    / "outputs"
                    / image_path.stem
                    / "text_regions"
                )
                regions = detect(image_bgr, model, args)
                backend_name = (
                    "paddle"
                    if model_name == "PP-OCRv5_server_det"
                    else f"paddle_{model_name}"
                )
                regions_path = save_backend(
                    image_path,
                    image_bgr,
                    model_name,
                    regions,
                    detector_dir / backend_name,
                    args,
                )
                summaries[image_path.stem].append(
                    {
                        "model": model_name,
                        "status": "ok",
                        "region_count": len(regions),
                        "regions": str(regions_path),
                    }
                )
                print(f"[OK] {image_path.stem} / {model_name}: {len(regions)} region(s)")
        finally:
            close = getattr(model, "close", None)
            if callable(close):
                close()

    for image_path in image_paths:
        image_output = repo_root() / "workspace" / "SR" / "AllCharac" / "outputs" / image_path.stem
        review_dir = image_output / "text_regions"
        paddle_dir = review_dir / "paddle"
        panels = [
            ("CRAFT", review_dir / "craft" / "boxes.png"),
            ("Paddle", paddle_dir / "boxes.png"),
        ]
        save_comparison(image_path.stem, review_dir, panels)
        (paddle_dir / "run.json").write_text(
            json.dumps(
                {"image": str(image_path), "backends": summaries[image_path.stem]},
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
    print(f"processed images: {len(image_paths)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
