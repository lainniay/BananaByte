"""Conditionally combine CRAFT and PaddleOCR text-region candidates."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw


RULES = {
    "overlap_metric": "intersection_over_smaller_bbox",
    "overlap_threshold": 0.50,
    "paddle_single_min_confidence": 0.65,
    "craft_single_min_confidence": 0.20,
    "max_image_area_ratio": 0.25,
    "minimum_width": 4,
    "minimum_height": 3,
}


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Combine CRAFT and Paddle text boxes.")
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--stem")
    selection.add_argument("--all", action="store_true")
    return parser.parse_args()


def load_regions(path: Path, source: str) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    output = []
    for index, region in enumerate(payload.get("regions", [])):
        item = dict(region)
        item["source"] = source
        item["source_index"] = index
        output.append(item)
    return output


def bbox(region: dict[str, Any]) -> np.ndarray:
    return np.asarray(region["bbox_xyxy"], dtype=np.float32)


def intersection_over_smaller(first: dict[str, Any], second: dict[str, Any]) -> float:
    a, b = bbox(first), bbox(second)
    width = max(0.0, min(a[2], b[2]) - max(a[0], b[0]) + 1.0)
    height = max(0.0, min(a[3], b[3]) - max(a[1], b[1]) + 1.0)
    intersection = width * height
    area_a = max(1.0, (a[2] - a[0] + 1.0) * (a[3] - a[1] + 1.0))
    area_b = max(1.0, (b[2] - b[0] + 1.0) * (b[3] - b[1] + 1.0))
    return intersection / min(area_a, area_b)


def candidate_groups(
    regions: list[dict[str, Any]], cross_source_only: bool = True
) -> list[list[dict[str, Any]]]:
    parent = list(range(len(regions)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(first: int, second: int) -> None:
        root_a, root_b = find(first), find(second)
        if root_a != root_b:
            parent[root_b] = root_a

    for first in range(len(regions)):
        for second in range(first + 1, len(regions)):
            if cross_source_only and regions[first]["source"] == regions[second]["source"]:
                continue
            if intersection_over_smaller(regions[first], regions[second]) >= RULES["overlap_threshold"]:
                union(first, second)

    groups: dict[int, list[dict[str, Any]]] = {}
    for index, region in enumerate(regions):
        groups.setdefault(find(index), []).append(region)
    return list(groups.values())


def merged_box(group: list[dict[str, Any]]) -> list[list[float]]:
    if len(group) == 1:
        return group[0]["box"]
    points = np.concatenate([np.asarray(region["box"], dtype=np.float32) for region in group])
    return cv2.boxPoints(cv2.minAreaRect(points)).astype(float).tolist()


def box_bbox(box: list[list[float]], width: int, height: int) -> list[int]:
    points = np.asarray(box)
    return [
        max(0, int(np.floor(points[:, 0].min()))),
        max(0, int(np.floor(points[:, 1].min()))),
        min(width - 1, int(np.ceil(points[:, 0].max()))),
        min(height - 1, int(np.ceil(points[:, 1].max()))),
    ]


def decide(group: list[dict[str, Any]], image_shape: tuple[int, int, int]) -> tuple[bool, str]:
    height, width = image_shape[:2]
    merged = merged_box(group)
    x0, y0, x1, y1 = box_bbox(merged, width, height)
    box_width, box_height = x1 - x0 + 1, y1 - y0 + 1
    if box_width < RULES["minimum_width"] or box_height < RULES["minimum_height"]:
        return False, "too_small"
    if box_width * box_height / max(1, width * height) > RULES["max_image_area_ratio"]:
        return False, "too_large"

    sources = {region["source"] for region in group}
    if sources == {"craft", "paddle"}:
        return True, "craft_paddle_agreement"
    confidence = max(float(region.get("confidence") or 0.0) for region in group)
    if sources == {"paddle"} and confidence >= RULES["paddle_single_min_confidence"]:
        return True, "paddle_high_confidence"
    if sources == {"craft"} and confidence >= RULES["craft_single_min_confidence"]:
        return True, "craft_high_confidence"
    return False, "single_detector_low_confidence"


def build_record(
    group: list[dict[str, Any]], image_shape: tuple[int, int, int]
) -> dict[str, Any]:
    height, width = image_shape[:2]
    box = merged_box(group)
    accepted, reason = decide(group, image_shape)
    confidences = {region["source"]: region.get("confidence") for region in group}
    return {
        "box": box,
        "bbox_xyxy": box_bbox(box, width, height),
        "confidence": max(float(value or 0.0) for value in confidences.values()),
        "debug_text": "",
        "sources": sorted(confidences),
        "source_confidences": confidences,
        "accepted": accepted,
        "accept_reason": reason,
        "members": [
            {"source": region["source"], "source_index": region["source_index"]}
            for region in group
        ],
    }


def draw(image: np.ndarray, records: list[dict[str, Any]], accepted_only: bool) -> np.ndarray:
    output = image.copy()
    shown = [record for record in records if record["accepted"] or not accepted_only]
    for index, record in enumerate(shown, start=1):
        color = (0, 220, 0) if record["accepted"] else (0, 0, 255)
        points = np.asarray(record["box"], dtype=np.int32).reshape(-1, 1, 2)
        cv2.polylines(output, [points], True, color, 1)
        x, y = record["bbox_xyxy"][:2]
        cv2.putText(output, str(index), (x, max(7, y)), cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)
    return output


def mask_for(image_shape: tuple[int, int, int], records: list[dict[str, Any]]) -> np.ndarray:
    mask = np.zeros(image_shape[:2], dtype=np.uint8)
    for record in records:
        points = np.asarray(record["box"], dtype=np.int32).reshape(-1, 1, 2)
        cv2.fillPoly(mask, [points], 255)
    return mask


def save_region_set(
    output_dir: Path,
    image_path: Path,
    image: np.ndarray,
    records: list[dict[str, Any]],
    detector: str,
    rules: dict[str, Any],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(image_path, output_dir / "original.png")
    cv2.imwrite(str(output_dir / "boxes.png"), draw(image, records, accepted_only=True))
    cv2.imwrite(str(output_dir / "mask.png"), mask_for(image.shape, records))
    payload = {
        "image": image_path.name,
        "detector": detector,
        "coordinates": "original_lr_image",
        "rules": rules,
        "kept_region_count": len(records),
        "regions": records,
    }
    (output_dir / "regions.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def comparison(panels: list[tuple[str, Path]], output_path: Path) -> None:
    images = [(label, Image.open(path).convert("RGB")) for label, path in panels if path.is_file()]
    if not images:
        return
    width, height = images[0][1].size
    label_height = 20
    canvas = Image.new("RGB", (width * len(images), height + label_height), "white")
    painter = ImageDraw.Draw(canvas)
    for index, (label, image) in enumerate(images):
        x = index * width
        painter.text((x + 2, 2), label, fill="black")
        canvas.paste(image.resize((width, height), Image.Resampling.NEAREST), (x, label_height))
    canvas.save(output_path)


def save_overview(stems: list[str], outputs_dir: Path) -> None:
    rows = []
    for stem in stems:
        path = outputs_dir / stem / "text_regions" / "comparison.png"
        if path.is_file():
            rows.append((stem, Image.open(path).convert("RGB")))
    if not rows:
        return
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
    canvas.save(outputs_dir / "text_region_ensemble_overview.png")


def process(stem: str) -> dict[str, Any]:
    root = repo_root() / "workspace" / "SR" / "AllCharac"
    lr_candidates = list((root / "LR").glob(f"{stem}.*"))
    if not lr_candidates:
        raise FileNotFoundError(f"LR image not found: {stem}")
    image_path = lr_candidates[0]
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Failed to read {image_path}")

    image_output = root / "outputs" / stem
    review_dir = image_output / "text_regions"
    craft_path = review_dir / "craft" / "regions.json"
    paddle_path = review_dir / "paddle" / "regions.json"
    regions = load_regions(craft_path, "craft") + load_regions(paddle_path, "paddle")
    conditional_records = [build_record(group, image.shape) for group in candidate_groups(regions)]
    conditional = [record for record in conditional_records if record["accepted"]]
    union = []
    for group in candidate_groups(regions, cross_source_only=False):
        record = build_record(group, image.shape)
        record["accepted"] = True
        record["accept_reason"] = "unconditional_union"
        union.append(record)

    conditional_dir = review_dir / "conditional"
    save_region_set(
        conditional_dir,
        image_path,
        image,
        conditional,
        "conditional_union_craft_paddle",
        RULES,
    )
    cv2.imwrite(
        str(conditional_dir / "candidates.png"),
        draw(image, conditional_records, accepted_only=False),
    )
    (conditional_dir / "candidates.json").write_text(
        json.dumps(
            {
                "image": image_path.name,
                "rules": RULES,
                "candidate_count": len(conditional_records),
                "regions": conditional_records,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    union_dir = review_dir / "union"
    save_region_set(
        union_dir,
        image_path,
        image,
        union,
        "unconditional_union_craft_paddle",
        {
            "accept": "all_candidates",
            "overlap_merge_metric": RULES["overlap_metric"],
            "overlap_merge_threshold": RULES["overlap_threshold"],
            "confidence_filter": None,
            "geometry_filter": None,
        },
    )
    comparison(
        [
            ("CRAFT", craft_path.parent / "boxes.png"),
            ("Paddle", paddle_path.parent / "boxes.png"),
            ("conditional", conditional_dir / "boxes.png"),
            ("union", union_dir / "boxes.png"),
        ],
        review_dir / "comparison.png",
    )
    print(
        f"[OK] {stem}: conditional {len(conditional)} / "
        f"{len(conditional_records)}, union {len(union)}"
    )
    return {
        "stem": stem,
        "conditional": len(conditional),
        "candidates": len(conditional_records),
        "union": len(union),
    }


def main() -> int:
    args = parse_args()
    lr_dir = repo_root() / "workspace" / "SR" / "AllCharac" / "LR"
    stems = [path.stem for path in sorted(lr_dir.iterdir()) if path.is_file()] if args.all else [args.stem]
    summary = [process(stem) for stem in stems]
    outputs_dir = repo_root() / "workspace" / "SR" / "AllCharac" / "outputs"
    save_overview(stems, outputs_dir)
    output = outputs_dir / "text_region_ensemble_summary.json"
    output.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
