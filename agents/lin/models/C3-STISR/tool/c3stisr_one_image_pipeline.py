"""Run one full-image C3-STISR paste-back experiment.

This script stays under agents/lin and is intentionally independent from the
shared state machine. It uses existing CRAFT regions to crop text, normalizes
each text crop for C3-STISR, runs the C3 checkpoint, and pastes the enhanced
crop back onto a 2x canvas for visual debugging.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw

from c3stisr_infer import (
    build_model,
    clue_debug,
    image_to_tensor,
    infer_with_priors,
    load_c3_module,
    load_recognizer,
    pushd,
    strip_module_prefix,
    tensor_to_image,
)


IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff")
DETAIL_SIGMA = 1.0
DETAIL_LIMIT = 32.0
DETAIL_STRENGTH = 1.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="One-image C3-STISR crop and paste-back experiment.")
    parser.add_argument("--stem", type=str, default=None, help="Image stem under workspace/SR/AllCharac/LR.")
    parser.add_argument("--image", type=Path, default=None, help="Explicit LR full-image path.")
    parser.add_argument("--regions", type=Path, default=None, help="Explicit CRAFT regions.json path.")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("workspace/SR/C3-STISR-RecLing-Final/model_best_0.pth"),
        help="C3-STISR checkpoint path.",
    )
    parser.add_argument(
        "--c3-root",
        type=Path,
        default=Path("agents/lin/models/C3-STISR/upstream"),
        help="C3-STISR repository root.",
    )
    parser.add_argument(
        "--tpgsr-root",
        type=Path,
        default=Path("agents/lin/models/TPGSR/upstream"),
        help="TPGSR repository root containing C3's CRNN definition.",
    )
    parser.add_argument(
        "--recognizer-checkpoint",
        type=Path,
        default=None,
        help="CRNN checkpoint. Defaults to recognizer_best_0.pth beside --checkpoint.",
    )
    parser.add_argument(
        "--prior-mode",
        choices=("none", "rec", "rec-ling"),
        default="rec-ling",
        help="C3 clue path. rec-ling reproduces the released full inference path.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help=(
            "Directory where debug outputs are written. By default, outputs go "
            "under the source image folder: <dataset>/outputs/<stem>/c3stisr."
        ),
    )
    parser.add_argument("--output-name", default="c3stisr", help="Per-stem experiment folder name.")
    parser.add_argument("--crop-mode", choices=("quad", "bbox"), default="quad")
    parser.add_argument("--scale", type=int, default=2, help="Paste-back scale for the full-image canvas.")
    parser.add_argument("--min-width", type=float, default=8.0, help="Skip regions narrower than this in LR pixels.")
    parser.add_argument("--min-height", type=float, default=6.0, help="Skip regions shorter than this in LR pixels.")
    parser.add_argument("--min-aspect", type=float, default=1.2, help="Skip regions below this width/height ratio.")
    parser.add_argument("--max-aspect", type=float, default=14.0, help="Skip regions above this width/height ratio.")
    parser.add_argument("--max-regions", type=int, default=20, help="Maximum accepted regions to process.")
    parser.add_argument(
        "--paste-mode",
        choices=("quad", "text-mask", "text-mask-color-match", "text-mask-luma", "detail-compare"),
        default="detail-compare",
        help=(
            "quad pastes the whole crop; text-mask pastes only polarity-corrected text pixels; "
            "text-mask-color-match matches all three C3 RGB channels to the bicubic canvas; "
            "text-mask-luma keeps C3 luminance while preserving bicubic chroma; detail-compare "
            "runs C3 once and writes both YCbCr-detail and HSV-detail fusion results."
        ),
    )
    parser.add_argument("--feather", type=int, default=3, help="Odd blur kernel for paste mask feathering; 0 disables.")
    parser.add_argument("--save-mode", choices=("tanh", "clamp"), default="clamp")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--verbose-import", action="store_true")
    return parser.parse_args()


def repo_root() -> Path:
    return Path(__file__).resolve().parents[5]


def resolve_from_repo(path: Path) -> Path:
    if path.is_absolute():
        return path
    return (repo_root() / path).resolve()


def default_lr_dir() -> Path:
    return repo_root() / "workspace" / "SR" / "AllCharac" / "LR"


def default_regions_root() -> Path:
    return repo_root() / "workspace" / "SR" / "AllCharac" / "outputs"


def legacy_regions_root() -> Path:
    return repo_root() / "workspace" / "SR" / "AllCharac" / "craft_text_regions"


def default_output_root(image_path: Path) -> Path:
    parent_name = image_path.parent.name.lower()
    if parent_name in {"lr", "hr"}:
        dataset_root = image_path.parent.parent
    else:
        dataset_root = image_path.parent
    return dataset_root / "outputs"


def default_regions_path(stem: str) -> Path:
    image_output = default_regions_root() / stem
    for relative in (
        Path("text_regions/union/regions.json"),
        Path("text_regions/conditional/regions.json"),
        Path("text_regions/craft/regions.json"),
        Path("craft_text_regions/regions.json"),
    ):
        organized = image_output / relative
        if organized.is_file():
            return organized
    return legacy_regions_root() / stem / "regions.json"


def find_image_for_stem(stem: str) -> Path:
    for extension in IMAGE_EXTENSIONS:
        candidate = default_lr_dir() / f"{stem}{extension}"
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"No LR image found for stem {stem!r} under {default_lr_dir()}")


def infer_stem(image_path: Path, stem: str | None) -> str:
    return stem if stem else image_path.stem


def read_regions(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    regions = payload.get("regions", [])
    if not isinstance(regions, list):
        raise ValueError(f"regions must be a list in {path}")
    return regions


def order_quad(points: np.ndarray) -> np.ndarray:
    points = points.astype(np.float32)
    sums = points.sum(axis=1)
    diffs = np.diff(points, axis=1).reshape(-1)
    ordered = np.zeros((4, 2), dtype=np.float32)
    ordered[0] = points[np.argmin(sums)]
    ordered[2] = points[np.argmax(sums)]
    ordered[1] = points[np.argmin(diffs)]
    ordered[3] = points[np.argmax(diffs)]
    return ordered


def bbox_to_quad(bbox_xyxy: list[int]) -> np.ndarray:
    x_min, y_min, x_max, y_max = bbox_xyxy
    return np.array(
        [
            [x_min, y_min],
            [x_max, y_min],
            [x_max, y_max],
            [x_min, y_max],
        ],
        dtype=np.float32,
    )


def region_quad(region: dict[str, Any], crop_mode: str) -> np.ndarray:
    if crop_mode == "bbox":
        return bbox_to_quad([int(v) for v in region["bbox_xyxy"]])
    box = np.array(region["box"], dtype=np.float32)
    if box.shape != (4, 2):
        raise ValueError(f"Invalid region box shape: {box.shape}")
    return order_quad(box)


def quad_size(quad: np.ndarray) -> tuple[int, int, float, float]:
    top = np.linalg.norm(quad[1] - quad[0])
    bottom = np.linalg.norm(quad[2] - quad[3])
    right = np.linalg.norm(quad[2] - quad[1])
    left = np.linalg.norm(quad[3] - quad[0])
    width = max(1, int(round(max(top, bottom))))
    height = max(1, int(round(max(left, right))))
    return width, height, float(max(top, bottom)), float(max(left, right))


def warp_crop(rgb_image: np.ndarray, quad: np.ndarray, width: int, height: int) -> Image.Image:
    src = quad.astype(np.float32)
    dst = np.array(
        [
            [0, 0],
            [width - 1, 0],
            [width - 1, height - 1],
            [0, height - 1],
        ],
        dtype=np.float32,
    )
    transform = cv2.getPerspectiveTransform(src, dst)
    crop = cv2.warpPerspective(rgb_image, transform, (width, height), flags=cv2.INTER_CUBIC)
    return Image.fromarray(crop, mode="RGB")


def gray3_image(image: Image.Image) -> Image.Image:
    gray = image.convert("L")
    return Image.merge("RGB", (gray, gray, gray))


def c3_mask_to_text_mask(c3_mask: np.ndarray) -> tuple[np.ndarray, bool]:
    """Turn the C3 input mask into a foreground-text mask.

    C3's fourth channel marks pixels darker than the crop mean. For light text
    on a dark background that marks the background, so use the crop border to
    detect that polarity and invert it for paste-back only.
    """
    binary = np.where(c3_mask >= 128, 255, 0).astype(np.uint8)
    border = np.concatenate((binary[0], binary[-1], binary[:, 0], binary[:, -1]))
    inverted = float(border.mean()) > 127.5
    return (255 - binary if inverted else binary), inverted


def color_match_crop(
    sr_rgb: np.ndarray,
    canvas_crop_rgb: np.ndarray,
    text_mask: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Match C3 colors to the destination using non-text pixels as reference."""
    background = text_mask < 16
    if int(background.sum()) < 16:
        return sr_rgb, {"applied": False, "reason": "too_few_background_pixels"}

    source = sr_rgb[background].astype(np.float32)
    target = canvas_crop_rgb[background].astype(np.float32)
    source_mean = source.mean(axis=0)
    target_mean = target.mean(axis=0)
    source_std = source.std(axis=0)
    target_std = target.std(axis=0)
    gain = np.clip(target_std / np.maximum(source_std, 1.0), 0.5, 2.0)
    matched = (sr_rgb.astype(np.float32) - source_mean) * gain + target_mean
    return matched.clip(0, 255).astype(np.uint8), {
        "applied": True,
        "source_background_mean_rgb": source_mean.tolist(),
        "target_background_mean_rgb": target_mean.tolist(),
        "gain_rgb": gain.tolist(),
    }


def preserve_canvas_chroma(
    sr_rgb: np.ndarray,
    canvas_crop_rgb: np.ndarray,
    text_mask: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Use C3 luminance while keeping the bicubic crop's chroma unchanged.

    C3 predicts RGB pixels rather than a resize kernel, so its three output
    channels can shift hue. Matching the C3 luminance to the destination
    background and taking Cr/Cb directly from the bicubic crop prevents that
    hue shift without discarding the sharper luminance strokes.
    """
    sr_ycc = cv2.cvtColor(sr_rgb, cv2.COLOR_RGB2YCrCb).astype(np.float32)
    canvas_ycc = cv2.cvtColor(canvas_crop_rgb, cv2.COLOR_RGB2YCrCb).astype(np.float32)
    background = text_mask < 16

    source_y = sr_ycc[..., 0]
    target_y = canvas_ycc[..., 0]
    if int(background.sum()) >= 16:
        source_mean = float(source_y[background].mean())
        target_mean = float(target_y[background].mean())
        source_std = float(source_y[background].std())
        target_std = float(target_y[background].std())
        gain = float(np.clip(target_std / max(source_std, 1.0), 0.5, 2.0))
        matched_y = (source_y - source_mean) * gain + target_mean
        luminance_match: dict[str, Any] = {
            "applied": True,
            "source_background_mean_y": source_mean,
            "target_background_mean_y": target_mean,
            "gain_y": gain,
        }
    else:
        matched_y = source_y
        luminance_match = {"applied": False, "reason": "too_few_background_pixels"}

    output_ycc = canvas_ycc.copy()
    output_ycc[..., 0] = np.clip(matched_y, 0, 255)
    output = cv2.cvtColor(output_ycc.astype(np.uint8), cv2.COLOR_YCrCb2RGB)
    return output, {
        "applied": True,
        "chroma_source": "bicubic_canvas",
        "luminance_source": "c3",
        "luminance_match": luminance_match,
    }


def c3_gray_detail(sr_rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return C3 grayscale and its clipped signed high-frequency residual."""
    gray = cv2.cvtColor(sr_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    smooth = cv2.GaussianBlur(gray, (0, 0), DETAIL_SIGMA)
    detail = np.clip(gray - smooth, -DETAIL_LIMIT, DETAIL_LIMIT)
    return gray.astype(np.uint8), detail


def fuse_c3_detail(
    sr_rgb: np.ndarray,
    canvas_crop_rgb: np.ndarray,
    color_space: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Add only C3 high-frequency grayscale detail to destination luminance/value."""
    _, detail = c3_gray_detail(sr_rgb)
    delta = detail * DETAIL_STRENGTH
    if color_space == "ycbcr":
        converted = cv2.cvtColor(canvas_crop_rgb, cv2.COLOR_RGB2YCrCb).astype(np.float32)
        converted[..., 0] = np.clip(converted[..., 0] + delta, 0, 255)
        output = cv2.cvtColor(converted.astype(np.uint8), cv2.COLOR_YCrCb2RGB)
        preserved = "CbCr"
        modified = "Y"
    elif color_space == "hsv":
        converted = cv2.cvtColor(canvas_crop_rgb, cv2.COLOR_RGB2HSV).astype(np.float32)
        converted[..., 2] = np.clip(converted[..., 2] + delta, 0, 255)
        output = cv2.cvtColor(converted.astype(np.uint8), cv2.COLOR_HSV2RGB)
        preserved = "HS"
        modified = "V"
    else:
        raise ValueError(f"Unsupported detail color space: {color_space}")
    return output, {
        "applied": True,
        "color_space": color_space,
        "preserved_channels": preserved,
        "modified_channel": modified,
        "detail_sigma": DETAIL_SIGMA,
        "detail_limit": DETAIL_LIMIT,
        "detail_strength": DETAIL_STRENGTH,
    }


def comparison_strip(items: list[tuple[str, Image.Image]], target_height: int) -> Image.Image:
    label_height = 18
    gap = 4
    panels: list[tuple[str, Image.Image]] = []
    for label, image in items:
        rgb = image.convert("RGB")
        width = max(1, round(rgb.width * target_height / max(1, rgb.height)))
        panels.append((label, rgb.resize((width, target_height), Image.BICUBIC)))

    canvas = Image.new(
        "RGB",
        (sum(image.width for _, image in panels) + gap * (len(panels) - 1), target_height + label_height),
        "white",
    )
    draw = ImageDraw.Draw(canvas)
    x = 0
    for label, image in panels:
        draw.text((x + 3, 2), label, fill="black")
        canvas.paste(image, (x, label_height))
        x += image.width + gap
    return canvas


def paste_quad(
    canvas_rgb: np.ndarray,
    sr_crop: Image.Image,
    c3_mask: np.ndarray,
    dst_quad_lr: np.ndarray,
    scale: int,
    feather: int,
    paste_mode: str,
    craft_mask_2x: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    dst_quad = dst_quad_lr.astype(np.float32) * float(scale)
    width, height, _, _ = quad_size(dst_quad_lr)
    paste_width = max(1, int(round(width * scale)))
    paste_height = max(1, int(round(height * scale)))
    sr_resized = np.asarray(sr_crop.resize((paste_width, paste_height), Image.BICUBIC).convert("RGB"))

    mask_inverted = False
    if paste_mode == "quad":
        local_paste_mask = np.full((paste_height, paste_width), 255, dtype=np.uint8)
    else:
        text_mask, mask_inverted = c3_mask_to_text_mask(c3_mask)
        local_paste_mask = cv2.resize(text_mask, (paste_width, paste_height), interpolation=cv2.INTER_NEAREST)
        local_paste_mask = cv2.dilate(local_paste_mask, np.ones((3, 3), dtype=np.uint8), iterations=1)

    src_quad = np.array(
        [
            [0, 0],
            [paste_width - 1, 0],
            [paste_width - 1, paste_height - 1],
            [0, paste_height - 1],
        ],
        dtype=np.float32,
    )
    transform = cv2.getPerspectiveTransform(src_quad, dst_quad)
    canvas_height, canvas_width = canvas_rgb.shape[:2]
    color_match: dict[str, Any] = {"applied": False, "reason": "paste_mode_does_not_request_it"}
    sr_for_paste = sr_resized
    if paste_mode in {
        "text-mask-color-match",
        "text-mask-luma",
        "text-mask-ycbcr-detail",
        "text-mask-hsv-detail",
    }:
        inverse_transform = cv2.getPerspectiveTransform(dst_quad, src_quad)
        canvas_crop = cv2.warpPerspective(
            canvas_rgb,
            inverse_transform,
            (paste_width, paste_height),
            flags=cv2.INTER_CUBIC,
        )
        if paste_mode == "text-mask-color-match":
            sr_for_paste, color_match = color_match_crop(sr_resized, canvas_crop, local_paste_mask)
        elif paste_mode == "text-mask-luma":
            sr_for_paste, color_match = preserve_canvas_chroma(sr_resized, canvas_crop, local_paste_mask)
        else:
            color_space = "ycbcr" if paste_mode == "text-mask-ycbcr-detail" else "hsv"
            sr_for_paste, color_match = fuse_c3_detail(sr_resized, canvas_crop, color_space)

    warped = cv2.warpPerspective(
        sr_for_paste,
        transform,
        (canvas_width, canvas_height),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )

    mask = cv2.warpPerspective(local_paste_mask, transform, (canvas_width, canvas_height), flags=cv2.INTER_LINEAR)
    if paste_mode != "quad" and craft_mask_2x is not None:
        mask = np.rint(mask.astype(np.float32) * (craft_mask_2x.astype(np.float32) / 255.0)).astype(np.uint8)
    if feather and feather > 1:
        kernel = feather if feather % 2 == 1 else feather + 1
        mask = cv2.GaussianBlur(mask, (kernel, kernel), 0)
    if paste_mode != "quad" and craft_mask_2x is not None:
        mask = np.minimum(mask, craft_mask_2x)

    alpha = (mask.astype(np.float32) / 255.0)[..., None]
    blended = canvas_rgb.astype(np.float32) * (1.0 - alpha) + warped.astype(np.float32) * alpha
    debug = {
        "paste_mode": paste_mode,
        "c3_mask_inverted_for_paste": mask_inverted,
        "craft_mask_applied": paste_mode != "quad" and craft_mask_2x is not None,
        "color_match": color_match,
    }
    return blended.clip(0, 255).astype(np.uint8), mask, local_paste_mask, sr_for_paste, debug


def load_model(args: argparse.Namespace):
    import torch

    c3_root = resolve_from_repo(args.c3_root)
    tpgsr_root = resolve_from_repo(args.tpgsr_root)
    checkpoint_path = resolve_from_repo(args.checkpoint)
    recognizer_checkpoint = (
        resolve_from_repo(args.recognizer_checkpoint)
        if args.recognizer_checkpoint
        else checkpoint_path.with_name("recognizer_best_0.pth")
    )
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    c3_module = load_c3_module(c3_root, args.verbose_import)
    with pushd(c3_root):
        model = build_model(c3_module).to(device)
        checkpoint = torch.load(str(checkpoint_path), map_location=device)
        state_dict = checkpoint["state_dict_G"] if isinstance(checkpoint, dict) and "state_dict_G" in checkpoint else checkpoint
        model.load_state_dict(strip_module_prefix(state_dict), strict=True)
        model.eval()
    recognizer = None
    if args.prior_mode != "none":
        recognizer = load_recognizer(tpgsr_root, recognizer_checkpoint, device)
    return model, recognizer, recognizer_checkpoint, device


def main() -> None:
    args = parse_args()
    if args.image is None and args.stem is None:
        raise ValueError("Pass either --stem or --image.")

    image_path = resolve_from_repo(args.image) if args.image else find_image_for_stem(args.stem)
    stem = infer_stem(image_path, args.stem)
    regions_path = resolve_from_repo(args.regions) if args.regions else default_regions_path(stem)
    output_root = resolve_from_repo(args.output_root) if args.output_root else default_output_root(image_path)
    output_dir = output_root / stem / args.output_name
    debug_dir = output_dir / "debug"
    crops_dir = debug_dir / "crops"
    crops_dir.mkdir(parents=True, exist_ok=True)

    if not image_path.is_file():
        raise FileNotFoundError(f"LR image not found: {image_path}")
    if not regions_path.is_file():
        raise FileNotFoundError(f"CRAFT regions not found: {regions_path}")

    regions = read_regions(regions_path)
    model, recognizer, recognizer_checkpoint, device = load_model(args)

    import torch

    lr_image = Image.open(image_path).convert("RGB")
    lr_rgb = np.asarray(lr_image)
    base_canvas = np.asarray(
        lr_image.resize((lr_image.width * args.scale, lr_image.height * args.scale), Image.BICUBIC)
    ).copy()
    dual_detail = args.paste_mode == "detail-compare"
    canvas = base_canvas.copy()
    canvas_post_ycbcr = base_canvas.copy()
    canvas_post_hsv = base_canvas.copy()
    canvas_pre_ycbcr = base_canvas.copy()
    canvas_pre_hsv = base_canvas.copy()

    canvas_image = Image.fromarray(base_canvas, mode="RGB")
    canvas_image.save(debug_dir / "bicubic_2x.png")

    craft_mask_2x: np.ndarray | None = None
    craft_mask_path = regions_path.parent / "mask.png"
    if craft_mask_path.is_file():
        mask = Image.open(craft_mask_path).convert("L")
        mask = mask.resize((base_canvas.shape[1], base_canvas.shape[0]), Image.NEAREST)
        mask.save(debug_dir / "union_mask_2x.png")
        craft_mask_2x = np.asarray(mask, dtype=np.uint8)

    used_records: list[dict[str, Any]] = []
    processed = 0
    combined_paste_mask = np.zeros(base_canvas.shape[:2], dtype=np.uint8)

    for index, region in enumerate(regions):
        record: dict[str, Any] = {
            "index": index,
            "bbox_xyxy": region.get("bbox_xyxy"),
            "confidence": region.get("confidence"),
            "debug_text": region.get("debug_text"),
            "status": "skipped",
        }
        try:
            quad = region_quad(region, args.crop_mode)
            crop_width, crop_height, width_f, height_f = quad_size(quad)
            aspect = width_f / max(1.0, height_f)
            record.update(
                {
                    "quad": quad.tolist(),
                    "crop_width": crop_width,
                    "crop_height": crop_height,
                    "aspect": aspect,
                }
            )

            if width_f < args.min_width or height_f < args.min_height:
                record["reason"] = "too_small"
                used_records.append(record)
                continue
            if aspect < args.min_aspect:
                record["reason"] = "too_vertical_or_square"
                used_records.append(record)
                continue
            if aspect > args.max_aspect:
                record["reason"] = "too_wide"
                used_records.append(record)
                continue
            if processed >= args.max_regions:
                record["reason"] = "max_regions_reached"
                used_records.append(record)
                continue

            crop = warp_crop(lr_rgb, quad, crop_width, crop_height)
            crop_prefix = crops_dir / f"{processed:03d}_region_{index:03d}"
            post_input = crop.convert("RGB").resize((64, 16), Image.BICUBIC)
            pre_crop = gray3_image(crop)
            pre_input = pre_crop.resize((64, 16), Image.BICUBIC)
            crop.save(crop_prefix.with_name(crop_prefix.name + "_lr.png"))
            post_input.save(crop_prefix.with_name(crop_prefix.name + "_post_input_64x16.png"))
            pre_input.save(crop_prefix.with_name(crop_prefix.name + "_pre_input_64x16.png"))

            post_tensor = image_to_tensor(crop, 64, 16).to(device)
            pre_tensor = image_to_tensor(pre_crop, 64, 16).to(device)
            c3_mask = np.rint(post_tensor[0, 3].detach().cpu().numpy() * 255.0).astype(np.uint8)
            Image.fromarray(c3_mask, mode="L").save(
                crop_prefix.with_name(crop_prefix.name + "_input_mask_64x16.png")
            )
            with torch.no_grad():
                post_sr_tensor, post_clues = infer_with_priors(
                    model, post_tensor, args.prior_mode, recognizer
                )
                pre_sr_tensor, pre_clues = infer_with_priors(
                    model, pre_tensor, args.prior_mode, recognizer
                )

            post_sr_crop = tensor_to_image(post_sr_tensor[0, :3, :, :], args.save_mode)
            pre_sr_crop = tensor_to_image(pre_sr_tensor[0, :3, :, :], args.save_mode)
            post_sr_crop.save(crop_prefix.with_name(crop_prefix.name + "_post_c3_128x32.png"))
            pre_sr_crop.save(crop_prefix.with_name(crop_prefix.name + "_pre_c3_128x32.png"))

            for route, sr_crop in (("post", post_sr_crop), ("pre", pre_sr_crop)):
                c3_gray, signed_detail = c3_gray_detail(np.asarray(sr_crop.convert("RGB")))
                Image.fromarray(c3_gray, mode="L").save(
                    crop_prefix.with_name(crop_prefix.name + f"_{route}_gray.png")
                )
                np.save(
                    crop_prefix.with_name(crop_prefix.name + f"_{route}_detail.npy"),
                    signed_detail,
                )
                detail_visual = np.clip(signed_detail + 128.0, 0, 255).astype(np.uint8)
                Image.fromarray(detail_visual, mode="L").save(
                    crop_prefix.with_name(crop_prefix.name + f"_{route}_detail_visual.png")
                )

            clue_summaries = {"post": clue_debug(post_clues), "pre": clue_debug(pre_clues)}
            for route, clues in (("post", post_clues), ("pre", pre_clues)):
                for clue_name, clue_tensor in clues.items():
                    np.save(
                        crop_prefix.with_name(
                            crop_prefix.name + f"_{route}_{clue_name}_probabilities.npy"
                        ),
                        clue_tensor.detach().cpu().numpy(),
                    )
            crop_prefix.with_name(crop_prefix.name + "_clues.json").write_text(
                json.dumps(clue_summaries, ensure_ascii=False, indent=2), encoding="utf-8"
            )

            if dual_detail:
                canvas_post_ycbcr, paste_mask, local_paste_mask, post_y_local, post_y_debug = paste_quad(
                    canvas_post_ycbcr,
                    post_sr_crop,
                    c3_mask,
                    quad,
                    args.scale,
                    args.feather,
                    "text-mask-ycbcr-detail",
                    craft_mask_2x,
                )
                canvas_post_hsv, _, _, post_v_local, post_v_debug = paste_quad(
                    canvas_post_hsv,
                    post_sr_crop,
                    c3_mask,
                    quad,
                    args.scale,
                    args.feather,
                    "text-mask-hsv-detail",
                    craft_mask_2x,
                )
                canvas_pre_ycbcr, _, _, pre_y_local, pre_y_debug = paste_quad(
                    canvas_pre_ycbcr,
                    pre_sr_crop,
                    c3_mask,
                    quad,
                    args.scale,
                    args.feather,
                    "text-mask-ycbcr-detail",
                    craft_mask_2x,
                )
                canvas_pre_hsv, _, _, pre_v_local, pre_v_debug = paste_quad(
                    canvas_pre_hsv,
                    pre_sr_crop,
                    c3_mask,
                    quad,
                    args.scale,
                    args.feather,
                    "text-mask-hsv-detail",
                    craft_mask_2x,
                )
                for name, image in (
                    ("post_y", post_y_local),
                    ("post_v", post_v_local),
                    ("pre_y", pre_y_local),
                    ("pre_v", pre_v_local),
                ):
                    Image.fromarray(image, mode="RGB").save(
                        crop_prefix.with_name(crop_prefix.name + f"_{name}.png")
                    )
                paste_debug = {
                    "post_y": post_y_debug,
                    "post_v": post_v_debug,
                    "pre_y": pre_y_debug,
                    "pre_v": pre_v_debug,
                }
                sr_for_paste = post_y_local
            else:
                canvas, paste_mask, local_paste_mask, sr_for_paste, paste_debug = paste_quad(
                    canvas,
                    post_sr_crop,
                    c3_mask,
                    quad,
                    args.scale,
                    args.feather,
                    args.paste_mode,
                    craft_mask_2x,
                )
            combined_paste_mask = np.maximum(combined_paste_mask, paste_mask)
            Image.fromarray(local_paste_mask, mode="L").save(
                crop_prefix.with_name(crop_prefix.name + "_paste_mask.png")
            )
            display_items = [
                ("LR crop", crop),
                ("Post", post_sr_crop),
                ("Pre", pre_sr_crop),
            ]
            if dual_detail:
                display_items.extend(
                    [
                        ("Post-Y", Image.fromarray(post_y_local, mode="RGB")),
                        ("Post-V", Image.fromarray(post_v_local, mode="RGB")),
                        ("Pre-Y", Image.fromarray(pre_y_local, mode="RGB")),
                        ("Pre-V", Image.fromarray(pre_v_local, mode="RGB")),
                    ]
                )
            elif args.paste_mode in {"text-mask-color-match", "text-mask-luma"}:
                matched = Image.fromarray(sr_for_paste, mode="RGB")
                suffix = "_color_matched.png" if args.paste_mode == "text-mask-color-match" else "_luma_chroma_preserved.png"
                label = "Color matched" if args.paste_mode == "text-mask-color-match" else "C3 Y + bicubic chroma"
                matched.save(crop_prefix.with_name(crop_prefix.name + suffix))
                display_items.append((label, matched))
            comparison_strip(display_items, 96).save(
                crop_prefix.with_name(crop_prefix.name + "_comparison.png")
            )

            record.update(
                {
                    "status": "used",
                    "post_sr_tensor_shape": list(post_sr_tensor.shape),
                    "post_sr_tensor_min": float(post_sr_tensor.min()),
                    "post_sr_tensor_max": float(post_sr_tensor.max()),
                    "pre_sr_tensor_shape": list(pre_sr_tensor.shape),
                    "pre_sr_tensor_min": float(pre_sr_tensor.min()),
                    "pre_sr_tensor_max": float(pre_sr_tensor.max()),
                    "prior_mode": args.prior_mode,
                    "clues": clue_summaries,
                    **paste_debug,
                }
            )
            processed += 1
        except Exception as exc:  # Keep one bad region from killing the whole visual debug run.
            record["reason"] = f"error: {type(exc).__name__}: {exc}"
        used_records.append(record)

    paste_mask_path = debug_dir / "paste_mask_2x.png"
    records_path = output_dir / "run.json"
    Image.fromarray(combined_paste_mask, mode="L").save(paste_mask_path)
    if dual_detail:
        post_y_path = output_dir / "result_post_y.png"
        post_v_path = output_dir / "result_post_v.png"
        pre_y_path = output_dir / "result_pre_y.png"
        pre_v_path = output_dir / "result_pre_v.png"
        Image.fromarray(canvas_post_ycbcr, mode="RGB").save(post_y_path)
        Image.fromarray(canvas_post_hsv, mode="RGB").save(post_v_path)
        Image.fromarray(canvas_pre_ycbcr, mode="RGB").save(pre_y_path)
        Image.fromarray(canvas_pre_hsv, mode="RGB").save(pre_v_path)
        comparison_strip(
            [
                ("Bicubic", canvas_image),
                ("Post-Y", Image.fromarray(canvas_post_ycbcr, mode="RGB")),
                ("Post-V", Image.fromarray(canvas_post_hsv, mode="RGB")),
                ("Pre-Y", Image.fromarray(canvas_pre_ycbcr, mode="RGB")),
                ("Pre-V", Image.fromarray(canvas_pre_hsv, mode="RGB")),
            ],
            base_canvas.shape[0],
        ).save(output_dir / "comparison.png")
        results = {
            "post_y": str(post_y_path),
            "post_v": str(post_v_path),
            "pre_y": str(pre_y_path),
            "pre_v": str(pre_v_path),
        }
    else:
        result_path = output_dir / "result.png"
        Image.fromarray(canvas, mode="RGB").save(result_path)
        comparison_strip(
            [("Bicubic 2x", canvas_image), ("C3 text fusion", Image.fromarray(canvas, mode="RGB"))],
            base_canvas.shape[0],
        ).save(output_dir / "comparison.png")
        results = {"result": str(result_path)}
    run_record = {
        "image": str(image_path),
        "regions": str(regions_path),
        "checkpoint": str(resolve_from_repo(args.checkpoint)),
        "recognizer_checkpoint": str(recognizer_checkpoint) if recognizer else None,
        "prior_mode": args.prior_mode,
        "save_mode": args.save_mode,
        "crop_mode": args.crop_mode,
        "paste_mode": args.paste_mode,
        "feather": args.feather,
        "scale": args.scale,
        "processed_region_count": processed,
        "results": results,
        "detail": {
            "sigma": DETAIL_SIGMA,
            "limit": DETAIL_LIMIT,
            "strength": DETAIL_STRENGTH,
        },
        "paste_mask": str(paste_mask_path),
        "records": used_records,
    }
    record_json = json.dumps(run_record, ensure_ascii=False, indent=2)
    records_path.write_text(record_json, encoding="utf-8")

    print(f"image: {image_path}")
    print(f"regions: {regions_path}")
    print(f"output_dir: {output_dir}")
    print(f"processed regions: {processed} / {len(regions)}")
    print(f"paste mode: {args.paste_mode}")
    print(f"results: {results}")


if __name__ == "__main__":
    main()
