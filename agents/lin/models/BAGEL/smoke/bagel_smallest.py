"""Run a bounded BAGEL image-editing probe through the official public Space.

This script does not import or run ``statemachine.py``. Each selected LR image
is uploaded once to ByteDance-Seed's public Hugging Face Space, submitted to
the named ``/edit_image`` Gradio endpoint, and saved with trace metadata for
manual comparison against the LR input and HR reference.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
from datetime import datetime
from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory

import gradio_client
from dotenv import load_dotenv
from gradio_client import Client as GradioClient, handle_file
from PIL import Image, ImageDraw, ImageOps

REPO_ROOT = Path(__file__).resolve().parents[5]
DEFAULT_INPUT_DIR = REPO_ROOT / "workspace" / "SR" / "COLLECT" / "selected_LR"
DEFAULT_OUTPUT_ROOT = DEFAULT_INPUT_DIR / "bagel"
DEFAULT_NAMES = ("IMG_001.png", "IMG_040_256.png")
DEFAULT_API_BASE = "https://bytedance-seed-bagel.hf.space"
DEFAULT_SPACE_ID = "ByteDance-Seed/BAGEL"
EXPERIMENT_DIR = "text_repair_minimal_validation"
DEFAULT_PROMPT = """This is a low-level image restoration and 4x super-resolution task, not a creative image-editing task.
First understand the complete scene and all visible text by using both the degraded glyph shapes and their surrounding visual context. Then produce the highest-quality faithful high-resolution reconstruction: remove blur, noise, compression artifacts, and aliasing, and restore natural edges, textures, and fine details.
Pay special attention to every character, word, number, logo, and sign. Infer the most likely intended text when degradation makes a glyph ambiguous, then repair and reconstruct it so that it is sharp, legible, semantically coherent, and consistent with the scene. Preserve the original language, wording, text position, layout, font style, colors, and logo identity as faithfully as possible. Do not translate the text, replace it with unrelated wording, or add text that is not part of the original scene.
Preserve the original scene, geometry, composition, colors, objects, and identity. Do not add or remove objects and do not redesign the image. Return only the restored 4x high-resolution image."""


def parse_args() -> argparse.Namespace:
    """Parse command-line options for the bounded two-image probe."""
    parser = argparse.ArgumentParser(
        description="Single-pass BAGEL restoration probe via the official Space."
    )
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--names", nargs="+", default=list(DEFAULT_NAMES))
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--api-base", default=DEFAULT_API_BASE)
    parser.add_argument(
        "--hf-token-env",
        default="HF_TOKEN",
        help="Environment variable containing the Hugging Face access token.",
    )
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument(
        "--seed",
        type=int,
        default=1,
        help="Positive seed for reproducibility; the BAGEL UI labels 0 as random.",
    )
    parser.add_argument("--cfg-text-scale", type=float, default=4.0)
    parser.add_argument("--cfg-image-scale", type=float, default=2.0)
    parser.add_argument("--cfg-interval", type=float, default=0.0)
    parser.add_argument("--timestep-shift", type=float, default=3.0)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--cfg-renorm-min", type=float, default=0.0)
    parser.add_argument(
        "--cfg-renorm-type",
        choices=("global", "local", "text_channel"),
        default="text_channel",
    )
    parser.add_argument("--request-timeout-seconds", type=float, default=120.0)
    parser.add_argument(
        "--generation-timeout-seconds", type=float, default=900.0
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    """Return a stable SHA-256 digest for one file."""
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def image_size(path: Path) -> list[int]:
    """Return image width and height."""
    with Image.open(path) as image:
        return [image.width, image.height]


def extract_output_path(result: object) -> tuple[Path, list[str]]:
    """Return the downloaded image path from a Gradio client result."""
    values = list(result) if isinstance(result, (list, tuple)) else [result]
    if not values:
        raise RuntimeError("BAGEL returned no output values")

    first = values[0]
    candidate: object
    if isinstance(first, (str, Path)):
        candidate = first
    elif isinstance(first, dict):
        candidate = first.get("path")
    else:
        candidate = None
    if not isinstance(candidate, (str, Path)):
        raise RuntimeError(f"BAGEL returned no downloadable image: {result!r}")

    output_path = Path(candidate)
    if not output_path.is_file():
        raise RuntimeError(
            f"BAGEL client output does not exist locally: {output_path}"
        )
    return output_path, [str(value) for value in values]


def find_hr(input_path: Path) -> Path | None:
    """Find the HR reference matching a selected LR filename."""
    stem = input_path.stem
    for suffix in ("_128", "_256", "_512"):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
            break
    hr_dir = input_path.parent.parent / "HR"
    for extension in (".png", ".jpg", ".jpeg"):
        candidate = hr_dir / f"{stem}{extension}"
        if candidate.exists():
            return candidate.resolve()
    return None


def fit_panel(image: Image.Image, size: tuple[int, int]) -> Image.Image:
    """Fit an image into a white comparison panel without cropping."""
    return ImageOps.pad(
        image.convert("RGB"),
        size,
        method=Image.Resampling.LANCZOS,
        color="white",
    )


def save_comparison(
    input_path: Path,
    output_path: Path,
    hr_path: Path | None,
    comparison_path: Path,
) -> None:
    """Save an LR/BAGEL/HR side-by-side comparison."""
    panel_size = (640, 640)
    sources = [("LR input", input_path), ("BAGEL output", output_path)]
    if hr_path is not None:
        sources.append(("HR reference", hr_path))
    label_height = 36
    sheet = Image.new(
        "RGB", (panel_size[0] * len(sources), panel_size[1] + label_height), "white"
    )
    draw = ImageDraw.Draw(sheet)
    for index, (label, path) in enumerate(sources):
        with Image.open(path) as image:
            panel = fit_panel(image, panel_size)
        x = index * panel_size[0]
        sheet.paste(panel, (x, label_height))
        draw.text((x + 10, 10), label, fill="black")
    sheet.save(comparison_path)


def write_review_csv(
    input_path: Path,
    output_path: Path,
    hr_path: Path | None,
    review_path: Path,
) -> None:
    """Create one per-image manual text-review worksheet."""
    fieldnames = [
        "input_name",
        "input_path",
        "output_path",
        "hr_path",
        "lr_human_text",
        "hr_ground_truth_text",
        "output_human_text",
        "semantic_status",
        "notes",
    ]
    row = {
        "input_name": input_path.name,
        "input_path": str(input_path),
        "output_path": str(output_path),
        "hr_path": str(hr_path) if hr_path else "",
        "lr_human_text": "",
        "hr_ground_truth_text": "",
        "output_human_text": "",
        "semantic_status": "",
        "notes": "",
    }
    with review_path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(row)


def main() -> None:
    """Run exactly one official BAGEL Space call per selected input."""
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    args = parse_args()
    if not 0 <= args.seed <= 1_000_000:
        raise ValueError("--seed must be between 0 and 1000000")
    if not 10 <= args.steps <= 100:
        raise ValueError("--steps must be between 10 and 100")

    input_dir = args.input_dir.resolve()
    input_paths = [(input_dir / name).resolve() for name in args.names]
    missing = [path for path in input_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Missing input images:\n" + "\n".join(str(path) for path in missing)
        )

    print("[plan] model=ByteDance-Seed/BAGEL-7B-MoT")
    print(f"[plan] space={args.api_base}")
    print(
        f"[plan] endpoint=/edit_image, "
        f"programmatic_zero_gpu_token={args.hf_token_env}"
    )
    print(f"[plan] experiment={EXPERIMENT_DIR}")
    print(f"[plan] one pass per image, requests={len(input_paths)}")
    print(
        f"[plan] seed={args.seed}, steps={args.steps}, "
        f"cfg_text={args.cfg_text_scale}, cfg_image={args.cfg_image_scale}"
    )
    for path in input_paths:
        print(f"[plan] {path.name} input={image_size(path)}")
    if args.dry_run:
        print("[dry-run] no upload or generation request made")
        return

    load_dotenv(REPO_ROOT / ".env")
    hf_token = os.getenv(args.hf_token_env, "").strip()
    if not hf_token:
        raise ValueError(
            f"{args.hf_token_env} is missing. Programmatic ZeroGPU calls need "
            "a Hugging Face token to associate the request with your free GPU "
            "quota. The official web Space can instead be used after signing in."
        )

    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    failures: list[str] = []
    with TemporaryDirectory(prefix="bagel_gradio_") as download_directory:
        client = GradioClient(
            DEFAULT_SPACE_ID,
            token=hf_token,
            verbose=True,
            download_files=download_directory,
            httpx_kwargs={"timeout": args.request_timeout_seconds},
        )
        for index, input_path in enumerate(input_paths, start=1):
            item_dir = (
                output_root
                / input_path.stem
                / EXPERIMENT_DIR
                / f"seed_{args.seed}"
            )
            item_dir.mkdir(parents=True, exist_ok=True)
            output_path = item_dir / "output.png"
            if output_path.exists():
                raise FileExistsError(
                    f"Refusing to overwrite existing result: {output_path}"
                )

            print(f"[{index}/{len(input_paths)}] submit: {input_path.name}")
            try:
                job = client.submit(
                    handle_file(input_path),
                    args.prompt,
                    False,
                    args.cfg_text_scale,
                    args.cfg_image_scale,
                    args.cfg_interval,
                    args.timestep_shift,
                    args.steps,
                    args.cfg_renorm_min,
                    args.cfg_renorm_type,
                    1024,
                    False,
                    0.3,
                    args.seed,
                    api_name="/edit_image",
                )
                result = job.result(timeout=args.generation_timeout_seconds)
            except Exception as error:
                failure = {
                    "created_at": datetime.now().astimezone().isoformat(),
                    "input_name": input_path.name,
                    "input_path": str(input_path),
                    "input_sha256": sha256_file(input_path),
                    "model": "ByteDance-Seed/BAGEL-7B-MoT",
                    "space_id": DEFAULT_SPACE_ID,
                    "endpoint": "/edit_image",
                    "gradio_client_version": gradio_client.__version__,
                    "error_type": type(error).__name__,
                    "error_message": str(error),
                    "prompt": args.prompt,
                    "config": {
                        "seed": args.seed,
                        "steps": args.steps,
                        "cfg_text_scale": args.cfg_text_scale,
                        "cfg_image_scale": args.cfg_image_scale,
                        "cfg_interval": args.cfg_interval,
                        "timestep_shift": args.timestep_shift,
                        "cfg_renorm_min": args.cfg_renorm_min,
                        "cfg_renorm_type": args.cfg_renorm_type,
                    },
                }
                with (item_dir / "attempts.jsonl").open(
                    "a", encoding="utf-8"
                ) as file:
                    file.write(json.dumps(failure, ensure_ascii=False) + "\n")
                failures.append(f"{input_path.name}: {error}")
                print(
                    f"[{index}/{len(input_paths)}] failed and recorded: "
                    f"{type(error).__name__}: {error}"
                )
                continue

            provider_output_path, provider_result = extract_output_path(result)
            output_bytes = provider_output_path.read_bytes()
            raw_suffix = provider_output_path.suffix or ".bin"
            raw_path = item_dir / f"output_api_original{raw_suffix}"
            raw_path.write_bytes(output_bytes)
            with Image.open(BytesIO(output_bytes)) as image:
                image.save(output_path, format="PNG")

            hr_path = find_hr(input_path)
            comparison_path = item_dir / "comparison.png"
            save_comparison(input_path, output_path, hr_path, comparison_path)
            review_path = item_dir / "human_text_review.csv"
            write_review_csv(
                input_path, output_path.resolve(), hr_path, review_path
            )
            metadata = {
                "experiment": "bagel_text_repair_minimal_validation",
                "input_name": input_path.name,
                "input_path": str(input_path),
                "input_sha256": sha256_file(input_path),
                "input_size": image_size(input_path),
                "output_path": str(output_path.resolve()),
                "output_sha256": sha256_file(output_path),
                "output_size": image_size(output_path),
                "comparison_path": str(comparison_path.resolve()),
                "hr_path": str(hr_path) if hr_path else None,
                "model": "ByteDance-Seed/BAGEL-7B-MoT",
                "space_id": DEFAULT_SPACE_ID,
                "space_url": args.api_base,
                "endpoint": "/edit_image",
                "gradio_client_version": gradio_client.__version__,
                "job_status": str(job.status()),
                "prompt": args.prompt,
                "config": {
                    "seed": args.seed,
                    "steps": args.steps,
                    "cfg_text_scale": args.cfg_text_scale,
                    "cfg_image_scale": args.cfg_image_scale,
                    "cfg_interval": args.cfg_interval,
                    "timestep_shift": args.timestep_shift,
                    "cfg_renorm_min": args.cfg_renorm_min,
                    "cfg_renorm_type": args.cfg_renorm_type,
                    "show_thinking": False,
                    "do_sample": False,
                    "text_temperature": 0.3,
                },
                "provider_result": provider_result,
                "provider_download_name": provider_output_path.name,
                "created_at": datetime.now().astimezone().isoformat(),
            }
            (item_dir / "metadata.json").write_text(
                json.dumps(metadata, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            print(f"[{index}/{len(input_paths)}] saved: {output_path}")

    if failures:
        raise RuntimeError(
            "BAGEL generation failed for:\n" + "\n".join(failures)
        )


if __name__ == "__main__":
    main()
