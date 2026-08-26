"""Run a fixed, single-pass Qwen-Image-Edit-2511 restoration probe.

This script intentionally does not import or run ``statemachine.py``.  Each
selected LR image is submitted to ModelScope API-Inference exactly once with
the same prompt. The raw model output and trace metadata are saved for later
human text review.
"""

from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import json
import mimetypes
import os
import subprocess
import sys
import time
from datetime import datetime
from importlib.metadata import PackageNotFoundError, version
from io import BytesIO
from pathlib import Path

import httpx
from dotenv import load_dotenv
from PIL import Image, ImageDraw, ImageOps

REPO_ROOT = Path(__file__).resolve().parents[5]
DEFAULT_INPUT_DIR = REPO_ROOT / "workspace" / "SR" / "COLLECT" / "selected_LR"
DEFAULT_OUTPUT_ROOT = DEFAULT_INPUT_DIR / "qwen"
DEFAULT_NAMES = (
    "IMG_040_128.png",
    "IMG_040_256.png",
    "IMG_044_128.png",
    "IMG_044_256.png",
)
DEFAULT_MODEL = "Qwen/Qwen-Image-Edit-2511"
DEFAULT_API_BASE = "https://api-inference.modelscope.cn"
EXPERIMENT_DIR = "text_repair_minimal_validation"
DEFAULT_PROMPT = """This is a low-level image restoration and 4x super-resolution task, not a creative image-editing task.
First understand the complete scene and all visible text by using both the degraded glyph shapes and their surrounding visual context. Then produce the highest-quality faithful high-resolution reconstruction: remove blur, noise, compression artifacts, and aliasing, and restore natural edges, textures, and fine details.
Pay special attention to every character, word, number, logo, and sign. Infer the most likely intended text when degradation makes a glyph ambiguous, then repair and reconstruct it so that it is sharp, legible, semantically coherent, and consistent with the scene. Preserve the original language, wording, text position, layout, font style, colors, and logo identity as faithfully as possible. Do not translate the text, replace it with unrelated wording, or add text that is not part of the original scene.
Preserve the original scene, geometry, composition, colors, objects, and identity. Do not add or remove objects and do not redesign the image. Return only the restored 4x high-resolution image."""


def parse_args() -> argparse.Namespace:
    """Parse command-line options for the bounded probe."""
    parser = argparse.ArgumentParser(
        description="Single-pass Qwen-Image-Edit-2511 restoration probe."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help="Directory containing LR images.",
    )
    parser.add_argument(
        "--names",
        nargs="+",
        default=list(DEFAULT_NAMES),
        help="Image filenames to process. Defaults to two 128/256 pairs.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Parent directory for per-image experiment outputs.",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--api-base", default=DEFAULT_API_BASE)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument(
        "--scale",
        type=int,
        default=4,
        help="Requested SR scale before applying ModelScope's maximum side limit.",
    )
    parser.add_argument(
        "--max-side",
        type=int,
        default=1664,
        help="Maximum requested output side for Qwen-Image API-Inference.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--guidance", type=float, default=4.0)
    parser.add_argument(
        "--request-timeout-seconds",
        type=float,
        default=60.0,
        help="HTTP timeout for each submit, poll, or download request.",
    )
    parser.add_argument(
        "--poll-timeout-seconds",
        type=float,
        default=600.0,
        help="Maximum total wait for one submitted generation task.",
    )
    parser.add_argument(
        "--poll-interval-seconds",
        type=float,
        default=5.0,
        help="Delay between task-status checks.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate inputs and print the plan without calling the API.",
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    """Return a stable SHA-256 digest for one file."""
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def image_size(path: Path) -> list[int]:
    """Read image width and height without retaining the image object."""
    with Image.open(path) as image:
        return [image.width, image.height]


def requested_output_size(path: Path, scale: int, max_side: int) -> str:
    """Build a bounded, approximately aspect-preserving Qwen size string."""
    width, height = image_size(path)
    target_width = width * scale
    target_height = height * scale
    shrink = min(1.0, max_side / max(target_width, target_height))
    target_width = max(64, round(target_width * shrink / 64) * 64)
    target_height = max(64, round(target_height * shrink / 64) * 64)
    target_width = min(target_width, max_side)
    target_height = min(target_height, max_side)
    return f"{target_width}x{target_height}"


def image_to_data_url(path: Path) -> str:
    """Encode a local image using ModelScope's documented data-URL format."""
    mime_type, _ = mimetypes.guess_type(path)
    if mime_type is None or not mime_type.startswith("image/"):
        mime_type = "image/png"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def require_success(response: httpx.Response, operation: str) -> None:
    """Raise an actionable API error while keeping credentials out of logs."""
    try:
        response.raise_for_status()
    except httpx.HTTPStatusError as error:
        body = response.text[:2000]
        raise RuntimeError(
            f"{operation} failed with HTTP {response.status_code}: {body}"
        ) from error


def submit_generation(
    client: httpx.Client,
    api_base: str,
    model: str,
    prompt: str,
    input_path: Path,
    size: str,
    seed: int,
    steps: int,
    guidance: float,
) -> tuple[str, dict[str, object], dict[str, str]]:
    """Submit exactly one asynchronous Qwen image-editing task."""
    payload: dict[str, object] = {
        "model": model,
        "prompt": prompt,
        "image_url": [image_to_data_url(input_path)],
        "size": size,
        "seed": seed,
        "steps": steps,
        "guidance": guidance,
    }
    response = client.post(
        f"{api_base.rstrip('/')}/v1/images/generations",
        headers={"X-ModelScope-Async-Mode": "true"},
        json=payload,
    )
    require_success(response, "ModelScope task submission")
    response_data = response.json()
    task_id = response_data.get("task_id")
    if not isinstance(task_id, str) or not task_id:
        raise RuntimeError(f"ModelScope response has no task_id: {response_data}")

    rate_limits = {
        key: value
        for key, value in response.headers.items()
        if key.lower().startswith("modelscope-ratelimit-")
    }
    return task_id, payload, rate_limits


def poll_generation(
    client: httpx.Client,
    api_base: str,
    task_id: str,
    timeout_seconds: float,
    interval_seconds: float,
) -> dict[str, object]:
    """Poll one submitted task without creating additional generations."""
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        response = client.get(
            f"{api_base.rstrip('/')}/v1/tasks/{task_id}",
            headers={"X-ModelScope-Task-Type": "image_generation"},
        )
        require_success(response, "ModelScope task polling")
        data = response.json()
        status = data.get("task_status")
        if status == "SUCCEED":
            return data
        if status == "FAILED":
            raise RuntimeError(f"ModelScope task failed: {data}")
        time.sleep(interval_seconds)
    raise TimeoutError(f"ModelScope task {task_id} exceeded {timeout_seconds} seconds")


def download_output(url: str, timeout_seconds: float) -> tuple[bytes, str]:
    """Download a result without forwarding the ModelScope bearer token."""
    response = httpx.get(url, timeout=timeout_seconds, follow_redirects=True)
    require_success(response, "ModelScope output download")
    return response.content, response.headers.get("content-type", "")


def save_api_output(
    data: bytes, content_type: str, item_dir: Path
) -> tuple[Path, Path]:
    """Preserve the provider bytes and save a normalized PNG for review."""
    mime_type = content_type.partition(";")[0].strip()
    suffix = mimetypes.guess_extension(mime_type) or ".bin"
    raw_path = item_dir / f"output_api_original{suffix}"
    raw_path.write_bytes(data)

    output_path = item_dir / "output.png"
    with Image.open(BytesIO(data)) as image:
        image.save(output_path, format="PNG")
    return raw_path, output_path


def find_hr(input_path: Path) -> Path | None:
    """Find the HR reference matching a selected LR filename, if present."""
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


def git_head() -> str | None:
    """Return the repository commit used for this run."""
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() or None


def package_version(name: str) -> str | None:
    """Return an installed package version when available."""
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def fit_panel(image: Image.Image, size: tuple[int, int]) -> Image.Image:
    """Fit an image into a white comparison panel without cropping it."""
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
    """Save an LR/output/HR side-by-side sheet for visual review."""
    panel_size = (640, 640)
    labeled_panels: list[tuple[str, Image.Image]] = []
    sources = [("LR input", input_path), ("Qwen 2511 output", output_path)]
    if hr_path is not None:
        sources.append(("HR reference", hr_path))

    for label, path in sources:
        with Image.open(path) as image:
            labeled_panels.append((label, fit_panel(image, panel_size)))

    label_height = 36
    sheet = Image.new(
        "RGB",
        (panel_size[0] * len(labeled_panels), panel_size[1] + label_height),
        "white",
    )
    draw = ImageDraw.Draw(sheet)
    for index, (label, panel) in enumerate(labeled_panels):
        x = index * panel_size[0]
        sheet.paste(panel, (x, label_height))
        draw.text((x + 10, 10), label, fill="black")
    comparison_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(comparison_path)


def write_review_csv(rows: list[dict[str, str]], path: Path) -> None:
    """Create the manual transcription and semantic-status worksheet."""
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
    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    """Run the fixed single-pass API probe and save traceable artifacts."""
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    args = parse_args()
    if args.scale < 1:
        raise ValueError("--scale must be at least 1")
    if not 64 <= args.max_side <= 1664:
        raise ValueError("--max-side must be between 64 and 1664")
    if not 0 <= args.seed < 2**31:
        raise ValueError("--seed must be in [0, 2^31)")
    if not 1 <= args.steps <= 100:
        raise ValueError("--steps must be between 1 and 100")
    if not 1.5 <= args.guidance <= 20.0:
        raise ValueError("--guidance must be between 1.5 and 20")
    if args.request_timeout_seconds <= 0 or args.poll_timeout_seconds <= 0:
        raise ValueError("timeout values must be positive")
    if args.poll_interval_seconds <= 0:
        raise ValueError("--poll-interval-seconds must be positive")

    input_dir = args.input_dir.resolve()
    input_paths = [(input_dir / name).resolve() for name in args.names]
    missing = [path for path in input_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Missing input images:\n" + "\n".join(str(path) for path in missing)
        )

    print(f"[plan] model={args.model}")
    print(f"[plan] api_base={args.api_base}")
    print(f"[plan] experiment={EXPERIMENT_DIR}")
    print(f"[plan] output_root={args.output_root.resolve()}")
    print(f"[plan] one pass per image, requests={len(input_paths)}")
    print(
        f"[plan] scale={args.scale}, max_side={args.max_side}, "
        f"seed={args.seed}, steps={args.steps}, guidance={args.guidance}"
    )
    for path in input_paths:
        size = requested_output_size(path, args.scale, args.max_side)
        print(f"[plan] {path.name} input={image_size(path)} requested={size}")
    if args.dry_run:
        print("[dry-run] no API request made")
        return

    load_dotenv(REPO_ROOT / ".env")
    api_token = os.getenv("MODELSCOPE_ACCESS_TOKEN", "").strip()
    if not api_token:
        raise ValueError(
            "MODELSCOPE_ACCESS_TOKEN is missing. Add it to the repository .env file."
        )

    run_dir = args.output_root.resolve()
    run_dir.mkdir(parents=True, exist_ok=True)

    client = httpx.Client(
        headers={
            "Authorization": f"Bearer {api_token}",
            "Content-Type": "application/json",
        },
        timeout=args.request_timeout_seconds,
    )

    records: list[dict[str, object]] = []
    try:
        for index, input_path in enumerate(input_paths, start=1):
            requested_size = requested_output_size(
                input_path, args.scale, args.max_side
            )
            print(
                f"[{index}/{len(input_paths)}] submit: "
                f"{input_path.name} -> {requested_size}"
            )
            item_dir = (
                run_dir
                / input_path.stem
                / EXPERIMENT_DIR
                / f"seed_{args.seed}"
            )
            item_dir.mkdir(parents=True, exist_ok=True)
            existing_output = item_dir / "output.png"
            if existing_output.exists():
                raise FileExistsError(
                    f"Refusing to overwrite existing result: {existing_output}"
                )

            task_id, _, rate_limits = submit_generation(
                client=client,
                api_base=args.api_base,
                model=args.model,
                prompt=args.prompt,
                input_path=input_path,
                size=requested_size,
                seed=args.seed,
                steps=args.steps,
                guidance=args.guidance,
            )
            print(f"[{index}/{len(input_paths)}] task_id={task_id}")
            task_result = poll_generation(
                client=client,
                api_base=args.api_base,
                task_id=task_id,
                timeout_seconds=args.poll_timeout_seconds,
                interval_seconds=args.poll_interval_seconds,
            )
            output_urls = task_result.get("output_images")
            if not isinstance(output_urls, list) or not output_urls:
                raise RuntimeError(
                    f"ModelScope task returned no output_images: {task_result}"
                )
            output_url = output_urls[0]
            if not isinstance(output_url, str):
                raise RuntimeError(f"Invalid output image URL: {output_url!r}")

            output_bytes, output_content_type = download_output(
                output_url, args.request_timeout_seconds
            )
            raw_output_path, output_path = save_api_output(
                output_bytes, output_content_type, item_dir
            )
            hr_path = find_hr(input_path)
            comparison_path = item_dir / "comparison.png"
            save_comparison(input_path, output_path, hr_path, comparison_path)

            record: dict[str, object] = {
                "input_name": input_path.name,
                "input_path": str(input_path),
                "input_sha256": sha256_file(input_path),
                "input_size": image_size(input_path),
                "raw_output_path": str(raw_output_path.resolve()),
                "raw_output_sha256": sha256_file(raw_output_path),
                "output_content_type": output_content_type,
                "output_path": str(output_path.resolve()),
                "output_sha256": sha256_file(output_path),
                "output_size": image_size(output_path),
                "comparison_path": str(comparison_path.resolve()),
                "hr_path": str(hr_path) if hr_path else None,
                "hr_sha256": sha256_file(hr_path) if hr_path else None,
                "hr_size": image_size(hr_path) if hr_path else None,
                "model": args.model,
                "api_base": args.api_base,
                "task_id": task_id,
                "task_result": task_result,
                "rate_limits": rate_limits,
                "prompt": args.prompt,
                "seed": args.seed,
                "steps": args.steps,
                "guidance": args.guidance,
                "requested_size": requested_size,
                "created_at": datetime.now().astimezone().isoformat(),
            }
            (item_dir / "metadata.json").write_text(
                json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            review_path = item_dir / "human_text_review.csv"
            write_review_csv(
                [
                    {
                        "input_name": input_path.name,
                        "input_path": str(input_path),
                        "output_path": str(output_path.resolve()),
                        "hr_path": str(hr_path) if hr_path else "",
                        "lr_human_text": "",
                        "hr_ground_truth_text": "",
                        "output_human_text": "",
                        "semantic_status": "",
                        "notes": "",
                    }
                ],
                review_path,
            )
            records.append(record)
            print(f"[{index}/{len(input_paths)}] saved: {output_path}")
            print(f"[{index}/{len(input_paths)}] review: {review_path}")
    finally:
        client.close()

    run_manifest = {
        "experiment": "qwen2511_text_repair_minimal_validation",
        "experiment_dir": EXPERIMENT_DIR,
        "definition": (
            "A general-purpose generative image model is applied once to a "
            "low-level restoration input without an Agent feedback loop."
        ),
        "request_count": len(input_paths),
        "model": args.model,
        "prompt": args.prompt,
        "config": {
            "api_base": args.api_base,
            "seed": args.seed,
            "steps": args.steps,
            "guidance": args.guidance,
            "scale": args.scale,
            "max_side": args.max_side,
            "request_timeout_seconds": args.request_timeout_seconds,
            "poll_timeout_seconds": args.poll_timeout_seconds,
            "poll_interval_seconds": args.poll_interval_seconds,
        },
        "finished_at": datetime.now().astimezone().isoformat(),
        "git_head": git_head(),
        "httpx_version": package_version("httpx"),
        "script_path": str(Path(__file__).resolve()),
        "script_sha256": sha256_file(Path(__file__).resolve()),
        "records": records,
    }
    for record in records:
        output_path = Path(str(record["output_path"]))
        manifest_path = output_path.parent / "manifest.json"
        image_manifest = {**run_manifest, "request_count": 1, "records": [record]}
        manifest_path.write_text(
            json.dumps(image_manifest, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    print(f"[done] run_dir={run_dir}")
    print(
        "[next] Fill each image's human_text_review.csv by transcribing "
        "LR, HR, and output text."
    )


if __name__ == "__main__":
    main()
