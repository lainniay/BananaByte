"""Run one scale-based OmniGen2 image restoration without changing official code."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import subprocess
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

import torch
from accelerate import Accelerator
from PIL import Image, ImageOps


DEFAULT_NEGATIVE_PROMPT = (
    "(((deformed))), blurry, over saturation, bad anatomy, disfigured, "
    "poorly drawn face, mutation, mutated, (extra_limb), (ugly), "
    "(poorly drawn hands), fused fingers, messy drawing, broken legs, "
    "censor, censored, censor_bar"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-dir", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--input-image", type=Path, required=True)
    parser.add_argument("--prompt-file", type=Path, required=True)
    parser.add_argument("--output-image", type=Path, required=True)
    parser.add_argument(
        "--mode",
        choices=("direct_scale", "preup_align"),
        default="direct_scale",
    )
    parser.add_argument("--scale", type=float, default=4.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--dtype", choices=("fp32", "fp16", "bf16"), default="bf16")
    parser.add_argument("--scheduler", choices=("euler", "dpmsolver++"), default="euler")
    parser.add_argument("--text-guidance-scale", type=float, default=5.0)
    parser.add_argument("--image-guidance-scale", type=float, default=2.0)
    parser.add_argument("--cfg-range-start", type=float, default=0.0)
    parser.add_argument("--cfg-range-end", type=float, default=1.0)
    parser.add_argument("--negative-prompt", default=DEFAULT_NEGATIVE_PROMPT)
    parser.add_argument("--enable-model-cpu-offload", action="store_true")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_metadata(path: Path, metadata: dict) -> None:
    path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def git_commit(repo_dir: Path) -> str:
    return subprocess.run(
        ["git", "-C", str(repo_dir), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def main() -> None:
    args = parse_args()
    if args.scale <= 0:
        raise ValueError("--scale must be positive")
    if args.steps <= 0:
        raise ValueError("--steps must be positive")

    repo_dir = args.repo_dir.resolve()
    model_path = args.model_path.resolve()
    input_path = args.input_image.resolve()
    prompt_path = args.prompt_file.resolve()
    output_path = args.output_image.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite output: {output_path}")

    prompt = prompt_path.read_text(encoding="utf-8").strip()
    if not prompt:
        raise ValueError("Prompt file is empty")

    input_image = ImageOps.exif_transpose(Image.open(input_path)).convert("RGB")
    input_width, input_height = input_image.size
    target_width = max(1, round(input_width * args.scale))
    target_height = max(1, round(input_height * args.scale))
    target_pixels = target_width * target_height
    if args.mode == "preup_align":
        condition_image = input_image.resize(
            (target_width, target_height), Image.Resampling.BICUBIC
        )
        align_res = True
        condition_resampling = "bicubic"
    else:
        condition_image = input_image
        align_res = False
        condition_resampling = None
    condition_width, condition_height = condition_image.size

    metadata_path = output_path.parent / "metadata.json"
    metadata = {
        "status": "running",
        "started_at_utc": utc_now(),
        "repo_commit": git_commit(repo_dir),
        "repo_dir": str(repo_dir),
        "model_path": str(model_path),
        "input_path": str(input_path),
        "input_sha256": sha256_file(input_path),
        "input_size": [input_width, input_height],
        "prompt_path": str(prompt_path),
        "prompt_sha256": sha256_file(prompt_path),
        "output_path": str(output_path),
        "mode": args.mode,
        "scale": args.scale,
        "target_size": [target_width, target_height],
        "condition_size": [condition_width, condition_height],
        "condition_resampling": condition_resampling,
        "seed": args.seed,
        "steps": args.steps,
        "dtype": args.dtype,
        "scheduler": args.scheduler,
        "text_guidance_scale": args.text_guidance_scale,
        "image_guidance_scale": args.image_guidance_scale,
        "cfg_range": [args.cfg_range_start, args.cfg_range_end],
        "align_res": align_res,
        "max_pixels": max(target_pixels, condition_width * condition_height),
        "max_input_image_side_length": max(condition_width, condition_height),
        "cuda_visible_device_name": (
            torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        ),
        "torch_version": torch.__version__,
    }
    write_metadata(metadata_path, metadata)

    try:
        sys.path.insert(0, str(repo_dir))
        official_inference = importlib.import_module("inference")

        accelerator = Accelerator(
            mixed_precision=args.dtype if args.dtype != "fp32" else "no"
        )
        weight_dtype = {
            "fp32": torch.float32,
            "fp16": torch.float16,
            "bf16": torch.bfloat16,
        }[args.dtype]

        official_args = argparse.Namespace(
            model_path=str(model_path),
            transformer_path=None,
            transformer_lora_path=None,
            scheduler=args.scheduler,
            enable_sequential_cpu_offload=False,
            enable_model_cpu_offload=args.enable_model_cpu_offload,
            enable_group_offload=False,
            enable_teacache=False,
            teacache_rel_l1_thresh=0.05,
            enable_taylorseer=False,
        )
        pipeline = official_inference.load_pipeline(
            official_args, accelerator, weight_dtype
        )
        generator = torch.Generator(device=accelerator.device).manual_seed(args.seed)

        result = pipeline(
            prompt=prompt,
            input_images=[condition_image],
            width=target_width,
            height=target_height,
            max_pixels=max(target_pixels, condition_width * condition_height),
            max_input_image_side_length=max(condition_width, condition_height),
            align_res=align_res,
            num_inference_steps=args.steps,
            max_sequence_length=1024,
            text_guidance_scale=args.text_guidance_scale,
            image_guidance_scale=args.image_guidance_scale,
            cfg_range=(args.cfg_range_start, args.cfg_range_end),
            negative_prompt=args.negative_prompt,
            num_images_per_prompt=1,
            generator=generator,
            output_type="pil",
        )
        output_image = result.images[0]
        output_image.save(output_path)
        if output_image.size != (target_width, target_height):
            raise RuntimeError(
                f"Output size {output_image.size} != target "
                f"{(target_width, target_height)}"
            )

        metadata.update(
            {
                "status": "complete",
                "finished_at_utc": utc_now(),
                "output_size": list(output_image.size),
                "output_sha256": sha256_file(output_path),
            }
        )
        write_metadata(metadata_path, metadata)
        print(json.dumps(metadata, indent=2, sort_keys=True))
    except Exception as exc:
        metadata.update(
            {
                "status": "failed",
                "finished_at_utc": utc_now(),
                "error": repr(exc),
                "traceback": traceback.format_exc(),
            }
        )
        write_metadata(metadata_path, metadata)
        raise


if __name__ == "__main__":
    main()
