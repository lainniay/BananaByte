"""Run one Agent/VLM SR pass for every image in COLLECT/selected_LR."""

import json
import os
import sys
from datetime import datetime
from pathlib import Path

from PIL import Image

from core.llm import GeminiLLM, OpenAILLM
from core.prompt import PromptLib
from agents.lin.statemachine import AgentContent, run


ANALYZER_MODEL = "gpt-5.1"
EDITOR_MODEL = "gemini-3-pro-image-preview"


def find_hr(hr_dir: Path, lr_path: Path) -> Path:
    stem = lr_path.stem
    for suffix in ("_128", "_256", "_512"):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
            break
    for extension in (".png", ".jpg", ".jpeg"):
        candidate = hr_dir / f"{stem}{extension}"
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"HR not found for {lr_path.name}")


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    root = Path(__file__).parents[2] / "workspace/SR/COLLECT"
    input_dir = root / "selected_LR"
    output_root = input_dir / "vlm_outputs"
    output_root.mkdir(parents=True, exist_ok=True)

    inputs = sorted(input_dir.glob("*.png"))
    analyzer = OpenAILLM(ANALYZER_MODEL)
    editor = GeminiLLM(
        EDITOR_MODEL,
        timeout=int(os.getenv("LIN_GEMINI_TIMEOUT_MS", "600000")),
    )
    prompt_lib = PromptLib(Path(__file__).parent / "prompts")

    print(f"[batch] {len(inputs)} inputs")
    for index, input_path in enumerate(inputs, start=1):
        item_dir = output_root / input_path.stem
        metadata_path = item_dir / "metadata.json"
        if metadata_path.exists():
            print(f"[{index}/{len(inputs)}] skip existing: {input_path.name}")
            continue

        hr_path = find_hr(root / "HR", input_path)
        print(f"[{index}/{len(inputs)}] process: {input_path.name}")
        ctx = AgentContent(
            input_path=str(input_path),
            output_dir=str(item_dir),
            ground_truth_path=str(hr_path),
            analyzer=analyzer,
            editor=editor,
            prompt_lib=prompt_lib,
            use_text_mask=False,
            use_text_fusion=False,
            use_fidelity=False,
            use_progressive=False,
            edit_retry_attempts=int(os.getenv("LIN_EDIT_RETRY_ATTEMPTS", "2")),
        )
        run(ctx)
        output_path = item_dir / f"round_{ctx.current_round}_out.png"

        with Image.open(input_path) as lr_image, Image.open(hr_path) as hr_image:
            metadata = {
                "input": str(input_path.resolve()),
                "hr": str(hr_path.resolve()),
                "output": str(output_path.resolve()),
                "lr_size": list(lr_image.size),
                "hr_size": list(hr_image.size),
                "analyzer_model": ANALYZER_MODEL,
                "editor_model": EDITOR_MODEL,
                "rounds": ctx.current_round,
                "analysis": ctx.last_analysis,
                "evaluation": ctx.last_evaluation,
                "psnr_history": ctx.psnr_history,
                "ssim_history": ctx.ssim_history,
                "overall_score_history": ctx.score_history,
                "created_at": datetime.now().astimezone().isoformat(),
            }
        item_dir.mkdir(parents=True, exist_ok=True)
        metadata_path.write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
