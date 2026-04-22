import json
import os
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

import cv2
import numpy as np
from pydantic import BaseModel, Field

from agents.zjc.calculate import calculate_uciqe, calculate_uiqm
from core.llm import GeminiLLM, OpenAILLM
from core.prompt import PromptLib
from core.schemas import ImageContent, Message, TextContent


class AnalyzeOutput(BaseModel):
    """AnalyzeOutput."""

    target: str
    nano_banana_prompt: str


class EvaluateOutput(BaseModel):
    """EvaluateOutput."""

    artifact_score: float = Field(description="Score for artifacts")
    over_adjustment_score: float = Field(description="Score for over-adjustment")
    color_accuracy_score: float = Field(description="Score for color accuracy")
    structural_integrity_score: float = Field(
        description="Score for structural integrity"
    )
    overall_score: float = Field(description="Overall score")
    should_continue: bool = Field(description="Whether the restoration should continue")


class ReflectOutput(BaseModel):
    """ReflectOutput."""

    decision: str = Field(description="'continue' or 'done'")
    memory: str = Field(description="Reflections to remember for the next round")


class State(Enum):
    """States of States Machine."""

    ANALYZE = "analyze"
    EDIT = "edit"
    EVALUATE = "evaluate"
    REFLECT = "reflect"
    DONE = "done"


@dataclass
class AgentContent:
    """The Content of Reflexion Agent."""

    input_path: str
    output_dir: str

    analyzer: OpenAILLM | GeminiLLM
    editor: GeminiLLM

    prompt_lib: PromptLib

    original_image: ImageContent | None = None
    current_image: ImageContent | None = None
    last_analysis: dict | None = None
    last_evaluation: dict | None = None
    reflections: list[str] = field(default_factory=list)
    history: list[Message] = field(default_factory=list)

    original_uiqm: float | None = None
    original_uciqe: float | None = None
    current_uiqm: float | None = None
    current_uciqe: float | None = None
    uiqm_history: list[float] = field(default_factory=list)
    uciqe_history: list[float] = field(default_factory=list)

    max_round: int = 5
    min_round: int = 3
    first_round: bool = True
    current_round: int = 0


def run(ctx: AgentContent) -> None:
    """Main Loop of Reflexion Agent."""
    ctx.original_image = ImageContent.from_file(ctx.input_path)
    print(f"[run] input image loaded: {ctx.input_path}")
    state = State.ANALYZE
    print(
        f"[run] start state machine, min_round={ctx.min_round}, max_round={ctx.max_round}"
    )

    while state is not State.DONE:
        print(f"[run] entering state={state.value}, round={ctx.current_round}")
        if state is State.ANALYZE:
            state = handle_analyze(ctx)
            continue
        if state is State.EDIT:
            state = handle_edit(ctx)
            continue
        if state is State.EVALUATE:
            state = handle_evaluate(ctx)
            continue
        if state is State.REFLECT:
            state = handle_reflect(ctx)
            continue

        raise ValueError(f"unknown state: {state}")

    print(f"[run] finished at round={ctx.current_round}")


def _image_to_bgr(image: ImageContent) -> np.ndarray:
    image_array = np.frombuffer(image.source, dtype=np.uint8)
    bgr = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError("failed to decode image content")
    return bgr


def handle_analyze(ctx: AgentContent) -> State:
    """Analyze the image and generate a restoration prompt."""
    memory = "No previous experience. This is the first restoration round."
    if not ctx.first_round:
        memory = "\n".join(f"- {ref}" for ref in ctx.reflections)

    image = ctx.original_image if ctx.first_round else ctx.current_image
    if image is None:
        raise ValueError("no image for analyze stage")

    prompt = TextContent(text=ctx.prompt_lib["start"].render(memory=memory))
    analyze_input = Message(content=[prompt, image])
    res = ctx.analyzer.generate_struct(analyze_input, schema=AnalyzeOutput)

    data = json.loads(res.text)
    print("[analyze] model output parsed")
    print(f"[analyze] target={data.get('target', '')}")
    print(f"[analyze] prompt={data.get('nano_banana_prompt', '')}")

    ctx.last_analysis = data
    ctx.history.append(analyze_input)
    ctx.history.append(res)
    ctx.first_round = False

    return State.EDIT


def handle_edit(ctx: AgentContent) -> State:
    """Edit image with Gemini according to analyze result."""
    if ctx.last_analysis is None:
        raise ValueError("no analyze result for edit stage")

    if ctx.current_image is None and ctx.original_image is None:
        raise ValueError("no image for edit stage")

    image = ctx.current_image or ctx.original_image
    if image is None:
        raise ValueError("no image for edit stage")
    prompt_text = json.dumps(ctx.last_analysis, ensure_ascii=False)
    edit_input = Message(content=[TextContent(text=prompt_text), image])
    res = ctx.editor.edit_image(edit_input)

    if not res.images:
        raise ValueError("edit stage returned no image")

    ctx.current_image = res.images[0]
    ctx.current_round += 1
    ctx.history.append(edit_input)
    ctx.history.append(res)

    output_dir = Path(ctx.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"round_{ctx.current_round}_out.jpg"
    ctx.current_image.save_to_file(output_path)
    print(f"[edit] round={ctx.current_round} image saved: {output_path}")

    return State.EVALUATE


def handle_evaluate(ctx: AgentContent) -> State:
    """Evaluate image quality using metrics and LLM judgment."""
    if ctx.original_image is None:
        raise ValueError("no original image for evaluate stage")
    if ctx.current_image is None:
        raise ValueError("no current image for evaluate stage")

    if ctx.original_uiqm is None or ctx.original_uciqe is None:
        original_bgr = _image_to_bgr(ctx.original_image)
        original_uiqm = calculate_uiqm(original_bgr)
        if not isinstance(original_uiqm, float):
            raise ValueError("calculate_uiqm returned unexpected submetrics tuple")
        ctx.original_uiqm = original_uiqm
        ctx.original_uciqe = calculate_uciqe(original_bgr)

    current_bgr = _image_to_bgr(ctx.current_image)
    current_uiqm = calculate_uiqm(current_bgr)
    if not isinstance(current_uiqm, float):
        raise ValueError("calculate_uiqm returned unexpected submetrics tuple")
    current_uciqe = calculate_uciqe(current_bgr)

    ctx.current_uiqm = current_uiqm
    ctx.current_uciqe = current_uciqe
    ctx.uiqm_history.append(current_uiqm)
    ctx.uciqe_history.append(current_uciqe)
    print(
        "[evaluate] metrics "
        f"original(uiqm={ctx.original_uiqm:.6f}, uciqe={ctx.original_uciqe:.6f}) -> "
        f"current(uiqm={ctx.current_uiqm:.6f}, uciqe={ctx.current_uciqe:.6f})"
    )

    prompt_text = ctx.prompt_lib["evaluation"].render(
        original_uiqm=f"{ctx.original_uiqm:.6f}",
        original_uciqe=f"{ctx.original_uciqe:.6f}",
        current_uiqm=f"{ctx.current_uiqm:.6f}",
        current_uciqe=f"{ctx.current_uciqe:.6f}",
    )
    evaluate_input = Message(
        content=[TextContent(text=prompt_text), ctx.original_image, ctx.current_image]
    )
    res = ctx.analyzer.generate_struct(evaluate_input, schema=EvaluateOutput)
    data = json.loads(res.text)
    print("[evaluate] llm evaluation parsed")
    print(
        "[evaluate] scores "
        f"artifact={data.get('artifact_score', 'n/a')}, "
        f"over_adjustment={data.get('over_adjustment_score', 'n/a')}, "
        f"color_accuracy={data.get('color_accuracy_score', 'n/a')}, "
        f"structure={data.get('structural_integrity_score', 'n/a')}, "
        f"overall={data.get('overall_score', 'n/a')}"
    )
    print(f"[evaluate] should_continue={data.get('should_continue', 'n/a')}")

    ctx.last_evaluation = data
    ctx.history.append(evaluate_input)
    ctx.history.append(res)

    return State.REFLECT


def handle_reflect(ctx: AgentContent) -> State:
    """Reflect on evaluation results and decide next action."""
    if ctx.last_evaluation is None:
        raise ValueError("no evaluate result for reflect stage")

    if ctx.current_round >= ctx.max_round:
        print("[reflect] reached max_round, force done")
        return State.DONE

    evaluation_text = json.dumps(ctx.last_evaluation, ensure_ascii=False)
    prompt_text = ctx.prompt_lib["reflection"].render(
        evaluation=evaluation_text,
        round=str(ctx.current_round),
        max_round=str(ctx.max_round),
    )
    reflect_input = Message(content=[TextContent(text=prompt_text)])
    res = ctx.analyzer.generate_struct(reflect_input, schema=ReflectOutput)
    data = json.loads(res.text)

    decision = str(data.get("decision", "")).strip().lower()
    memory = str(data.get("memory", "")).strip()

    if ctx.current_round < ctx.min_round and decision == "done":
        print(
            "[reflect] decision=done before min_round, "
            f"force continue (round={ctx.current_round}, min_round={ctx.min_round})"
        )
        decision = "continue"

    print(f"[reflect] decision={decision}")
    print(f"[reflect] memory={memory}")

    ctx.history.append(reflect_input)
    ctx.history.append(res)
    if memory:
        ctx.reflections.append(memory)

    if decision == "done":
        print("[reflect] transition -> done")
        return State.DONE
    if decision == "continue":
        print("[reflect] transition -> analyze")
        return State.ANALYZE

    raise ValueError(f"reflect stage returned invalid decision: {decision}")


ctx = AgentContent(
    input_path="../../workspace/U45_1/in.png",
    output_dir="../../workspace/U45_1/",
    analyzer=OpenAILLM(
        "kimi-k2.5",
        api_key=os.getenv("KIMI_API_KEY"),
        base_url=os.getenv("KIMI_API_BASE"),
    ),
    editor=GeminiLLM("gemini-3-pro-image-preview"),
    prompt_lib=PromptLib("./prompts/"),
)

run(ctx)
