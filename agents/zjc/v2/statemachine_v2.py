import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Literal, cast

from rich.console import Console
from rich.json import JSON
from rich.panel import Panel
from rich.table import Table

from agents.zjc import calculate
from agents.zjc.format import Analyze, EvaluationResponse, ReflectResponse
from core import GeminiLLM, ImageContent, PromptLib
from core.schemas import Message, TextContent


class State(Enum):
    """States of States Machine."""

    ANALYZE = "analyze"
    EDIT = "edit"
    EVALUATE = "evaluate"
    REFLECT = "reflect"
    DONE = "done"


Branchs = Literal["ColorDistortion", "ContrastReduction", "TextureBlurring"]

openai_config = GeminiLLM.GenerateConfig(temperature=0, top_p=0.1, seed=39)

console = Console()


def _save_img(img: ImageContent, output_dir: str, name: str) -> Path:
    """Save image to output_dir with given name, create dir if needed."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}.jpg"
    img.save_to_file(path)
    return path


@dataclass
class AgentContent:
    """TODO."""

    input_path: str
    output_dir: str

    analyzer: GeminiLLM
    editor: GeminiLLM
    evaluator: GeminiLLM
    reflector: GeminiLLM

    prompt_lib: PromptLib

    input_img: ImageContent | None = None
    cur_img: ImageContent | None = None

    first: bool = True

    last_branch: Branchs | None = None
    cur_branchs: list[Branchs] = field(
        default_factory=lambda: [
            "ColorDistortion",
            "ContrastReduction",
            "TextureBlurring",
        ]
    )

    candidates: dict[Branchs, ImageContent | None] = field(
        default_factory=lambda: {
            "ColorDistortion": None,
            "ContrastReduction": None,
            "TextureBlurring": None,
        }
    )

    analyzes: dict[Branchs, str] = field(
        default_factory=lambda: {
            "ColorDistortion": "",
            "ContrastReduction": "",
            "TextureBlurring": "",
        }
    )

    evaluations: dict[Branchs, str] = field(
        default_factory=lambda: {
            "ColorDistortion": "",
            "ContrastReduction": "",
            "TextureBlurring": "",
        }
    )

    max_round: int = 5
    min_round: int = 0
    cur_round: int = 0


def run(ctx: AgentContent) -> None:
    """TODO."""
    ctx.input_img = ImageContent.from_file(ctx.input_path)
    ctx.cur_img = ctx.input_img

    _save_img(ctx.input_img, ctx.output_dir, "original")
    console.print(
        Panel(
            f"Input: [bold]{ctx.input_path}[/]  max_rounds={ctx.max_round}",
            title="[bold blue]Start[/]",
        )
    )

    state = State.ANALYZE

    handlers = {
        State.ANALYZE: handle_analyze,
        State.EDIT: handle_edit,
        State.EVALUATE: handle_evaluate,
        State.REFLECT: handle_reflect,
        State.DONE: handle_done,
    }

    while state != State.DONE:
        handler = handlers[state]
        state = handler(ctx)

    handlers[State.DONE](ctx)


def handle_analyze(ctx: AgentContent) -> State:
    """Analyze current image for each active branch."""
    if ctx.cur_img:
        _save_img(ctx.cur_img, ctx.output_dir, f"r{ctx.cur_round}_in")

    console.print(
        Panel(
            f"Round [cyan]{ctx.cur_round + 1}/{ctx.max_round}[/]  branches={ctx.cur_branchs}",
            title="[bold red]Analyze[/]",
        )
    )

    def _do_branch(name: Branchs) -> None:
        if ctx.cur_img:
            mes = Message(
                content=[TextContent(text=ctx.prompt_lib[name].render()), ctx.cur_img]
            )
            res = ctx.analyzer.generate_struct(
                mes, schema=Analyze, config=openai_config
            )
            ctx.analyzes[name] = json.loads(res.text).get("prompt", "")
            console.print(f"  [dim]\\[{name}][/]")
            console.print(JSON(res.text, indent=4))

    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = [executor.submit(_do_branch, name) for name in ctx.cur_branchs]
        for future in as_completed(futures):
            future.result()

    return State.EDIT


def handle_edit(ctx: AgentContent) -> State:
    """Edit image for each active branch and save results."""
    console.print(Panel("", title="[bold red]Edit[/]"))

    def _do_branch(name: Branchs) -> None:
        if ctx.cur_img and ctx.analyzes[name]:
            mes = Message(content=[TextContent(text=ctx.analyzes[name]), ctx.cur_img])
            res = ctx.editor.edit_image(mes)
            if res.images:
                ctx.candidates[name] = res.images[0]
                path = _save_img(
                    res.images[0], ctx.output_dir, f"{name}_r{ctx.cur_round}"
                )
                console.print(f"  [dim]\\[{name}][/] [green]saved[/] -> {path.name}")

    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = [executor.submit(_do_branch, name) for name in ctx.cur_branchs]
        for future in as_completed(futures):
            future.result()

    return State.EVALUATE


def handle_evaluate(ctx: AgentContent) -> State:
    """Evaluate each candidate image with metrics and LLM judgment."""
    console.print(Panel("", title="[bold red]Evaluate[/]"))

    def _do_branch(name: Branchs) -> None:
        if ctx.cur_img and ctx.candidates[name]:
            uiqm = calculate.calculate_uiqm(cast(ImageContent, ctx.candidates[name]))
            uciqe = calculate.calculate_uciqe(cast(ImageContent, ctx.candidates[name]))
            mes = Message(
                content=[
                    TextContent(text=ctx.prompt_lib["evaluate"].render()),
                    TextContent(text="first:"),
                    ctx.cur_img,
                    TextContent(text="second:"),
                    cast(ImageContent, ctx.candidates[name]),
                ]
            )
            res = ctx.evaluator.generate_struct(
                mes, schema=EvaluationResponse, config=openai_config
            )
            data = json.loads(res.text)
            data["branch"] = name
            data["uiqm"] = uiqm
            data["uciqe"] = uciqe
            ctx.evaluations[name] = json.dumps(data)
            console.print(f"  [dim]\\[{name}][/]")
            console.print(JSON(res.text, indent=4))
            console.print(
                f"    uiqm=[yellow]{uiqm:.4f}[/]  uciqe=[yellow]{uciqe:.4f}[/]  "
                f"score=[yellow]{data.get('overall_score', '?')}[/]"
            )

    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = [executor.submit(_do_branch, name) for name in ctx.cur_branchs]
        for future in as_completed(futures):
            future.result()

    return State.REFLECT


def handle_reflect(ctx: AgentContent) -> State:
    """Reflect on evaluations, pick the best branch, and decide whether to continue."""
    console.print(Panel("", title="[bold red]Reflect[/]"))

    evaluation_summary = json.dumps(
        {
            name: json.loads(ctx.evaluations[name])
            for name in ctx.cur_branchs
            if ctx.evaluations[name]
        },
    )

    mes = Message(
        content=[
            TextContent(
                text=ctx.prompt_lib["reflect"].render(evaluation=evaluation_summary),
            ),
        ],
    )
    res = ctx.reflector.generate_struct(
        mes, schema=ReflectResponse, config=openai_config
    )
    data = json.loads(res.text)

    best_branch: Branchs = cast(Branchs, data["best_branch"])
    should_continue: bool = data["should_continue"]

    best_img = ctx.candidates.get(best_branch)
    if best_img is not None:
        ctx.cur_img = best_img
    ctx.last_branch = best_branch

    all_branches: list[Branchs] = [
        "ColorDistortion",
        "ContrastReduction",
        "TextureBlurring",
    ]
    ctx.cur_branchs = [branch for branch in ctx.cur_branchs if branch != best_branch]

    for branch in all_branches:
        ctx.candidates[branch] = None
        ctx.evaluations[branch] = ""
        ctx.analyzes[branch] = ""

    ctx.cur_round += 1

    if len(ctx.cur_branchs) == 0:
        return State.DONE

    # 未达最低轮数, 强制执行下一轮
    if not should_continue and ctx.cur_round < ctx.min_round:
        console.print(
            f"  [yellow]LLM wants stop but min_round not met ({ctx.cur_round}/{ctx.min_round}), force continue[/]"
        )
        should_continue = True

    table = Table(show_header=False, box=None)
    table.add_row("[cyan]best_branch[/]", best_branch)
    table.add_row("[cyan]should_continue[/]", str(should_continue))
    table.add_row("[cyan]remaining[/]", str(ctx.cur_branchs))
    table.add_row("[cyan]round[/]", f"{ctx.cur_round}/{ctx.max_round}")
    console.print(table)

    reasoning = data.get("reasoning", "")
    if reasoning:
        console.print(f"  [dim italic]{reasoning}[/]")

    # if not should_continue or ctx.cur_round >= ctx.max_round:
    # return State.DONE
    return State.ANALYZE


def handle_done(ctx: AgentContent) -> State:
    """Finish the state machine and save the final result."""
    console.print(
        Panel(
            f"Rounds completed: [cyan]{ctx.cur_round}[/]", title="[bold green]Done[/]"
        )
    )

    if ctx.cur_img:
        output_path = Path(ctx.output_dir) / "out.jpg"
        ctx.cur_img.save_to_file(output_path)
        console.print(f"  Final image saved -> [bold]{output_path}[/]")

    return State.DONE


llm = GeminiLLM(
    "gemini-3.1-flash-lite-preview",
    api_key=os.getenv("OPENAI_API_KEY"),
    timeout=1200 * 1000,
)


ctx = AgentContent(
    input_path="../../workspace/LSUI_2376/in.jpg",
    output_dir="../../workspace/LSUI_2376/",
    analyzer=llm,
    editor=GeminiLLM("gemini-3.1-flash-image-preview", timeout=1200 * 1000),
    evaluator=GeminiLLM(
        "gemini-3.1-pro-preview",
        api_key=os.getenv("OPENAI_API_KEY"),
        timeout=1200 * 1000,
    ),
    reflector=GeminiLLM(
        "gemini-3.1-pro-preview",
        api_key=os.getenv("OPENAI_API_KEY"),
        timeout=1200 * 1000,
    ),
    prompt_lib=PromptLib("./prompts_v2/"),
)

run(ctx)
