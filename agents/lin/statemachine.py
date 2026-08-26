import json
import io
import os
import sys
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

import httpx
from PIL import Image
from pydantic import BaseModel, Field

from core.llm import GeminiLLM, OpenAILLM
from core.prompt import PromptLib
from core.schemas import ImageContent, Message, TextContent

from agents.lin.metrics import evaluate as calc_metrics


from PIL import ImageFilter




class AnalyzeOutput(BaseModel):
    """AnalyzeOutput."""
    target: str=Field(description="需要改善的目标")
    nano_banana_prompt: str=Field(description="分析得到的优化方案")


class EvaluateOutput(BaseModel):
    """EvaluateOutput."""
    sharpness_score:float=Field(description="清晰度评分 1-10")
    texture_score:float=Field(description="纹理细节评分 1-10")
    overall_score:float=Field(description="整体质量评分 1-10")



class ReflectOutput(BaseModel):
    """ReflectionOutput."""
    decision:str=Field(description="'continue' or 'done' ")
    memory:str=Field(description="本轮经验")

class State(Enum):
    """States of States Machine."""
    ANALYZE = "analyze"
    EDIT = "edit"
    EVALUATE = "evaluate"
    REFLECT = "reflect"
    DONE = "done"


@dataclass
class AgentContent:
    input_path: str
    output_dir: str
    analyzer: OpenAILLM | GeminiLLM
    editor: GeminiLLM
    prompt_lib: PromptLib
    ground_truth_path: str | None = None

    original_image: ImageContent | None = None
    current_image: ImageContent | None = None
    text_mask_image: ImageContent | None = None
    last_analysis: dict | None = None
    last_evaluation: dict | None = None
    reflections: list[str] = field(default_factory=list)
    history: list[Message] = field(default_factory=list)

    curr_psnr: float | None = None
    curr_ssim: float | None = None
    psnr_history: list[float] = field(default_factory=list)
    ssim_history: list[float] = field(default_factory=list)
    #记录每轮的overall——score.用边际收益递减判断
    score_history:list[float]=field(default_factory=list)
    min_improve:float=0.3 #提升阈值

    max_round: int = 5
    min_round: int = 3
    first_round: bool = True
    current_round: int = 0

    # 可选优化
    use_fidelity: bool = False
    use_progressive: bool = False
    use_text_mask: bool = False
    use_text_fusion: bool = False
    text_mask_path: str | None = None
    text_mask_scale: int = 1
    text_fusion_alpha: float = 0.2
    text_fusion_method: str = "lanczos"
    text_fusion_feather_radius: float = 0.0
    masked_prompt_name: str = "analyze_masked"
    edit_retry_attempts: int = 2
    edit_retry_wait_seconds: int = 15
    progressive_resolutions: list[str] = field(
        default_factory=lambda: ["1k", "2k", "2k"]
    )

def _ic_to_pil(ic:ImageContent)->Image.Image:
    """ImageContent ->PIL Image."""
    return Image.open(io.BytesIO(ic.source)).convert("RGB")


def _pil_to_ic(img: Image.Image) -> ImageContent:
    """PIL Image -> ImageContent."""
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return ImageContent(source=buf.getvalue(), mime_type="image/png")


def _default_text_mask_path(input_path: str) -> Path:
    input_file = Path(input_path)
    allcharac = input_file.parents[1]
    return allcharac / "craft_text_regions" / input_file.stem / "mask.png"


def _load_text_mask(ctx: AgentContent) -> ImageContent:
    mask_path = Path(ctx.text_mask_path) if ctx.text_mask_path else _default_text_mask_path(ctx.input_path)
    if not mask_path.exists():
        raise FileNotFoundError(f"text mask not found: {mask_path}")

    mask = Image.open(mask_path).convert("L")
    if ctx.text_mask_scale > 1:
        width, height = mask.size
        mask = mask.resize(
            (width * ctx.text_mask_scale, height * ctx.text_mask_scale),
            Image.Resampling.NEAREST,
        )
    mask_rgb = Image.merge("RGB", (mask, mask, mask))
    return _pil_to_ic(mask_rgb)


def _resampling_method(name: str) -> Image.Resampling:
    methods = {
        "lanczos": Image.Resampling.LANCZOS,
        "bicubic": Image.Resampling.BICUBIC,
        "nearest": Image.Resampling.NEAREST,
    }
    if name not in methods:
        raise ValueError(f"unsupported text_fusion_method: {name}")
    return methods[name]


def _apply_text_fusion(ctx: AgentContent) -> None:
    """Pull text-mask regions toward deterministic traditional upsampling."""
    if ctx.original_image is None or ctx.current_image is None:
        raise ValueError("text fusion requires original_image and current_image")
    if ctx.text_mask_image is None:
        raise ValueError("text fusion requires text_mask_image")
    if not 0.0 <= ctx.text_fusion_alpha <= 1.0:
        raise ValueError("text_fusion_alpha must be between 0 and 1")

    creative = _ic_to_pil(ctx.current_image)
    original = _ic_to_pil(ctx.original_image)
    traditional = original.resize(
        creative.size,
        _resampling_method(ctx.text_fusion_method),
    )
    mask = _ic_to_pil(ctx.text_mask_image).convert("L").resize(
        creative.size,
        Image.Resampling.NEAREST,
    )
    if ctx.text_fusion_feather_radius > 0:
        mask = mask.filter(ImageFilter.GaussianBlur(radius=ctx.text_fusion_feather_radius))

    text_region = Image.blend(traditional, creative, ctx.text_fusion_alpha)
    fused = Image.composite(text_region, creative, mask)
    ctx.current_image = _pil_to_ic(fused)
    print(
        "[edit] text fusion applied "
        f"alpha={ctx.text_fusion_alpha}, method={ctx.text_fusion_method}, size={creative.size}"
    )


def _edit_image_with_timeout_retry(
    ctx: AgentContent,
    msg: Message,
    config: GeminiLLM.EditImageConfig,
) -> Message:
    """Retry Gemini image editing when the proxy/API response stalls."""
    last_error: BaseException | None = None
    attempts = max(1, ctx.edit_retry_attempts)

    for attempt in range(1, attempts + 1):
        try:
            if attempts > 1:
                print(f"[edit] Gemini image edit attempt {attempt}/{attempts}")
            return ctx.editor.edit_image(msg, config=config)
        except httpx.TimeoutException as exc:
            last_error = exc
            if attempt >= attempts:
                break
            print(
                "[edit] Gemini image edit timed out; "
                f"retrying after {ctx.edit_retry_wait_seconds}s"
            )
            time.sleep(ctx.edit_retry_wait_seconds)

    raise RuntimeError(
        "Gemini image edit timed out. The image-generation request reached the "
        "HTTP read timeout, usually because the proxy/model response is too slow. "
        "Try increasing LIN_GEMINI_TIMEOUT_MS or lowering the requested image size."
    ) from last_error



def run(ctx: AgentContent) -> None:
    ctx.original_image = ImageContent.from_file(ctx.input_path)
    if ctx.use_text_mask:
        ctx.text_mask_image = _load_text_mask(ctx)
        print("[run] text mask loaded")
    print(f"[run] input image loaded: {ctx.input_path}")
    state = State.ANALYZE
    print("[run] start state machine")

    while state is not State.DONE:
        print(f"[run] entering state={state.value}, round={ctx.current_round}")

        if state is State.ANALYZE:
            state = handle_analyze(ctx)
            continue
        if state is State.EDIT:
            state = handle_edit(ctx)
            continue
        if state is State.EVALUATE:
            state = handle_eval(ctx)
            continue
        if state is State.REFLECT:
            state = handle_reflect(ctx)
            continue

        raise ValueError(f"unknown state: {state}")

    print(f"========Finished at round: {ctx.current_round}===============")


def handle_analyze(ctx: AgentContent) -> State:
    """A restroration plan."""
    memory="首次超分，无历史经验"
    if not ctx.first_round:
        memory="\n".join(f" -{ref}" for ref in ctx.reflections)
        #过去每一轮reflect产生的经验拼成一个markdown列表

    image=ctx.original_image if ctx.first_round else ctx.current_image
    #后面每一轮分析上一轮绣出来的图
    if image is None:
        raise ValueError("没有可用于分析的图像")

    prompt_name = ctx.masked_prompt_name if ctx.use_text_mask else "analyze"
    prompt_text=ctx.prompt_lib[prompt_name].render(memory=memory) #从prompts/目录里加在所有的.md,把analyze的memory转换成实际文字
    content = [TextContent(text=prompt_text), image]
    if ctx.use_text_mask and ctx.text_mask_image is not None:
        content.append(ctx.text_mask_image)
    msg=Message(content=content) #一段文字+一张图，一起发给模型
    res=ctx.analyzer.generate_struct(msg,schema=AnalyzeOutput) #让模型必须按某个固定的json结构回答

    data=json.loads(res.text)
    print(f"[analyze] target={data.get('target','')}")
    print(f"[analyze] prompt={data.get('nano_banana_prompt','')}")

    ctx.last_analysis=data
    ctx.history.append(msg)
    ctx.history.append(res)
    ctx.first_round=False

    return State.EDIT




def handle_edit(ctx: AgentContent) -> State:
    """Nano banana 处理图像."""
    if ctx.last_analysis is None:
        raise ValueError("No analyze result for edit stage")

    if ctx.current_image is None and ctx.original_image is None:
        raise ValueError("No image found")

    image=ctx.current_image or ctx.original_image
    prompt_text=json.dumps(ctx.last_analysis,ensure_ascii=False)
    content = [TextContent(text=prompt_text), image]
    if ctx.use_text_mask and ctx.text_mask_image is not None:
        content.append(ctx.text_mask_image)
    msg=Message(content=content)

    # progressive: 根据轮次选 resolution
    if ctx.use_progressive:
        res_list = ctx.progressive_resolutions
        idx = min(ctx.current_round, len(res_list) - 1)
        resolution = res_list[idx]
        print(f"[edit] progressive resolution={resolution}")
    else:
        resolution = None

    config = GeminiLLM.EditImageConfig(temperature=0.4)
    if resolution:
        config.image_size = resolution
    res=_edit_image_with_timeout_retry(ctx,msg,config)

    if not res.images:
        raise ValueError("SR 未返回图像")

    ctx.current_image=res.images[0]
    ctx.current_round+=1

    # fidelity: SR 完之后用低频对齐抑制幻觉
    if ctx.use_fidelity:
        from agents.lin.fidelity import fidelity_blend
        original_pil = _ic_to_pil(ctx.original_image)
        current_pil = _ic_to_pil(ctx.current_image)
        blended = fidelity_blend(original_pil, current_pil, alpha=0.3)
        ctx.current_image = _pil_to_ic(blended)
        print("[edit] fidelity blend applied")

    if ctx.use_text_fusion:
        _apply_text_fusion(ctx)

    ctx.history.append(msg)
    ctx.history.append(res)


    #存图
    Path(ctx.output_dir).mkdir(parents=True,exist_ok=True)
    output_path=Path(ctx.output_dir)/f"round_{ctx.current_round}_out.png"
    ctx.current_image.save_to_file(output_path)
    print(f"[edit round={ctx.current_round} saved:{output_path}]")

    return State.EVALUATE


def handle_eval(ctx: AgentContent) -> State:
    """有参考指标只记录,LLM评估只看图+无参考."""
    if ctx.original_image is None or ctx.current_image is None:
        raise ValueError("缺少图像")

    current_pil=_ic_to_pil(ctx.current_image)

    #有参考指标
    lr_path=Path(ctx.input_path)
    hr_path=(
        Path(ctx.ground_truth_path)
        if ctx.ground_truth_path
        else lr_path.parent.parent/"HR"/lr_path.name
    )
    gt_pil=Image.open(hr_path).convert("RGB") if hr_path.exists() else None
    scores=calc_metrics(current_pil,ground_truth=gt_pil)


    ctx.curr_psnr=scores.get("psnr")
    ctx.curr_ssim=scores.get("ssim")
    ctx.psnr_history.append(ctx.curr_psnr or 0)
    ctx.ssim_history.append(ctx.curr_ssim or 0)
    print(f"[eval psnr={ctx.curr_psnr},ssim={ctx.curr_ssim}]")


    #无参考指标
    lfd=scores.get("low_freq_dev")
    ee=scores.get("edge_expansion")
    print(f"[eval] low_freq_dev={lfd},edge_expansion={ee}")

    #LLM评估
    lfd_str=f"{lfd:.2f}" if lfd else "N/A"
    ee_str=f"{ee:.2f}" if ee else "N/A"
    prompt_text=ctx.prompt_lib["evaluate"].render(low_freq_dev=lfd_str,edge_expansion=ee_str)
    msg=Message(content=[TextContent(text=prompt_text),ctx.original_image,ctx.current_image])

    res=ctx.analyzer.generate_struct(msg,schema=EvaluateOutput)
    data = json.loads(res.text)
    overall=float(data.get("overall_score")or 0)
    ctx.score_history.append(overall)
    print(f"[eval] overall={data.get('overall_score')}")

    ctx.last_evaluation=data
    ctx.history.append(msg) #把评估的每一轮的提示词输入与回答全都放memory里
    ctx.history.append(res)
    return State.REFLECT


def handle_reflect(ctx: AgentContent) -> State:
    """反思评估结果,决定继续或结束."""
    if ctx.last_evaluation is None:
        raise ValueError("No eval result")

    if ctx.current_round>=ctx.max_round:
        print(f"达到max_round={ctx.max_round},结束")
        return State.DONE

    if len(ctx.score_history)>=2 and ctx.current_round>=ctx.min_round:
        improve=ctx.score_history[-1]-ctx.score_history[-2]
        print(f"[reflect] 本轮提升Δ={improve:.2f}")
        if improve<ctx.min_improve:
            print(f"边际效益递减(Δ<{improve:.2f}<{ctx.min_improve}),直接截止")
            return State.DONE

    eval_text=json.dumps(ctx.last_evaluation,ensure_ascii=False)
    prompt_text=ctx.prompt_lib["reflect"].render(
        evaluation=eval_text,
        round=str(ctx.current_round),
        max_round=str(ctx.max_round),

    )

    msg=Message(content=[TextContent(text=prompt_text)])
    res=ctx.analyzer.generate_struct(msg,schema=ReflectOutput)
    data=json.loads(res.text)

    decision=str(data.get("decision","")).strip().lower()
    memory=str(data.get("memory","")).strip()

    if ctx.current_round<ctx.min_round and decision=="done":
        print(f"未达 min_round={ctx.min_round},强制继续")
        decision="continue"

    print(f"decision={decision}")
    if memory:
        ctx.reflections.append(memory)
        print(f"memory={memory}")

    if decision == "done":
        print("================Done======================")
        return State.DONE
    if decision == "continue":
        print("==========[NEED ANOTHER TRY]====================")
        return State.ANALYZE
    raise ValueError(f"reflect returned invalid decision: {decision}")


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    base=Path(__file__).parents[2]/"workspace/SR/COLLECT"
    editor_timeout_ms=int(os.getenv("LIN_GEMINI_TIMEOUT_MS", "600000"))
    edit_retry_attempts=int(os.getenv("LIN_EDIT_RETRY_ATTEMPTS", "2"))

    analyzer=OpenAILLM("gpt-5.1")
    editor=GeminiLLM("gemini-3-pro-image-preview", timeout=editor_timeout_ms)
    prompt_lib=PromptLib(Path(__file__).parent / "prompts")

    stems=os.getenv("LIN_IMAGE_STEMS", "IMG_003,IMG_009").split(",")
    for stem in stems:
        input_path=base/"LR_256"/f"{stem}.png"
        ctx = AgentContent(
            input_path=str(input_path),
            output_dir=str(base/"outputs"/stem),
            analyzer=analyzer,
            editor=editor,
            prompt_lib=prompt_lib,
            use_text_mask=False,
            use_text_fusion=False,
            use_fidelity=False,
            use_progressive=False,
            edit_retry_attempts=edit_retry_attempts,
        )
        run(ctx)


