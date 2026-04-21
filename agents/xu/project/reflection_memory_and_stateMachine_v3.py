import base64
import io
import os
from dataclasses import dataclass
from enum import Enum
from pathlib import Path  # <--- 添加这一行

from dotenv import load_dotenv
from PIL import Image  # 确保环境已安装: pip install pillow

# 确保导入了核心组件
from core import GeminiLLM, ImageContent, Message, OpenAILLM, TextContent, create_llm

# --- 1. 辅助模块：图像预处理 ---


def get_compressed_base64(path: str, max_size: int = 1024) -> str:
    """压缩图像并转为 Base64, 大幅降低网络错误风险."""
    print(f"📦 正在压缩处理图像: {os.path.basename(path)} ...")
    with Image.open(path) as img:
        # 统一转 RGB 模式（JPEG 不支持 Alpha 通道，防止报错）
        if img.mode != "RGB":
            img = img.convert("RGB")

        # 缩放到合理尺寸，Gemini 对 1024px 的反光识别能力极佳
        img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)

        buffered = io.BytesIO()
        # 质量 85% 是体积与画质的黄金平衡点
        img.save(buffered, format="JPEG", quality=85, optimize=True)
        return base64.b64encode(buffered.getvalue()).decode("utf-8")


# --- 2. 状态机定义 ---


class State(Enum):
    """状态机的各个阶段."""

    SEGMENT = "segment"  # 接收预分割结果
    IMAGINE_BACKGROUND = "imagine_background"  # 阶段 1: 背景想象补全
    MERGE_BASE = "merge_base"  # 阶段 2 启动: 融合原图与想象背景
    EDIT = "edit"  # 迭代精修阶段
    EVALUATE = "evaluate"  # 专家评审
    DONE = "done"  # 完成


@dataclass
class AgentContext:
    """Agent 的上下文, 用于在状态机中传递数据."""

    task_description: str
    image_client: GeminiLLM
    critic_client: OpenAILLM
    seg_result_txt: str  # 预先得到的分割结果文本

    # 图像存储
    original_image: ImageContent | None = None  # 原始输入图
    background_base: ImageContent | None = None  # 想象补全后的纯背景底图
    current_image: ImageContent | None = None  # 当前轮次的产出图

    # 评审反馈
    latest_feedback: str = ""
    is_perfect: bool = False

    # 迭代控制
    max_iterations: int = 3
    current_iteration: int = 0


# --- 3. 辅助工具函数 ---


def _extract_last_image(message: Message) -> ImageContent | None:
    """从 Message 对象中提取最后一张图片内容."""
    if isinstance(message.content, list):
        return next(
            (
                item
                for item in reversed(message.content)
                if isinstance(item, ImageContent)
            ),
            None,
        )
    return None


def _get_expert_feedback(
    critic_client: OpenAILLM, image_content: ImageContent, system_instruction: str
) -> str:
    """辅助函数: 调用文本大模型对单张图像进行反思性评审."""
    # 构造用户输入，包含图片和引导语
    critic_input = Message(
        role="user",
        content=[image_content, TextContent(text="请根据上述图像提供评审反馈.")],
    )

    try:
        response = critic_client.generate(
            messages=[critic_input],
            system_prompt=system_instruction,
            temperature=1.0,
        )

        if isinstance(response.content, list):
            return "".join(
                [
                    item.text
                    for item in response.content
                    if isinstance(item, TextContent)
                ]
            )
        return str(response.content)
    except Exception as e:
        print(f"❌ 评审模型调用失败: {e}")
        return "评审服务暂时不可用，假设需要继续优化."


# --- 4. 状态处理逻辑 (已整合自动保存功能) ---


def handle_segment(ctx: AgentContext) -> State:
    """分割状态: 确认预分割结果已就绪."""
    print("-> 🎯 [segment] 接收并确认预分割结果...")
    if not ctx.seg_result_txt or "box_2d" not in ctx.seg_result_txt:
        raise ValueError("CTX 中未找到有效的预分割结果文本 (seg_result_txt).")
    print(f"   分割结果确认: {ctx.seg_result_txt.strip()[:50]}...")
    return State.IMAGINE_BACKGROUND


def handle_imagine(ctx: AgentContext) -> State:
    """阶段 1: 背景想象补全."""
    print("\n--- 🟢 阶段 1: 🎃 [imagine_background] 正在生成纯背景底图... ---")

    inpaint_instruction = Message(
        role="user",
        content=[
            TextContent(
                text=(
                    f"参考语义分割结果: {ctx.seg_result_txt}\n"
                    "任务：请完全移除分割结果标注的[main object]，和 [reflective layer]，仅根据剩下的区域生成背景底图。"
                )
            )
        ],
    )

    if not ctx.original_image:
        raise ValueError("CTX 中未找到原始图像 (original_image).")

    inpaint_response = ctx.image_client.edit_image(
        messages=[
            Message(role="user", content=[ctx.original_image]),
            inpaint_instruction,
        ]
    )

    bg_img = _extract_last_image(inpaint_response)
    if bg_img:
        ctx.background_base = bg_img
        # --- 自动保存步骤图 ---
        bg_img.save_to_file("step1_imagined_background.png")
        print("💾 已保存步骤图: step1_imagined_background.png")
        return State.MERGE_BASE

    raise ValueError("背景生成失败")


def handle_merge(ctx: AgentContext) -> State:
    """阶段 2 启动: 融合原图与想象背景."""
    print("\n--- 🔵 阶段 2: 🛠️ [merge_base] 正在进行初始融合... ---")

    merge_prompt = (
        "要求：根据原图(含主体)和生成的反射层想象，负叠加到原图，使得原图的反射层被抵消，剩余部分，也即正常背景, 确保表现正常"
        "确保主体不要动"
    )

    if not ctx.original_image or not ctx.background_base:
        raise ValueError(
            "CTX 中缺少必要的图像数据 (original_image 或 background_base)."
        )

    context_messages = [
        Message(role="user", content=[ctx.original_image]),
        Message(role="user", content=[ctx.background_base]),
        Message(role="user", content=[TextContent(text=merge_prompt)]),
    ]

    edit_response = ctx.image_client.edit_image(messages=context_messages)
    merged_img = _extract_last_image(edit_response)

    if merged_img:
        ctx.current_image = merged_img
        # --- 自动保存步骤图 ---
        merged_img.save_to_file("step2_initial_merge.png")
        print("💾 已保存步骤图: step2_initial_merge.png")
        return State.EVALUATE

    raise ValueError("融合失败")


def handle_edit(ctx: AgentContext) -> State:
    """迭代精修阶段."""
    print(f"\n--- 🔄 第 {ctx.current_iteration + 1} 轮精修: 🎨 [edit] ---")

    refine_prompt = (
        f"上一轮不理想。专家反馈：\n{ctx.latest_feedback}\n"
        "请根据反馈优化。保持原图真实感，消除反射，修复衔接。"
    )

    if not ctx.current_image or not ctx.background_base or not ctx.original_image:
        raise ValueError(
            "CTX 中缺少必要的图像数据 (current_image 或 background_base 或 original_image)."
        )

    context_messages = [
        Message(role="user", content=[ctx.original_image]),
        Message(role="user", content=[ctx.current_image]),
        Message(role="user", content=[ctx.background_base]),
        Message(role="user", content=[TextContent(text=refine_prompt)]),
    ]

    edit_response = ctx.image_client.edit_image(messages=context_messages)
    refined_img = _extract_last_image(edit_response)

    if refined_img:
        ctx.current_image = refined_img
        # --- 自动保存步骤图 (带迭代编号) ---
        file_name = f"step3_refine_iter_{ctx.current_iteration + 1}.png"
        refined_img.save_to_file(file_name)
        print(f"💾 已保存步骤图: {file_name}")

        ctx.current_iteration += 1
        return State.EVALUATE

    return State.DONE


def handle_evaluate(ctx: AgentContext) -> State:
    """专家评审阶段."""
    print("-> 🔍 [evaluate] 正在多维度审查质量...")

    if not ctx.current_image:
        raise ValueError("CTX 中未找到当前图像 (current_image) 以供评审.")

    # 分别获取背景和光学的反馈
    bg_critique = _get_expert_feedback(
        ctx.critic_client,
        ctx.current_image,
        "你是背景融合专家。检查背景衔接是否自然？若完美请回‘无需改进’。",
    )
    light_critique = _get_expert_feedback(
        ctx.critic_client,
        ctx.current_image,
        "你是光学专家。检查玻璃反光和重影是否消除？若完美请回‘无需改进’。",
    )

    ctx.latest_feedback = f"【背景】: {bg_critique}\n【反光】: {light_critique}"
    print(f"专家综合反馈:\n{ctx.latest_feedback}")

    if "无需改进" in bg_critique and "无需改进" in light_critique:
        print("✅ [evaluate] 融合质量达标！")
        return State.DONE

    if ctx.current_iteration >= ctx.max_iterations:
        print("⚠️ [evaluate] 达到最大迭代次数，停止。")
        return State.DONE

    return State.EDIT


# --- 主程序逻辑保持 run_agent 和 main 不变 ---


# --- 5. 状态机主循环 优化版---


def run_agent(ctx: AgentContext) -> Message | None:
    """FSM 主循环."""
    print(f"🚀 FSM Agent 开始视觉任务: {ctx.task_description}")

    # --- 优化点 1: 利用封装的 ImageContent 重新规范化原始图像 ---
    # 既然 original_image 初始存的是 Base64 字符串，
    # 我们可以直接利用 ImageContent(source=..., mime_type=...) 确保属性完整
    if ctx.original_image and isinstance(ctx.original_image.source, str):
        ctx.original_image = ImageContent(
            source=ctx.original_image.source, mime_type="image/jpeg"
        )

    state = State.SEGMENT

    while state != State.DONE:
        print(f"\n--- 🔄 FSM 状态切换: {state.value} ---")
        # 这种映射写法比 if-else 更清晰
        handlers = {
            State.SEGMENT: handle_segment,
            State.IMAGINE_BACKGROUND: handle_imagine,
            State.MERGE_BASE: handle_merge,
            State.EDIT: handle_edit,
            State.EVALUATE: handle_evaluate,
        }

        handler = handlers.get(state)
        if handler:
            state = handler(ctx)
        else:
            raise ValueError(f"未知的状态: {state}")

    print("✨ [DONE] 状态机执行完毕.")

    # --- 优化点 2: 利用 Message 构造函数 ---
    if ctx.current_image:
        # 直接返回封装好的 Message 对象
        return Message(role="model", content=[ctx.current_image])
    return None


# --- 6. 测试入口 ---


def main() -> None:
    """主测试函数: 针对状态机驱动的反射移除任务 (优化封装版)."""
    load_dotenv()

    # 1. 初始化模型
    # 注意：确保 create_llm 内部能够处理你的 Message 对象
    image_llm = create_llm(
        provider="gemini",
        model="gemini-2.5-flash-image",
        timeout=180 * 1000,
    )

    critic_llm = create_llm(
        provider="openai",
        model="openai/kimi-k2.5",
        timeout=90.0,
    )

    # 2. 读取并处理图像
    input_path = Path(
        r"D:\Develop\BananaByte\workspace\Reflection_Removal\Nature\blended\3_19.jpg"
    )
    if not input_path.exists():
        print(f"❌ 找不到图片文件: {input_path}")
        return

    # --- 核心改进：利用 ImageContent 封装加载 ---
    # 如果你需要压缩，建议在 from_file 逻辑外预处理，或者保持 ImageContent 的纯净
    try:
        # 这里直接使用封装好的类方法，它会自动识别 MIME 类型并转 Base64
        initial_img_content = ImageContent.from_file(input_path)
    except ValueError as e:
        print(f"❌ 加载图像失败: {e}")
        return

    # 3. 实例化 Context
    task = "消除原图中所有覆盖在物体表面的白色强光反射和玻璃重影，并让阴影衔接更加柔和."

    # 预分割结果
    seg_rs_txt: str = """
    [
      {"box_2d": [272, 369, 611, 747], "label": "main object potted plant"},
      {"box_2d": [1, 23, 831, 467], "label": "reflective layer glasspane"}
    ]
    """

    ctx = AgentContext(
        task_description=task,
        image_client=image_llm,
        critic_client=critic_llm,
        seg_result_txt=seg_rs_txt,
        original_image=initial_img_content,  # 直接传递封装对象
        max_iterations=3,
    )

    # 4. 运行 Agent
    result_msg = run_agent(ctx)

    # 5. 处理并保存最终结果
    # --- 核心改进：利用 Message 属性和 ImageContent 存盘 ---
    if result_msg and result_msg.images:
        # result_msg.images 是我们刚才在 Message 类里定义的 property
        final_image = result_msg.images[0]
        output_file = "fsm_final_reflection_removed.png"

        try:
            # 调用封装好的保存方法，它会自动处理 Base64 解码和目录创建
            final_image.save_to_file(output_file)
            print(f"✨ 处理完成！最终图像已保存至: {output_file}")
        except Exception as e:
            print(f"❌ 保存文件时出错: {e}")
    else:
        print("❌ Agent 未能成功返回结果图像.")


if __name__ == "__main__":
    main()
