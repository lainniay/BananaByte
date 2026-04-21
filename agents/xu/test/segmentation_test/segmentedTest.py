import base64
import io
import os

from dotenv import load_dotenv
from PIL import Image  # 确保环境已安装: pip install pillow

# 确保导入了核心组件
from core import GeminiLLM, ImageContent, Message, OpenAILLM, TextContent, create_llm

# --- 辅助模块：图像预处理 ---


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


# --- 模块 1: 视觉记忆模块 ---


class VisualMemory:
    """存储包含图像和反思文本的 Message 对象历史."""

    def __init__(self) -> None:
        self.history: list[Message] = []

    def add_message(self, message: Message) -> None:
        """将消息添加到历史记录中."""
        self.history.append(message)

    def get_last_image_message(self) -> Message | None:
        """获取最后一条包含图像的消息."""
        for msg in reversed(self.history):
            if isinstance(msg.content, list) and any(
                isinstance(item, ImageContent) for item in msg.content
            ):
                return msg
        return None


# --- 模块 2: 视觉反思智能体 ---


class VisualReflectionAgent:
    """迭代式视觉反思与图像编辑智能体."""

    def __init__(
        self,
        image_client: GeminiLLM,
        # segment_client 不需要，直接传入结果
        critic_client: OpenAILLM,
        max_iterations: int = 5,
    ) -> None:
        self.image_client = image_client
        self.critic_client = critic_client
        self.memory = VisualMemory()
        self.max_iterations = max_iterations

    def run(
        self,
        task_description: str = "",
        # seg_prompt 不需要
        # 接收预先得到的分割结果文本
        seg_result_txt: str = "",
        input_image_base64: str = "",
        mime_type: str = "image/jpeg",
    ) -> Message | None:
        """运行迭代任务."""
        print(f"🚀 开始视觉任务: {task_description}")

        # 构建初始内容并存入记忆
        init_content: list[TextContent | ImageContent] = [
            TextContent(text=f"任务目标: {task_description}"),
            ImageContent(source=input_image_base64, mime_type=mime_type),
        ]

        current_msg = Message(role="user", content=init_content)
        self.memory.add_message(current_msg)

        # --- 流程配置 ---
        # seg_result_txt 包含你预先提供的：主体(potted plant) 和 反射层(reflective layer/glass pane)

        # 阶段 1：背景想象补全 (Imagination Phase)
        print("\n--- 🟢 阶段 1: 背景想象补全 ---")
        inpaint_instruction = Message(
            role="user",
            content=[
                TextContent(
                    text=(
                        f"参考语义分割结果: {seg_result_txt}\n"
                        "第一步任务：请完全移除图像中的【主体】和【反射层】区域.\n"
                        "利用周围环境的纹理进行想象补全，生成一张纯粹的背景底图，不要保留任何盆栽或玻璃反光的痕迹."
                    )
                )
            ],
        )

        # 只传第一张原图和补全指令
        inpaint_response = self.image_client.edit_image(
            messages=[self.memory.history[0], inpaint_instruction]
        )
        self.memory.add_message(inpaint_response)

        # 获取补全后的背景图
        # 获取补全后的背景图
        background_img = self._extract_last_image(inpaint_response)

        # --- FIX STARTS HERE ---
        if background_img is None:
            print("❌ 错误：未能从响应中提取到背景图像.")
            return None

        # 现在 background_img 确定不是 None，类型检查通过
        current_msg = Message(role="user", content=[background_img])
        # --- FIX ENDS HERE ---

        # 阶段 2：迭代融合与专家评审 (Refinement Loop)
        print("\n--- 🔵 阶段 2: 图像对比融合与反光消除 ---")
        for i in range(self.max_iterations):
            print(f"\n--- 🔄 第 {i + 1} 轮迭代 ---")

            # A. 执行阶段 (对比融合)
            print("-> 🎨 正在执行背景融合与反光剥离...")

            # 构造融合指令
            # 逻辑：将上一轮的结果（或背景图）与最原始图对比，保留原图真实感但剥离反射
            merge_text = (
                "任务：对比提供的背景底图与原图.\n"
                "要求：在保持原图背景结构的基础上，参考背景底图的纯净度，"
                "彻底消除原图中的反射层残留（光斑、虚假重影）.\n"
                "请确保最终图像看起来像是直接拍摄背景，没有玻璃阻隔."
            )

            edit_instruction = Message(
                role="user",
                content=[TextContent(text=merge_text)],
            )

            # 核心上下文：原图 (history[0]) + 辅助背景图/上一轮结果 (current_msg) + 指令
            context_messages = [self.memory.history[0], current_msg, edit_instruction]
            edit_response = self.image_client.edit_image(messages=context_messages)
            self.memory.add_message(edit_response)

            # B. 专家评审阶段
            print(f"-> 🔍 正在调用 {self.critic_client.model} 进行多维度审查...")

            bg_critique = self._get_expert_feedback(
                edit_response,
                "检查背景衔接是否自然，是否有由于移除反射导致的背景空洞或模糊？若完美请回‘无需改进’.",
            )

            light_critique = self._get_expert_feedback(
                edit_response,
                "重点检查反射层（玻璃反光）是否还存在残余？阴影是否符合物理规律？若完全消除请回‘无需改进’.",
            )

            feedback_summary = (
                f"【背景融合反馈】: {bg_critique}\n【反光剥离反馈】: {light_critique}"
            )
            print(f"专家综合反馈:\n{feedback_summary}")

            # C. 判定终止条件
            if all("无需改进" in c for c in [bg_critique, light_critique]):
                print("✅ 融合质量达标，已成功消除反射层并补全背景.")
                return edit_response

            # D. 准备下一轮优化
            print("-> 🛠️ 质量不理想，准备重新融合...")
            last_img = self._extract_last_image(edit_response)

            # --- FIX STARTS HERE ---
            if last_img is None:
                print("⚠️ 无法获取当前轮次的图片，终止迭代.")
                break
            # --- FIX ENDS HERE ---

            refine_text = (
                f"上一轮融合不理想.专家指出：\n{feedback_summary}\n"
                "请重新比对背景补全图，加大消除反光的力度，修复上述方位的问题."
            )

            current_msg = Message(
                role="user", content=[TextContent(text=refine_text), last_img]
            )
            self.memory.add_message(current_msg)

        print("⚠️ 达到最大迭代次数，返回当前最优结果.")
        return self.memory.get_last_image_message()

    # 辅助工具函数（需在类中定义）
    def _extract_last_image(self, response: Message) -> ImageContent | None:
        """从 Message 对象中提取最后一张图片内容."""
        if isinstance(response.content, list):
            return next(
                (
                    item
                    for item in reversed(response.content)
                    if isinstance(item, ImageContent)
                ),
                None,
            )
        return None

    def _get_expert_feedback(self, image_msg: Message, system_instruction: str) -> str:
        """调用文本大模型对图像进行反思性评审."""
        # 1. 尝试提取图片
        only_images = [
            item for item in image_msg.content if isinstance(item, ImageContent)
        ]

        # 2. 构造用户输入
        # 如果模型支持多模态，我们可以传图片；
        # 如果是纯文本模型（如 Kimi 部分版本），我们至少要传一个文本说明，不能留空.
        user_content = []
        if only_images:
            user_content.extend(only_images)
            # 添加一段引导语，防止某些 API 在只有图片没有文字时报错
            user_content.append(
                TextContent(text="请根据上述图像和系统指令提供评审反馈.")
            )
        else:
            # 彻底没有图片时的保底
            user_content.append(TextContent(text="请执行评审任务."))

        critic_input = Message(role="user", content=user_content)

        try:
            response = self.critic_client.generate(
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
            return "评审服务暂时不可用，请继续尝试优化."


# --- 模块 3: 测试入口 ---


def main() -> None:
    """主测试函数: 针对反射移除任务进行迭代优化测试."""
    load_dotenv()

    # 1. 初始化执行模型
    image_llm = create_llm(
        provider="gemini",
        model="gemini-2.5-flash-image",
        timeout=180 * 1000,
    )

    # 2. 初始化评审模型
    critic_llm = create_llm(
        provider="openai",
        model="openai/kimi-k2.5",
        timeout=90.0,
    )

    # 3. 读取并待处理的本地图片
    input_path = (
        r"D:\Develop\BananaByte\workspace\Reflection_Removal\Nature\blended\3_19.jpg"
    )
    if not os.path.exists(input_path):
        print(f"❌ 找不到图片文件: {input_path}")
        return

    # 使用压缩函数替代直接读取，限制长边 1024px
    encoded_string = get_compressed_base64(input_path, max_size=1024)

    # 4. 实例化并运行 Agent
    agent = VisualReflectionAgent(
        image_client=image_llm, critic_client=critic_llm, max_iterations=2
    )

    task = "移除图片背景中不自然的反射光斑，特别是覆盖在物体表面的白色强光，并让阴影衔接更加柔和."

    # 你预先得到的分割结果
    seg_rs_txt: str = """
[
  {"box_2d": [272, 369, 611, 747], "label": "the main object, a green potted plant in a white pot"},
  {"box_2d": [1, 23, 831, 467], "label": "the reflective layer, a glass pane showing a reflection of an office room"}
]
"""

    result_msg = agent.run(
        task_description=task,
        seg_result_txt=seg_rs_txt,  # 将预分割结果传入
        input_image_base64=encoded_string,
        mime_type="image/jpeg",
    )

    # 5. 处理并保存最终结果
    if result_msg and isinstance(result_msg.content, list):
        save_count = 0
        for item in result_msg.content:
            if isinstance(item, ImageContent):
                save_count += 1
                file_name = f"final_reflection_removed_{save_count}.png"
                with open(file_name, "wb") as f:
                    f.write(base64.b64decode(item.source))
                print(f"✨ 处理完成！最终图像已保存至: {file_name}")

        if save_count == 0:
            print("⚠️ 结果消息中未包含有效图像内容.")
    else:
        print("❌ Agent 未能成功返回结果.")


if __name__ == "__main__":
    main()
