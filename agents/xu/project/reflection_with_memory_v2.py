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
        self, image_client: GeminiLLM, critic_client: OpenAILLM, max_iterations: int = 5
    ) -> None:
        self.image_client = image_client
        self.critic_client = critic_client
        self.memory = VisualMemory()
        self.max_iterations = max_iterations

    def run(
        self,
        task_description: str,
        input_image_base64: str,
        mime_type: str = "image/jpeg",
    ) -> Message | None:
        """运行迭代任务: 生成 -> 专家评审 -> 反思修改."""
        print(f"🚀 开始视觉任务: {task_description}")

        # 1. 构建初始内容并存入记忆
        init_content: list[TextContent | ImageContent] = [
            TextContent(text=f"任务目标: {task_description}"),
            ImageContent(source=input_image_base64, mime_type=mime_type),
        ]

        current_msg = Message(role="user", content=init_content)
        self.memory.add_message(current_msg)

        # --- 迭代闭环开始 ---
        for i in range(self.max_iterations):
            print(f"\n--- 🔄 第 {i + 1} 轮迭代 ---")

            # A. 执行阶段 (Gemini)
            print("-> 🎨 正在调用 Gemini 优化图像...")
            edit_instruction = Message(
                role="user",
                content=[
                    TextContent(
                        text=f"请根据任务要求和反馈执行编辑: {task_description}"
                    )
                ],
            )

            # --- 记忆传输优化 ---
            # 为了防止多次迭代后消息列表中包含太多图片导致 Protocol Error，
            # 这里仅传递【最初的 Message】和【包含最新图+反馈的当前 Message】。
            context_messages = [self.memory.history[0], current_msg, edit_instruction]

            edit_response = self.image_client.edit_image(messages=context_messages)
            self.memory.add_message(edit_response)

            # B. 专家评审阶段
            print(f"-> 🔍 正在调用 {self.critic_client.model} 进行多维度专家审查...")

            bg_critique = self._get_expert_feedback(
                edit_response,
                "你是一位背景融合审查专家。检查背景衔接是否自然，是否有不自然的反射光斑。请用方位词描述问题，若完美请回‘无需改进’。",
            )

            obj_critique = self._get_expert_feedback(
                edit_response,
                "你是一位物体逻辑专家。检查图像中是否有物体形态扭曲或因移除反射导致的结构错误。请用方位词指出位置，若完美请回‘无需改进’。",
            )

            light_critique = self._get_expert_feedback(
                edit_response,
                "你是一位光学专家。重点检查反光是否彻底移除，剩余阴影是否符合物理规律。指出不合理的反射光残余方位。若完美请回‘无需改进’。",
            )

            feedback_summary = (
                f"【背景反馈】: {bg_critique}\n"
                f"【物体反馈】: {obj_critique}\n"
                f"【光影反馈】: {light_critique}"
            )
            print(f"专家综合反馈:\n{feedback_summary}")

            # C. 判定终止条件
            if all(
                "无需改进" in c for c in [bg_critique, obj_critique, light_critique]
            ):
                print("✅ 所有维度专家达成一致，图像质量达标。")
                return edit_response

            # D. 准备下一轮优化指令
            print("-> 🛠️ 汇总反馈，准备下一轮精修...")
            next_step_content: list[TextContent | ImageContent] = []

            refine_text = (
                f"专家评审指出以下问题，请在下一轮修复中重点解决：\n{feedback_summary}\n"
                "请保持图像其他部分不变，仅针对上述方位进行优化。"
            )
            next_step_content.append(TextContent(text=refine_text))

            # 核心优化：只取最后生成的这一张图片带入下一轮，防止 Base64 堆叠
            if isinstance(edit_response.content, list):
                last_img = next(
                    (
                        item
                        for item in reversed(edit_response.content)
                        if isinstance(item, ImageContent)
                    ),
                    None,
                )
                if last_img:
                    next_step_content.append(last_img)

            current_msg = Message(role="user", content=next_step_content)
            self.memory.add_message(current_msg)

        print("⚠️ 达到最大迭代次数，返回当前最优结果。")
        return self.memory.get_last_image_message()

    def _get_expert_feedback(self, image_msg: Message, system_instruction: str) -> str:
        """调用文本大模型对图像进行反思性评审."""
        only_images: list[TextContent | ImageContent] = [
            item for item in image_msg.content if isinstance(item, ImageContent)
        ]

        critic_input = Message(role="user", content=only_images)
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

    # 3. 读取并【压缩】待处理的本地图片
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
        image_client=image_llm, critic_client=critic_llm, max_iterations=5
    )

    task = "移除图片背景中不自然的反射光斑，特别是覆盖在物体表面的白色强光，并让阴影衔接更加柔和。"

    result_msg = agent.run(
        task_description=task, input_image_base64=encoded_string, mime_type="image/jpeg"
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
            print("⚠️ 结果消息中未包含有效图像内容。")
    else:
        print("❌ Agent 未能成功返回结果。")


if __name__ == "__main__":
    main()
