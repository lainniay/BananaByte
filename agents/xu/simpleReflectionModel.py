import base64
import os

from dotenv import load_dotenv

# 确保导入了核心组件
from core import GeminiLLM, ImageContent, Message, OpenAILLM, TextContent, create_llm

# --- 模块 1: 视觉记忆模块 ---


class VisualMemory:
    """Store Message objects containing images and reflection text."""

    def __init__(self) -> None:
        self.history: list[Message] = []

    def add_message(self, message: Message) -> None:
        """Add a message to the history.

        Args:
            message: The Message object to add to the history.
        """
        self.history.append(message)

    def get_last_image_message(self) -> Message | None:
        """获取最后一条包含图像的消息."""
        for msg in reversed(self.history):
            if isinstance(msg.content, list) and any(
                item.type == "image" for item in msg.content
            ):
                return msg
        return None


# --- 模块 2: 视觉反思智能体 ---


class VisualReflectionAgent:
    """Agent for iterative visual reflection and image editing tasks.

    Uses GeminiLLM for image editing and OpenAILLM for expert feedback on
    background, objects, and lighting aspects of images.
    """

    def __init__(
        self, image_client: GeminiLLM, critic_client: OpenAILLM, max_iterations: int = 5
    ) -> None:
        self.image_client = image_client  # 专门负责 edit_image
        self.critic_client = critic_client  # 专门负责 generate (VQA 反思)
        self.memory = VisualMemory()
        self.max_iterations = max_iterations

    def run(
        self,
        task_description: str,
        input_image_base64: str,
        mime_type: str = "image/jpeg",
    ) -> Message | None:
        """Run iterative visual reflection and editing task.

        Args:
            task_description: Description of the visual editing task.
            input_image_base64: Base64-encoded input image.
            mime_type: MIME type of the image (default: "image/jpeg").

        Returns:
            Final message with edited image, or None if task fails.
        """
        print(f"🚀 开始视觉任务: {task_description}")

        # 1. 显式构建初始内容列表
        init_content: list[TextContent | ImageContent] = [
            TextContent(text=f"任务: {task_description}"),
            ImageContent(source=input_image_base64, mime_type=mime_type),
        ]

        current_msg = Message(role="user", content=init_content)
        self.memory.add_message(current_msg)

        # --- 迭代开始 ---
        for i in range(self.max_iterations):
            print(f"\n--- 🔄 第 {i + 1} 轮迭代 ---")

            # A. 执行/编辑阶段 (Gemini)
            print("-> 🎨 正在调用 Gemini 优化图像...")
            edit_instruction = Message(
                role="user",
                content=[TextContent(text=f"根据任务要求执行编辑: {task_description}")],
            )

            edit_response = self.image_client.edit_image(
                messages=[current_msg, edit_instruction]
            )
            self.memory.add_message(edit_response)

            # B. 解耦反思阶段 (Kimi / OpenAI 接口)
            print(f"-> 🔍 正在调用 {self.critic_client.model} 进行专家审查...")

            bg_critique = self._get_expert_feedback(
                edit_response,
                "你是一位背景审查专家。检查背景衔接是否自然，是否有不该出现的反射物。请用方位词描述问题，若完美请回‘无需改进’。",
            )

            obj_critique = self._get_expert_feedback(
                edit_response,
                "你是一位物体逻辑专家。检查是否有物体形态扭柱、幻觉或结构错误。请用方位词描述位置，若完美请回‘无需改进’。",
            )

            light_critique = self._get_expert_feedback(
                edit_response,
                "你是一位光学专家。检查反光来源和阴影是否符合物理规律。指出不合理的反射光方位。若完美请回‘无需改进’。",
            )

            feedback_summary = (
                f"背景: {bg_critique}\n物体: {obj_critique}\n光影: {light_critique}"
            )
            print(f"专家综合反馈:\n{feedback_summary}")

            # C. 判定终止条件
            if (
                "无需改进" in bg_critique
                and "无需改进" in obj_critique
                and "无需改进" in light_critique
            ):
                print("✅ 专家达成一致，图像质量达标。")
                return edit_response

            # D. 准备下一轮优化
            print("-> 🛠️ 准备下一轮优化指令...")
            next_step_content: list[TextContent | ImageContent] = []

            refine_text = (
                f"请根据以下评审反馈修复图像，特别是提到的方位点：\n{feedback_summary}"
            )
            next_step_content.append(TextContent(text=refine_text))

            if isinstance(edit_response.content, list):
                for item in edit_response.content:
                    if isinstance(item, ImageContent):
                        next_step_content.append(item)

            current_msg = Message(role="user", content=next_step_content)
            self.memory.add_message(current_msg)

        print("达到最大迭代次数，返回最终结果。")
        return self.memory.get_last_image_message()

    def _get_expert_feedback(self, image_msg: Message, system_instruction: str) -> str:
        """调用文本模型进行图像反思."""
        only_images: list[TextContent | ImageContent] = [
            item for item in image_msg.content if isinstance(item, ImageContent)
        ]
        critic_input = Message(role="user", content=only_images)

        response = self.critic_client.generate(
            messages=[critic_input],
            system_prompt=system_instruction,
            temperature=1,
        )

        if isinstance(response.content, list):
            text_parts = [
                item.text for item in response.content if isinstance(item, TextContent)
            ]
            return "".join(text_parts)

        return str(response.content)


# Kimi 评审测试，跳过 Gemini 调用，直接构造消息进行评审
# def main():
#     load_dotenv()

#     # 1. 暂时不初始化 Gemini，或者传入 None
#     # image_llm = create_llm(...)

#     # 2. 初始化 Kimi 评审模型
#     critic_llm = create_llm(
#         provider="openai",
#         model="openai/kimi-k2.5",
#         timeout=60.0,
#     )

#     # 3. 读取本地图片
#     input_path = (
#         r"D:\Develop\BananaByte\workspace\Reflection_Removal\Nature\blended\1-2_31.jpg"
#     )
#     if not os.path.exists(input_path):
#         print(f"❌ 找不到图片文件: {input_path}")
#         return

#     with open(input_path, "rb") as image_file:
#         encoded_string = base64.b64encode(image_file.read()).decode("utf-8")

#     # 4. 核心步骤：手动模拟一个来自 Gemini 的消息对象
#     # 这样你就跳过了真正的 API 调用，直接进入评审环节
#     mock_gemini_msg = Message(
#         role="user",
#         content=[
#             TextContent(text="这是模拟的生成结果。"),
#             ImageContent(source=encoded_string, mime_type="image/jpeg"),
#         ],
#     )

#     # 5. 实例化 Agent
#     # 注意：image_client 传入 None，因为我们暂时不用它
#     agent = VisualReflectionAgent(
#         image_client=None,  # type: ignore
#         critic_client=critic_llm,
#     )

#     print("🚀 跳过图像生成，直接开始 Kimi 专家评审测试...")

#     # 6. 直接调用评审逻辑
#     try:
#         bg_critique = agent._get_expert_feedback(
#             mock_gemini_msg,
#             "你是一位背景审查专家。检查背景衔接是否自然，是否有不该出现的反射物。请用方位词描述问题，若完美请回‘无需改进’。",
#         )
#         print(f"\n🔍 Kimi 背景评审结果:\n{bg_critique}")

#     except Exception as e:
#         print(f"❌ 评审过程出错: {e}")


def main() -> None:
    """Initialize LLMs, load image, run VisualReflectionAgent, and save result, test for reflectionmodel."""
    load_dotenv()

    # 1. 初始化生成模型 (Gemini)
    image_llm = create_llm(
        provider="gemini", model="gemini-2.5-flash-image", timeout=180 * 1000
    )

    # 2. 初始化评审模型 (Kimi / OpenAI 接口)
    critic_llm = create_llm(
        provider="openai",
        model="openai/kimi-k2.5",
        timeout=90.0,
    )

    # 3. 读取本地图片
    input_path = (
        r"D:\Develop\BananaByte\workspace\Reflection_Removal\Nature\blended\3_19.jpg"
    )
    if not os.path.exists(input_path):
        print(f"❌ 找不到图片文件: {input_path}")
        return

    with open(input_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode("utf-8")

    # 4. 运行 Agent
    agent = VisualReflectionAgent(
        image_client=image_llm, critic_client=critic_llm, max_iterations=2
    )

    task = "移除图片背景中不自然的反射光斑，并让阴影衔接更加柔和。"

    result_msg = agent.run(task_description=task, input_image_base64=encoded_string)

    # 5. 保存结果
    if result_msg and isinstance(result_msg.content, list):
        for idx, item in enumerate(result_msg.content):
            if isinstance(item, ImageContent):
                file_name = f"output_result_3-19_{idx}.png"
                with open(file_name, "wb") as f:
                    f.write(base64.b64decode(item.source))
                print(f"✨ 图像已保存至: {file_name}")


if __name__ == "__main__":
    main()
