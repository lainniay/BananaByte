import os

from dotenv import load_dotenv

from core import Message, create_llm
from core.schemas import ImageContent, TextContent

# 1. 加载环境变量 (读取 .env 中的 OPENAI_API_KEY 和 OPENAI_API_BASE)
load_dotenv()


def test_openai_image_to_text():
    # 确保测试图片存在
    input_path = "tests/test1.png"
    if not os.path.exists(input_path):
        print(f"❌ 错误：找不到输入图片 {input_path}")
        return

    print("🚀 正在初始化 OpenAI 兼容模型...")
    # 2. 创建 LLM 实例
    # 建议使用支持多模态的模型，如 gpt-4o, gpt-4-turbo 或 claude-3 (通过 litellm)
    llm = create_llm(
        provider="openai",
        model="openai/kimi-k2.5",
        timeout=30.0,  # OpenAI 默认单位是秒
    )

    # 3. 准备图片和提示词
    print(f"读取图片并转换为 Base64: {input_path}")
    # ImageContent.from_file 内部通常会处理读取和编码
    img = ImageContent.from_file(input_path)

    # 构造包含图片和文本的消息
    # 角色设为 user，内容包含图片对象和文本对象
    query = Message(
        role="user",
        content=[
            img,
            TextContent(
                text="请详细描述这张图片的内容，包括主体、背景颜色和整体氛围。"
            ),
        ],
    )

    print("📝 正在请求模型分析图片，请稍候...")
    try:
        # 4. 调用生成接口 (图生文使用 generate)
        # 注意：OpenAILLM.generate 内部会将 Message 转换为 OpenAI 要求的格式
        res = llm.generate(
            messages=[query],
            system_prompt="你是一个专业的图像分析助手。",
            # , temperature=1
        )

        # 5. 处理并打印结果
        if res.content and isinstance(res.content, str):
            print("\n✨ 模型回复内容：")
            print("-" * 30)
            print(res.content)
            print("-" * 30)
        else:
            print("⚠️ 模型未返回有效的文本内容。")

    except Exception as e:
        print(f"❌ 运行出错: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    test_openai_image_to_text()
