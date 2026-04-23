# import base64
# import os

# from dotenv import load_dotenv

# from core import Message, create_llm
# from core.schemas import ImageContent, TextContent

# # 1. 加载环境变量 (读取 .env 中的 GOOGLE_API_KEY)
# load_dotenv()


# def test_image_edit() -> None:
#     """Test image editing with reflection removal using Gemini model.
#     Loads an image, sends it to the Gemini model with a prompt to identify and remove
#     glass reflections, and saves the processed image to disk.
#     """
#     # 确保测试图片存在
#     input_path = r"workspace\Reflection_Removal\Nature\blended\1-2_31.jpg"
#     if not os.path.exists(input_path):
#         print(f"❌ 错误:找不到输入图片 {input_path}")
#         return

#     print("🚀 正在初始化 Gemini 模型...")
#     # 2. 创建 LLM 实例
#     # 注意：确保你的 provider 和 model 名称与 core 中的定义一致
#     llm = create_llm(provider="gemini", model="gemini-2.5-flash-image")

#     # 3. 准备图片和提示词
#     print(f"读取图片: {input_path}")
#     img = ImageContent.from_file(input_path)

#     # 构造消息
#     query = Message(
#         role="user",
#         content=[
#             img,
#             TextContent(
#                 text="请将这张图片上的玻璃反射识别出来并去除,反射的特点可能是不平行,不占据主题,一旦你确定好反射部分,请尽力恢复去除反射后的原场景"
#             ),
#         ],
#     )

#     print("🎨 正在请求模型编辑图片,请稍候...")
#     try:
#         # 4. 调用编辑接口
#         res = llm.edit_image(messages=[query])  # 某些实现可能需要列表格式

#         # 5. 处理并保存结果
#         generated_images = res.images
#         if not generated_images:
#             print("⚠️ 模型未返回任何图片。")
#             return

#         for idx, image_data in enumerate(generated_images):
#             # 获取 base64 字符串
#             b64_str = image_data.source

#             # 自动生成文件名防止覆盖 (如 after_0.jpg, after_1.jpg)
#             output_filename = f"after_{idx}.jpg"

#             with open(output_filename, "wb") as f:
#                 f.write(base64.b64decode(b64_str))

#             print(f"✅ 成功保存处理后的图片至: {output_filename}")

#     except Exception as e:
#         print(f"❌ 运行出错: {e}")


# if __name__ == "__main__":
#     test_image_edit()
