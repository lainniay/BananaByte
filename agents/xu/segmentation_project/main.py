import json
import os

from dotenv import load_dotenv
from google import genai
from google.genai import types
from google.genai.types import HttpOptions
from PIL import Image, ImageDraw, ImageFont

# 1. 加载配置
load_dotenv()

GEMINI_TIMEOUT_MS = 60 * 1000
# 建议在 .env 中配置 GEMINI_API_KEY 和 GEMINI_API_BASE
client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY"),
    http_options=HttpOptions(
        base_url=os.getenv("GEMINI_API_BASE"), timeout=GEMINI_TIMEOUT_MS
    ),
)


def draw_and_save_result(
    image_path: str,
    model_output_text: str,
    output_filename: str = "detection_result.jpg",
) -> None:
    """解析模型输出的 JSON 并将检测框和标签绘制在图片上."""
    # 加载原始图片
    try:
        img = Image.open(image_path)
    except Exception as e:
        print(f"无法打开图片: {e}")
        return

    # 统一转换为 RGBA 方便绘图，最后保存时再转回 RGB
    img = img.convert("RGBA")
    img_w, img_h = img.size
    draw = ImageDraw.Draw(img)

    # 尝试加载字体
    try:
        # Windows 常用字体路径，Linux/Mac 可能需要换成 'DejaVuSans.ttf' 或类似
        font = ImageFont.truetype("arial.ttf", 18)
    except Exception:
        font = ImageFont.load_default()

    # 解析 JSON 数据
    try:
        # 清洗可能存在的 Markdown 标签
        clean_json = model_output_text.replace("```json", "").replace("```", "").strip()
        data = json.loads(clean_json)
    except Exception as e:
        print(f"JSON 解析失败: {e}")
        print("原始输出内容:", model_output_text)
        return

    # 预定义颜色池
    colors = ["#FF3333", "#33FF33", "#3333FF", "#FFFF33", "#FF33FF", "#33FFFF"]

    for i, item in enumerate(data):
        box = item.get("box_2d")
        label = item.get("label", "Object")
        color = colors[i % len(colors)]

        if box and len(box) == 4:
            # Gemini 坐标转换: [y0, x0, y1, x1] (0-1000) -> 像素坐标
            ymin, xmin, ymax, xmax = box
            left = xmin * img_w / 1000
            top = ymin * img_h / 1000
            right = xmax * img_w / 1000
            bottom = ymax * img_h / 1000

            # --- 绘制逻辑 ---
            # 1. 画外框
            draw.rectangle([left, top, right, bottom], outline=color, width=3)

            # 2. 准备标签文字
            # 计算文字背景区域 (使用 textbbox)
            text_pos = (left + 8, top + 8)  # 标在框内左上角，偏移 8 像素
            text_box = draw.textbbox(text_pos, label, font=font)

            # 画文字背景小矩形（稍微扩充一点点边距）
            bg_rect = [
                text_box[0] - 4,
                text_box[1] - 2,
                text_box[2] + 4,
                text_box[3] + 2,
            ]
            draw.rectangle(bg_rect, fill=color)

            # 3. 写字 (白色)
            draw.text(text_pos, label, fill="white", font=font)

    # --- 保存逻辑 ---
    # JPEG 不支持 RGBA，必须转回 RGB
    final_img = img.convert("RGB")
    final_img.show()
    final_img.save(output_filename, quality=95)
    print(f"\n处理完成, 结果已保存至: {output_filename}")


if __name__ == "__main__":
    # 请确保路径正确
    input_image = r"agents\xu\segmentation_proj\image.png"

    if not os.path.exists(input_image):
        print(f"错误: 找不到文件 {input_image}")
    else:
        # 构造 Prompt：明确要求 box_2d 和 label，不要 mask
        query = "Detect two parts: main object and the possible reflective layer(possibly a quadrilateral area and takes only part of the image)."
        prompt = (
            f"{query}. Output a JSON list of objects. Each object must contain: "
            "'box_2d' [ymin, xmin, ymax, xmax] (normalized 0-1000) and 'label'. "
            "Do not include any mask data."
        )

        print("正在请求 Gemini 2.5 Flash (推理中)...")

        # 打开并缩放图片（减小传输压力）
        raw_im = Image.open(input_image)
        raw_im.thumbnail((1024, 1024), Image.Resampling.LANCZOS)

        try:
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=[prompt, raw_im],
                config=types.GenerateContentConfig(
                    temperature=0.2,  # 低温度使坐标更精确
                    response_mime_type="application/json",
                ),
            )

            print("模型识别到以下物体:")
            if not response.text:
                print("未检测到任何物体.")
            else:
                print(response.text)
                # 调用绘图与保存
                draw_and_save_result(input_image, response.text)

        except Exception as e:
            print(f"模型调用或处理发生错误: {e}")
