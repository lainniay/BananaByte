import json

from PIL import Image, ImageDraw, ImageFont


def draw_and_save_result(
    image_path: str,
    model_output_json: str,
    output_filename: str = "output_detected.jpg",
) -> None:
    """解析 Gemini 格式的 JSON 并在图上绘制检测框."""
    """坐标格式假设为 [ymin, xmin, ymax, xmax] (0-1000 归一化)."""
    # 1. 加载图片
    try:
        img = Image.open(image_path).convert("RGBA")
    except Exception as e:
        print(f"无法打开图片: {e}")
        return

    img_w, img_h = img.size
    draw = ImageDraw.Draw(img)

    # 2. 尝试加载字体 (Windows 默认 Arial, 失败则使用默认)
    try:
        font = ImageFont.truetype("arial.ttf", 20)
    except Exception:
        font = ImageFont.load_default()

    # 3. 解析数据
    try:
        # 清理可能存在的 Markdown 代码块标签
        clean_json = model_output_json.replace("```json", "").replace("```", "").strip()
        data = json.loads(clean_json)
    except Exception as e:
        print(f"JSON 解析失败: {e}")
        return

    # 颜色池
    colors = ["#FF3333", "#33FF33", "#3333FF", "#FFFF33", "#FF33FF", "#33FFFF"]

    # 4. 绘图循环
    for i, item in enumerate(data):
        box = item.get("box_2d")
        label = item.get("label", "Object")
        color = colors[i % len(colors)]

        if box and len(box) == 4:
            # 坐标转换: [y0, x0, y1, x1] -> 像素坐标
            ymin, xmin, ymax, xmax = box
            left = xmin * img_w / 1000
            top = ymin * img_h / 1000
            right = xmax * img_w / 1000
            bottom = ymax * img_h / 1000

            # 画框
            draw.rectangle([left, top, right, bottom], outline=color, width=4)

            # 画标签背景
            text_pos = (left + 5, top + 5)
            # 兼容旧版本 PIL 的 textbbox
            if hasattr(draw, "textbbox"):
                t_box = draw.textbbox(text_pos, label, font=font)
                draw.rectangle(
                    [t_box[0] - 2, t_box[1] - 2, t_box[2] + 2, t_box[3] + 2], fill=color
                )

            # 写字
            draw.text(text_pos, label, fill="white", font=font)

    # 5. 保存结果
    final_img = img.convert("RGB")
    final_img.save(output_filename, quality=95)
    print(f"绘制完成！结果保存至: {output_filename}")
    final_img.show()


# --- 快速测试入口 ---
if __name__ == "__main__":
    # 模拟图片路径（请确保你本地有这张图，或者换成一个存在的路径）
    test_image_path = (
        r"D:\Develop\BananaByte\workspace\Reflection_Removal\Nature\blended\3_19.jpg"
    )

    # 模拟 Gemini 返回的典型 JSON 字符串
    test_json = """
    [
      {"box_2d": [272, 369, 611, 747], "label": "main object potted plant"},
      {"box_2d": [1, 23, 831, 467], "label": "reflective layer glasspane"}
    ]
    """

    # 运行（如果 test.jpg 不存在会报错，请替换为你的文件名）
    import os

    if os.path.exists(test_image_path):
        draw_and_save_result(test_image_path, test_json)
    else:
        print(f"请先准备一张名为 {test_image_path} 的图片放在当前目录下进行测试。")
