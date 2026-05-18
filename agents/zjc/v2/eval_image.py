import os

from agents.zjc.calculate import calculate_uciqe, calculate_uiqm
from core.schemas import ImageContent


def evaluate_single_image(image_path: str) -> None:
    """读取单张图片并计算水下图像质量评价指标."""
    if not os.path.exists(image_path):
        print(f"Error: 文件不存在 -> {image_path}")
        return

    img = ImageContent.from_file(image_path)

    uciqe_val = calculate_uciqe(img)
    uiqm_val = calculate_uiqm(img)

    # 输出结果
    print("-" * 30)
    print(f"图片路径: {image_path}")
    print(f"UCIQE: {uciqe_val:.6f}")
    print(f"UIQM:  {uiqm_val:.6f}")
    print("-" * 30)


if __name__ == "__main__":
    # 在此处硬编码输入图片路径
    # 您可以根据需要修改此处的路径
    img1 = "../../workspace/U45_32/nb.png"

    evaluate_single_image(img1)
