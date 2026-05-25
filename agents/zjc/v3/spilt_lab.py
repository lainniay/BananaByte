from pathlib import Path

import cv2
import numpy as np

from core import ImageContent


def lab_to_bgr(
    l_channel: np.ndarray,
    a_channel: np.ndarray,
    b_channel: np.ndarray,
) -> np.ndarray:
    """将 LAB 三通道合成为 BGR 图像."""
    return cv2.cvtColor(
        cv2.merge([l_channel, a_channel, b_channel]),
        cv2.COLOR_LAB2BGR,
    )


def read_image(path: Path, flags: int) -> np.ndarray:
    """读取图片并在失败时抛出明确错误."""
    img = cv2.imread(str(path), flags)
    if img is None:
        raise ValueError(f"图片读取失败: {path}")
    return img


def resize_channel(channel: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """将单通道图片缩放到指定宽高."""
    if channel.shape[:2] == (size[1], size[0]):
        return channel
    return cv2.resize(channel, size, interpolation=cv2.INTER_CUBIC)


def extract_lab_channel(
    path: Path, channel_index: int, size: tuple[int, int]
) -> np.ndarray:
    """从图片中提取 LAB 指定通道并缩放."""
    img = read_image(path, cv2.IMREAD_UNCHANGED)

    if img.ndim == 2:
        channel = img
    else:
        if img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        channel = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)[:, :, channel_index]

    return resize_channel(channel, size)


def compose_lab_image(
    l_path: Path,
    a_path: Path,
    b_path: Path,
    output_path: Path,
    original_l_path: Path | None = None,
) -> None:
    """用给定 LAB 通道合成图片并保存."""
    l_channel = read_image(l_path, cv2.IMREAD_GRAYSCALE)
    height, width = l_channel.shape[:2]
    size = (width, height)

    if original_l_path is not None:
        original_l_channel = read_image(original_l_path, cv2.IMREAD_GRAYSCALE)
        original_l_channel = resize_channel(original_l_channel, size)
        l_channel = cv2.addWeighted(original_l_channel, 0.5, l_channel, 0.5, 0)

    a_channel = extract_lab_channel(a_path, 1, size)
    b_channel = extract_lab_channel(b_path, 2, size)

    new_img = lab_to_bgr(l_channel, a_channel, b_channel)
    if not cv2.imwrite(str(output_path), new_img):
        raise ValueError(f"图片保存失败: {output_path}")


def save_lab_channels(image: ImageContent, output_dir: Path = Path(".")) -> None:  # noqa : D103
    # 从二进制图片数据解码
    img_array = np.frombuffer(image.source, dtype=np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

    if img is None:
        raise ValueError("图片解码失败")

    # OpenCV 解码为 BGR，需要转换为 LAB 颜色空间
    lab_img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab_img)

    neutral_a = np.full_like(a_channel, 128)
    neutral_b = np.full_like(b_channel, 128)
    bright_l = np.full_like(l_channel, 180)

    # L 没有色彩方向，使用灰度图展示亮度变化
    l_image = l_channel
    # a 轴：绿色 <-> 红/洋红
    a_image = lab_to_bgr(bright_l, a_channel, neutral_b)
    # b 轴：蓝色 <-> 黄色
    b_image = lab_to_bgr(bright_l, neutral_a, b_channel)

    cv2.imwrite(str(output_dir / "l_channel.png"), l_image)
    cv2.imwrite(str(output_dir / "a_channel.png"), a_image)
    cv2.imwrite(str(output_dir / "b_channel.png"), b_image)


if __name__ == "__main__":
    input_path = Path("../../../workspace/Severe/0/in.jpg")
    if input_path.exists():
        img = ImageContent.from_file(str(input_path))
        save_lab_channels(img)

    new_l_path = Path("new_l.png")
    original_l_path = Path("l_channel.png")
    a_path = Path("a_channel.png")
    b_path = Path("b_channel.png")
    if (
        new_l_path.exists()
        and original_l_path.exists()
        and a_path.exists()
        and b_path.exists()
    ):
        compose_lab_image(
            new_l_path,
            a_path,
            b_path,
            Path("new_lab_2.png"),
            original_l_path,
        )
