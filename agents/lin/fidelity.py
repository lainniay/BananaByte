"""保真度锚点 - 用低频对齐抑制幻觉."""

import numpy as np
from PIL import Image, ImageFilter


def image_to_array(image: Image.Image) -> np.ndarray:
    """PIL Image → float32 numpy 数组.

    Args:
        image: PIL Image（自动转 RGB）.

    Returns:
        (H, W, 3) float32 数组，范围 [0, 255].
    """
    return np.array(image.convert("RGB"), dtype=np.float32)


def array_to_image(array: np.ndarray) -> Image.Image:
    """float32 numpy 数组 → PIL Image.

    Args:
        array: (H, W, 3) float32 数组.

    Returns:
        RGB 模式 PIL Image.
    """
    return Image.fromarray(np.clip(array, 0, 255).astype(np.uint8))


def low_pass(image_arr: np.ndarray, sigma: float = 2.0) -> np.ndarray:
    """高斯低通滤波，提取低频成分.

    Args:
        image_arr: float32 图像数组.
        sigma: 高斯核标准差.

    Returns:
        模糊后的数组.
    """
    img = array_to_image(image_arr)
    blurred = img.filter(ImageFilter.GaussianBlur(radius=sigma))
    return image_to_array(blurred)


def fidelity_blend(
    original: Image.Image,
    sr_output: Image.Image,
    alpha: float = 0.3,
    sigma: float = 2.0,
) -> Image.Image:
    """保真度锚点混合.

    核心思路：
    1. 将原始低分图上采样到 SR 输出同尺寸
    2. 分别提取两者的低频成分 (L_orig, L_sr)
    3. delta = L_orig - L_sr（结构偏移量）
    4. result = SR_output + alpha * delta

    这样 SR 输出的高频细节保留，但低频结构被拉回原始图。

    Args:
        original: 原始低分图.
        sr_output: 超分输出图.
        alpha: 锚点强度，0=不锚定，1=完全对齐低频.
        sigma: 低通滤波的 sigma.

    Returns:
        锚定后的 PIL Image.
    """
    # 上采样原图到 SR 输出尺寸
    sr_w, sr_h = sr_output.size
    original_up = original.resize((sr_w, sr_h), Image.Resampling.BILINEAR)

    orig_arr = image_to_array(original_up)
    sr_arr = image_to_array(sr_output)

    # 提取低频
    l_orig = low_pass(orig_arr, sigma)
    l_sr = low_pass(sr_arr, sigma)

    # 计算偏移并修正
    delta = l_orig - l_sr
    result = sr_arr + alpha * delta

    return array_to_image(result)
