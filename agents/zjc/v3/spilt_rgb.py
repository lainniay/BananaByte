import cv2
import numpy as np

from core import ImageContent


def save_rgb_channels(image: ImageContent) -> None:  # noqa : D103
    # 从二进制图片数据解码
    img_array = np.frombuffer(image.source, dtype=np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

    if img is None:
        raise ValueError("图片解码失败")

    # 注意：OpenCV 默认顺序是 BGR
    b_channel, g_channel, r_channel = cv2.split(img)

    # 保存三个单通道灰度图
    cv2.imwrite("red_channel.png", r_channel)
    cv2.imwrite("green_channel.png", g_channel)
    cv2.imwrite("blue_channel.png", b_channel)


img = ImageContent.from_file("../../../workspace/Severe/2/in.png")
save_rgb_channels(img)
