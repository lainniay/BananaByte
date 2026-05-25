import cv2
import numpy as np

from core import ImageContent

mime_to_ext = {"image/jpeg": ".jpg", "image/png": ".png"}


def adjust_rgb_blance(  # noqa : D103 TODO:
    image: ImageContent, r_gain: float = 1.0, g_gain: float = 1.0, b_gain: float = 1.0
) -> ImageContent | None:
    img_array = np.frombuffer(image.source, dtype=np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    if img is not None:
        ext = mime_to_ext.get(image.mime_type)
        if ext is None:
            return None

        img_float = img.astype(np.float32)
        b, g, r = cv2.split(img_float)
        r *= r_gain
        g *= g_gain
        b *= b_gain
        fixed = cv2.merge([b, g, r])
        fixed = np.clip(fixed, 0, 255).astype(np.uint8)
        success, encoded_img = cv2.imencode(ext, fixed)
        if success:
            return ImageContent(source=encoded_img.tobytes(), mime_type=image.mime_type)
        return None
    return None


def get_color_mean(  # noqa : D103
    image: ImageContent,
) -> tuple[np.float64, np.float64, np.float64]:
    img_array = np.frombuffer(image.source, dtype=np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    if img is None:
        return (np.float64(0), np.float64(0), np.float64(0))
    b_mean, g_mean, r_mean, _ = cv2.mean(img)
    return (np.float64(b_mean), np.float64(g_mean), np.float64(r_mean))
