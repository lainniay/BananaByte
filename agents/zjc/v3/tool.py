
import cv2
import numpy as np

from core import ImageContent

img_cont = ImageContent.from_file("../../../workspace/U45_1/in.png")

img_arr = np.frombuffer(img_cont.source, dtype=np.uint8)

img = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)

img_float: np.ndarray = np.array([])

if img is not None:
    img_float = img.astype(np.float32)

b, g, r = cv2.split(img_float)

for _ in range(2):
    r *= 1.35
    g *= 0.95
    b *= 0.85

fixed = cv2.merge([b, g, r])
fixed = np.clip(fixed, 0, 255).astype(np.uint8)

success, encoded_img = cv2.imencode(".png", fixed)

if success:
    with open("../../../workspace/U45_1/color.png", "wb") as f:
        f.write(encoded_img.tobytes())

# @tool(description="Adjusts the color balance of an image. Specifically, it extracts the Red (R), Green (G), and Blue (B) channels of each pixel and multiplies them by user-specified coefficients to change the overall color tone of the image.")
# def adjust_color_balance(
#         img:Annotated[ImageContent, Field(description="The input image")]
# ) -> ImageContent:
