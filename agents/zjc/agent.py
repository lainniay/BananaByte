import base64
import mimetypes
from pathlib import Path

from core import create_llm
from core.schemas import ImageContent, Message, TextContent

llm = create_llm(model="openai/kimi-k2.5", timeout=240)

ROOT_DIR = Path(__file__).resolve().parents[2]

img = ImageContent.from_file(
    ROOT_DIR / "workspace/LowLevelEval/Underwater_Enhancement/LSUI/input/0.jpg"
)

banana = create_llm(model="gemini-3-pro-image-preview", provider="gemini")

START_PROMPT = """
You are a specialized Low-Level Image Processing AI. Your sole function is to translate user requirements into a sequence of standard low-level computer vision operations, including their specific parameters, intensities, or kernel sizes.

Given the context below, determine the exact processing pipeline required.

Constraints:
- Output ONLY the sequence of operations separated by ' -> '.
- For each operation, you MUST specify the estimated parameters, degree, or intensity in parentheses.
- Do not include any explanations, code, or conversational text.
- Use canonical algorithm names and standard parameter conventions.
- Example format: Grayscale Conversion -> Gaussian Blur (kernel=5x5, sigma=1.5) -> Canny Edge Detection (low_thresh=50, high_thresh=150) -> Morphological Dilation (kernel=3x3, iterations=2).

Input Data:
[Image Context]: {image_type}
[User Requirement]: {require}
"""

REQUIRE_PROMPT = """
This is an underwater image with obvious color cast, low contrast, and scattered fog. Please carefully analyze the main scene, eliminate the interference caused by water absorption and suspended particles, and restore a clear, fog-free version with true colors. Ensure that all elements except for the degradation factors are consistent with the real underwater environment, without introducing new artifacts or over-enhancement.

CRITICAL ELEMENT LOCK: remove only the color shift and fog; do not add, remove, or alter any original object, edge, texture.
"""

prompt = START_PROMPT.format(image_type=img.mime_type, require=REQUIRE_PROMPT)

res = llm.generate(
    messages=Message(
        role="user",
        content=[img, TextContent(text=prompt)],
    )
)

nano_banana_prompt = Message(
    role="user",
    content=[img, TextContent(text=f"Please apply this exact pipeline: {res.text}")],
)
banana_res = banana.edit_image(messages=nano_banana_prompt)

images = banana_res.images

if not images:
    raise ValueError("Gemini not return image")

output_dir = ROOT_DIR / "workspace"
output_dir.mkdir(parents=True, exist_ok=True)

for idx, image in enumerate(images, start=1):
    ext = mimetypes.guess_extension(image.mime_type) or ".jpg"
    name = f"after{ext}" if idx == 1 else f"after_{idx}{ext}"
    path = output_dir / name
    with open(path, "wb") as f:
        f.write(base64.b64decode(image.source))
