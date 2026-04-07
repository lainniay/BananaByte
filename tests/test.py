import base64

from core import Message, create_llm
from core.schemas import ImageContent, TextContent

img = ImageContent.from_file("test/test.jpg")

llm = create_llm(provider="gemini", model="gemini-3-pro-image-preview")

query = Message(role="user", content=[img, TextContent(text="改为线稿风格")])

res = llm.edit_image(messages=query)

img = res.images

for i in img:
    b64 = i.source
    with open("after.jpg", "wb") as f:
        f.write(base64.b64decode(b64))
