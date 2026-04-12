from core import ImageContent, create_llm
from core.schemas import Message, TextContent

llm = create_llm(model="gemini-2.5-flash-image", provider="gemini")


img = ImageContent.from_file(path="../workspace/test.jpg")

txt = TextContent(text="Change the color tone to pink and output the picture")


ress = llm.edit_image(messages=Message(role="user", content=[img, txt]))


ress.images[-1].save_to_file("../workspace/test_out.jpg")
