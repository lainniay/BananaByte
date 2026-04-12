from langsmith import Client

from core import ImageContent, create_llm
from core.schemas import Message, TextContent

llm = create_llm(model="gemini-2.5-flash-image", provider="gemini")


img = ImageContent.from_file(path="../workspace/test.jpg")

txt = TextContent(text="Change the color tone to purple and output the picture")


cli = Client()


ress = llm.edit_image(messages=Message(role="user", content=[img, txt]))

cli.flush()
