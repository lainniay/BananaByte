from core import create_llm
from core.schemas import ImageContent, Message, TextContent

llm = create_llm(model="openai/kimi-k2.5", provider="openai")

mes = Message(role="user", content="hello")

res = llm.generate(mes).text

print(res)

img = ImageContent(source="../workspace/test.jpg")

mess = Message(role="user", content=[TextContent(text="yes"), img])

ress = llm.generate(mes)
