import langsmith

from core import create_llm
from core.schemas import Message

llm = create_llm(model="openai/kimi-k2.5", provider="openai")


mes = Message(role="user", content="hello")

client = langsmith.Client()

res = llm.generate(mes).text
print(res)

mess = Message(role="user", content="say hi again")
_ = llm.generate(mess)

client.flush()
