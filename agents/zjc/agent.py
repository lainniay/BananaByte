from core import Message, create_llm

llm = create_llm(model="openai/kimi-k2.5")

banana = create_llm(model="gemini-2.5-flash-image")

mes = Message(role="user", content="yes")
