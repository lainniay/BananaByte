from core import PromptLib, create_llm
from core.schemas import Message

llm = create_llm("openai/kimi-k2.5", "openai")

prompt_lib = PromptLib("./prompts/")

res_list: list[Message] = []

res_list.append(llm.generate(messages=Message(content=prompt_lib["start"].render())))

print(res_list[-1].text)
