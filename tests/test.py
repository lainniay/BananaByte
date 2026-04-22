from pydantic import BaseModel
from rich import print_json

from core.llm import OpenAILLM
from core.schemas import Message

llm = OpenAILLM(model="gpt-image-2")


class ReturnSchema(BaseModel):
    name: str
    age: int
    words: str


config = llm.GenerateConfig(timeout=120, temperature=1.0, top_p=0.9, seed=100)


res = llm.generate_struct(
    messages=Message(content="hello"),
    system_prompt=" you are a cat girl",
    schema=ReturnSchema,
    config=config,
)

json_text = res.text

print_json(json_text)

answer: ReturnSchema = ReturnSchema.model_validate_json(res.text)
