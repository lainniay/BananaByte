import json

from core import PromptLib, create_llm
from core.schemas import ImageContent, Message, TextContent

llm = create_llm("openai/kimi-k2.5", "openai")
banana = create_llm("gemini-3-pro-image-preview", "gemini")

prompt_lib = PromptLib("./prompts/")

res_list: list[Message] = []

image = ImageContent.from_file("../../workspace/0_in.jpg")

mes_start = Message(
    content=[
        TextContent(
            text=prompt_lib["start"].render(
                memory="do not have memory, determine by input image"
            )
        ),
        image,
    ]
)

res_list.append(llm.generate(messages=mes_start))

data = json.loads(res_list[-1].text)

banana_prompt = Message(content=[TextContent(text=json.dumps(data)), image])

res_list.append(banana.edit_image(banana_prompt))

res_list[-1].images[0].save_to_file("../../workspace/0_out.jpg")
