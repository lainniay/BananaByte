import re
from pathlib import Path

from core import ImageContent, PromptLib, create_llm
from core.schemas import Message, TextContent

ROOT_DIR = Path(__file__).resolve().parents[2]
AGENT_DIR = Path(__file__).resolve().parent
INPUT_IMAGE_PATH = ROOT_DIR / "workspace" / "0_in.jpg"
OUTPUT_IMAGE_PATH = ROOT_DIR / "workspace" / "0_out.jpg"


def _extract_prompt_section(content: str) -> str:
    prompt_header = re.search(r"(?im)^(#{1,6})\s*prompt\s*:?\s*$", content)
    if not prompt_header:
        raise ValueError("LLM 输出中未找到 # Prompt 标题")

    header_level = len(prompt_header.group(1))
    prompt_content = content[prompt_header.end() :]
    next_header = re.search(rf"(?im)^#{{1,{header_level}}}\s+", prompt_content)
    if next_header:
        prompt_content = prompt_content[: next_header.start()]

    result = prompt_content.strip()
    if not result:
        raise ValueError("# Prompt 标题下没有可用内容")
    return result


llm = create_llm("openai/kimi-k2.5", "openai", timeout=1200)

banana = create_llm("gemini-2.5-flash-image", "gemini", timeout=1200 * 1000)


image_input = ImageContent.from_file(INPUT_IMAGE_PATH)

prompt_lib = PromptLib(directory=str(AGENT_DIR / "prompts"))

start_prompt = TextContent(text=prompt_lib.get("start").render())

res_list: list[Message] = []

res_list.append(llm.generate(messages=Message(content=[start_prompt, image_input])))

prompt_text = _extract_prompt_section(res_list[-1].text)
print(prompt_text)


res_list.append(
    banana.edit_image(
        messages=Message(
            content=[
                image_input,
                TextContent(text=prompt_text),
            ]
        )
    )
)

res_list[-1].images[0].save_to_file(str(OUTPUT_IMAGE_PATH))
