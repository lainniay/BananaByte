import sys

from core.llm import GeminiLLM

llm = GeminiLLM(model="gemini")

sys.stdout.write("hello")
