import os

from dotenv import load_dotenv
from google import genai
from google.genai.types import HttpOptions

load_dotenv()

client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY"),
    http_options=HttpOptions(base_url=os.getenv("GEMINI_API_BASE"), timeout=10000),
)

try:
    print("--- 正在获取可用模型列表 ---")
    models = client.models.list()
    for m in models:
        # 打印模型 ID 和支持的方法
        print(f"模型名称: {m.name}")
except Exception as e:
    print(f"获取失败，请检查 API Key 或 Base URL。错误信息: {e}")
