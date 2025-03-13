import os
import requests
import json
import time
from dotenv import load_dotenv
from bs4 import BeautifulSoup
from IPython.display import Markdown, display, update_display
from openai import OpenAI
from google import genai
from google.genai.types import (
    FunctionDeclaration,
    GenerateContentConfig,
    GenerateImagesConfig,
    AutomaticFunctionCallingConfig,
    GoogleSearch,
    HarmBlockThreshold,
    HarmCategory,
    MediaResolution,
    Part,
    Retrieval,
    SafetySetting,
    Tool,
    ToolCodeExecution,
    VertexAISearch,
)
from io import BytesIO
from PIL import Image
from pydub import AudioSegment
from pydub.playback import play
from pydub import AudioSegment
from pydub.playback import play
import gradio as gr
from huggingface_hub import InferenceClient


load_dotenv(override=True)
api_key = os.getenv('HF_TOKEN')

client = InferenceClient(
    provider="together",
    api_key=api_key,
)

user_prompt = """
Rewrite this Python code in C++ with the fastest possible implementation that produces identical output in the least time. "
Respond only with C++ code; do not explain your work other than a few comments. "
Pay attention to number types to ensure no int overflows. Remember to #include all necessary C++ packages such as iomanip.\n\n"
"""

pi = """
import time

def calculate(iterations, param1, param2):
    result = 1.0
    for i in range(1, iterations+1):
        j = i * param1 - param2
        result -= (1/j)
        j = i * param1 + param2
        result += (1/j)
    return result

start_time = time.time()
result = calculate(100_000_000, 4, 1) * 4
end_time = time.time()

print(f"Result: {result:.12f}")
print(f"Execution Time: {(end_time - start_time):.6f} seconds")
"""

messages = [
	{
		"role": "user",
		"content": user_prompt + pi
	}
]

completion = client.chat.completions.create(
    model="Qwen/Qwen2.5-Coder-32B-Instruct", 
	messages=messages, 
	max_tokens=500,
)

print(completion.choices[0].message.content)