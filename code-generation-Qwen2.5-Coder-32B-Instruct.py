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
Respond only with C++ code; do not explain your work other than a few comments; also generate a unit test code for the c++ code;"
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
	max_tokens=1000,
)

print(completion.choices[0].message.content)

"""
```cpp
#include <iostream>
#include <iomanip>
#include <chrono>

// Function to calculate the result
double calculate(long long iterations, int param1, int param2) {
    double result = 1.0;
    for (long long i = 1; i <= iterations; ++i) {
        long long j = i * param1 - param2;
        if (j != 0) result -= 1.0 / j;
        j = i * param1 + param2;
        if (j != 0) result += 1.0 / j;
    }
    return result;
}

// Unit test for the calculate function
void test_calculate() {
    long long iterations = 100000000;
    int param1 = 4;
    int param2 = 1;
    double expected_result = calculate(iterations, param1, param2) * 4;
    std::cout << "Test Result: " << std::fixed << std::setprecision(12) << expected_result << std::endl;
}

int main() {
    auto start_time = std::chrono::high_resolution_clock::now();
    double result = calculate(100000000, 4, 1) * 4;
    auto end_time = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> duration = end_time - start_time;

    std::cout << "Result: " << std::fixed << std::setprecision(12) << result << std::endl;
    std::cout << "Execution Time: " << std::fixed << std::setprecision(6) << duration.count() << " seconds" << std::endl;

    // Run unit test
    test_calculate();

    return 0;
}
```

```cpp
// Unit test code
#include <cassert>

void test_calculate() {
    long long iterations = 100000000;
    int param1 = 4;
    int param2 = 1;
    double expected_result = calculate(iterations, param1, param2) * 4;
    // Assuming the expected result is pre-calculated and known
    // For demonstration, we use a placeholder value
    double placeholder_expected_result = 0.999999999999; // Replace with actual expected result
    assert(std::abs(expected_result - placeholder_expected_result) < 1e-12);
    std::cout << "Unit test passed!" << std::endl;
}
```

"""