import os
import requests
import json
from dotenv import load_dotenv
from bs4 import BeautifulSoup
from IPython.display import Markdown, display, update_display
import google.generativeai as genai
from google.generativeai.types import Tool

import gradio as gr

def model_prompt():
    system_message = "You are an assistant that is great at telling complex jokes"
    user_prompt = "Tell a light-hearted joke for an audience of Software Developer"
    
    gemini_model = genai.GenerativeModel(
        'gemini-2.0-flash',
        system_instruction = system_message
    )

    # Set model parameters
    generation_config = genai.GenerationConfig(
        temperature=0.9,
        top_p=1.0,
        top_k=32,
        candidate_count=1,
        max_output_tokens=8192,
    )

    response = gemini_model.generate_content(
        [user_prompt],
        generation_config=generation_config,
        stream=True
    )

    for chunk in response:
        print(chunk.text)

def stream_gemini(prompt):
    response = gemini_model.generate_content(
        [prompt],
        #generation_config = generation_config,
        stream=True
    )

    #yield response.text
    for chunk in response:
       yield chunk.text

class Website:
    url: str
    title: str
    text: str

    def __init__(self, url):
        self.url = url
        response = requests.get(url)
        self.body = response.content
        soup = BeautifulSoup(self.body, 'html.parser')
        self.title = soup.title.string if soup.title else "No title found"
        for irrelevant in soup.body(["script", "style", "img", "input"]):
            irrelevant.decompose()
        self.text = soup.body.get_text(separator="\n", strip=True)

    def get_contents(self):
        return f"Webpage Title:\n{self.title}\nWebpage Contents:\n{self.text}\n\n"

def stream_brochure(company_name, url):
    prompt = f"Please generate a company brochure for {company_name}. Here is their landing page:\n"
    prompt += Website(url).get_contents()
    print(prompt)
    stream_gemini(prompt)


load_dotenv(override=True)
api_key = os.getenv('GOOGLE_GENERATIVE_AI_API_KEY')

genai.configure(api_key=api_key)
# model_prompt()

force_dark_mode = """
function refresh() {
    const url = new URL(window.location);
    if (url.searchParams.get('__theme') !== 'dark') {
        url.searchParams.set('__theme', 'dark');
        window.location.href = url.href;
    }
}
"""
#system_message = "You are a helpful assistant that responds in markdown, please search web for more information"
system_message = "You are an assistant that analyzes the contents of a company website landing page \
and creates a short brochure about the company for prospective customers, investors and recruits. Respond in markdown."
gemini_model = genai.GenerativeModel(
        'gemini-2.0-flash',
        system_instruction = system_message
    )
 # Set model parameters
generation_config = genai.GenerationConfig(
    temperature=0.9,
    top_p=1.0,
    top_k=32,
    candidate_count=1,
    max_output_tokens=8192,
)

view = gr.Interface(
    #fn = stream_gemini,
    fn = stream_brochure,
    inputs=[
        gr.Textbox(label="Company name:"),
        gr.Textbox(label="Landing page URL including http:// or https://"),
    ],
    outputs=[gr.Markdown(label="Response:")],
    flagging_mode="never"
)
view.launch()