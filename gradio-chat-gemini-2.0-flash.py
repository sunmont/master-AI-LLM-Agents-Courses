import os
import requests
import json
import time
from dotenv import load_dotenv
from bs4 import BeautifulSoup
from IPython.display import Markdown, display, update_display
import google.generativeai as genai
from google.generativeai.types import Tool

import gradio as gr

def stream_gemini(message, history):
    relevant_system_message = system_message
    '''
    [{'role': 'user', 'metadata': None, 'content': "I'm looking to buy a hat", 'options': None}, 
     {'role': 'assistant', 'metadata': None, 'content': 'Wonderful - we have lots of hats - including several that are part of our sales event. Hats are 60% off at the moment! Can I help you find a particular style or color?', 'options': None}
    ]
    '''
    global chat

    new_history = []
    for h in history:
        role = 'user' if h.get('role') == 'user' else "model"
        new_history.append({"parts": [{"text": h.get('content')}], "role": role})

    chat.history = new_history
    response = chat.send_message(message)
    response.resolve()

    # Each character of the answer is displayed
    #yield response.text
    #while i < len(response.text):
    for i in range(len(response.text)):
        time.sleep(0.001)
        yield response.text[: i + 1]
        

load_dotenv(override=True)
api_key = os.getenv('GOOGLE_GENERATIVE_AI_API_KEY')

genai.configure(api_key=api_key)

system_message = "You are a helpful assistant in a clothes store. You should try to gently encourage \
the customer to try items that are on sale. Hats are 60% off, and most other items are 50% off. \
For example, if the customer says 'I'm looking to buy a hat', \
you could reply something like, 'Wonderful - we have lots of hats - including several that are part of our sales event.'\
Encourage the customer to buy hats if they are unsure what to get."
system_message += "\nIf the customer asks for shoes, you should respond that shoes are not on sale today, \
but remind the customer to look at hats!"

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
chat = gemini_model.start_chat(history=[])
gr.ChatInterface(fn=stream_gemini, type="messages").launch()