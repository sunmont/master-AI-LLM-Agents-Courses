import os
import re
import random
from dotenv import load_dotenv
from huggingface_hub import login
from datasets import load_dataset, Dataset, DatasetDict
from items import Item
from loaders import ItemLoader
import matplotlib.pyplot as plt
from collections import Counter, defaultdict
import numpy as np
import pickle
from google import genai
from google.genai import types
import time
import google.generativeai as google_genai

from items import Item
from testing import Tester 

load_dotenv(override=True)
token = os.getenv('HF_TOKEN')
google_api_key = os.getenv('GOOGLE_GENERATIVE_AI_API_KEY')
os.environ["GOOGLE_API_KEY"] = google_api_key
os.environ['HF_TOKEN'] = token
#login(token)
model_name = "gemini-2.0-flash"

train_data = []
test_data = []
with open('train.pkl', 'rb') as file:
    train_data = pickle.load(file, encoding="utf-8")

with open('test.pkl', 'rb') as file:
    test_data = pickle.load(file, encoding="utf-8")

def gemini_messages_for(item):
    system_message = "You estimate prices of items. Reply only with the price, no explanation"
    
    # Modify the test prompt by removing "to the nearest dollar" and "Price is $"
    # This ensures that the model receives a cleaner, simpler prompt.
    user_prompt = item.test_prompt().replace(" to the nearest dollar", "").replace("\n\nPrice is $", "")

    # Reformat messages to Gemini’s expected format: messages = [{'role':'user', 'parts': ['hello']}]
    return [
        {"role": "system", "parts": [system_message]},  # System-level instruction
        {"role": "user", "parts": [user_prompt]},       # User's query
        {"role": "model", "parts": ["Price is $"]}  # Assistant's expected prefix for response
    ]

def get_price(s):
    s = s.replace('$', '').replace(',', '')  # Remove currency symbols and formatting
    match = re.search(r"[-+]?\d*\.\d+|\d+", s)  # Regular expression to find a number
    return float(match.group()) if match else 0  # Convert matched value to float, return 0 if no match

def gemini_2_flash(item):
    messages = gemini_messages_for(item)  # Generate messages for the model
    system_message = messages[0]['parts'][0]  # Extract system-level instruction
    user_messages = messages[1:]  # Remove system message from messages list

    global model_name
    gemini = google_genai.GenerativeModel(
        model_name=model_name, # gemini-1.5-flash-002
        system_instruction=system_message
    )

    # Adding a delay to avoid hitting the API rate limit and getting a "ResourceExhausted: 429" error
    time.sleep(5)
    
    # Generate response using Gemini API
    response = gemini.generate_content(
        contents=user_messages,
        generation_config=google_genai.GenerationConfig(max_output_tokens=5)
    )

    # Extract text response and convert to numerical price
    price = get_price(response.text)
    print(f"price:{price}")
    return price

# model_name="tunedModels/test-tuned-model-uhqxkmsefwb8" # gemini-1.5-flash-002
# google_genai.configure(api_key=google_api_key)
# Tester.test(gemini_2_flash, test_data[:20])
# exit()

wandb_integration = {"type": "wandb", "wandb": {"project": "gpt-pricer"}}
client = genai.Client()
for model_info in client.models.list():
    print(model_info.name)

# for model_info in client.tunings.list():
#     print(model_info.name)

# job = client.tunings.get(name="tunedModels/test-tuned-model-uhqxkmsefwb8")
# print(job)

training_dataset = [ [item.test_prompt().replace(" to the nearest dollar", "").replace("\n\nPrice is $", ""), f"{item.price}"] for item in test_data[:20] ]

training_dataset=types.TuningDataset(
        examples=[
            types.TuningExample(
                text_input=i,
                output=o,
            )
            for i,o in training_dataset
        ],
    )

tuning_job = client.tunings.tune(
    base_model='models/gemini-1.5-flash-001-tuning',
    training_dataset=training_dataset,
    config=types.CreateTuningJobConfig(
        epoch_count= 5,
        batch_size=4,
        learning_rate=0.001,
        tuned_model_display_name="test tuned model"
    )
)

# tunedModels/test-tuned-model-crl6lm9pbyp1
# tunedModels/test-tuned-model-uhqxkmsefwb8

time.sleep(15)
print(tuning_job)
while tuning_job.state != types.JOB_STATES_SUCCEEDED: #job.State.JOB_STATE_QUEUED:
    time.sleep(15)

print(tuning_job)