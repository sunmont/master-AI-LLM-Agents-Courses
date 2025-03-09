import torch
from huggingface_hub import login
from transformers import pipeline
from diffusers import DiffusionPipeline
from datasets import load_dataset
#import soundfile as sf
#from IPython.display import Audio
import os
from dotenv import load_dotenv

def check_gpu():
    print('__CUDNN VERSION:', torch.backends.cudnn.version())
    print('__Number CUDA Devices:', torch.cuda.device_count())
    print('__CUDA Device Name:',torch.cuda.get_device_name(0))
    print('__CUDA Device Total Memory [GB]:',torch.cuda.get_device_properties(0).total_memory/1e9)

# check_gpu()

load_dotenv(override=True)
token = os.getenv('HF_TOKEN')
login(token)

# Sentiment Analysis
'''
classifier = pipeline("sentiment-analysis", device="cuda")
result = classifier("I'm super excited to be on the way to LLM mastery!")
print(result)
'''

# Named Entity Recognition
# ner = pipeline("ner", grouped_entities=True, device="cuda")
# result = ner("Barack Obama was the 44th president of the United States.")
# print(result)

# Text Summarization
# summarizer = pipeline("summarization", device="cuda")
# text = """The Hugging Face transformers library is an incredibly versatile and powerful tool for natural language processing (NLP).
# It allows users to perform a wide range of tasks such as text classification, named entity recognition, and question answering, among others.
# It's an extremely popular library that's widely used by the open-source data science community.
# It lowers the barrier to entry into the field by providing Data Scientists with a productive, convenient way to work with transformer models.
# """
# summary = summarizer(text, max_length=50, min_length=25, do_sample=False)
# print(summary[0]['summary_text'])

# Another translation, showing a model being specified
# translator = pipeline(task="translation",
#                       model="facebook/nllb-200-distilled-600M",
#                       torch_dtype=torch.bfloat16,
#                       device="cuda")

# result = translator("The Data Scientists were truly amazed by the power and simplicity of the HuggingFace pipeline API.",
#                         src_lang = 'eng_Latn',
#                         tgt_lang = 'zho_Hans')
# print(result[0]['translation_text'])
# # 数据科学家们对 HuggingFace管道API的实力和简单性感到惊

# Image Generation
image_gen = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-2",
    torch_dtype=torch.float16,
    use_safetensors=True,
    variant="fp16"
    ).to("cuda")

text = "chinese traditional kung fu players" #"A class of Data Scientists learning about AI, in the surreal style of Salvador Dali"
image = image_gen(prompt=text).images[0]
image.save("kf.png")
image.show()