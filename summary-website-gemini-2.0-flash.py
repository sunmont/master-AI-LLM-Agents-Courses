import os
import requests
from dotenv import load_dotenv
from bs4 import BeautifulSoup
from IPython.display import Markdown, display
import google.generativeai as genai

class Website:
    url: str
    title: str
    text: str

    def __init__(self, url):
        """
        Create this Website object from the given url using the BeautifulSoup library
        """
        self.url = url
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/117.0.0.0 Safari/537.36"}
        response = requests.get(url, headers=headers)
        soup = BeautifulSoup(response.content, 'html.parser')
        self.title = soup.title.string if soup.title else "No title found"
        for irrelevant in soup.body(["script", "style", "img", "input"]):
            irrelevant.decompose()
        self.text = soup.body.get_text(separator="\n", strip=True)

'''
def get_message_parts(site):
    user_prompt = f"You are looking at a website titled {site.title}"
    user_prompt += "\nThe contents of this website is as follows; \
please provide a short summary of this website in markdown. \
If it includes news or announcements, then summarize these too.\n\n"
    user_prompt += site.text
    return user_prompt
'''
def get_content(site):
    user_prompt = f"{site.title}"
    user_prompt += f"\n{site.text}"
    return user_prompt

system_instruction = "You are an assistant that analyzes the contents of a website \
and provides a short summary, ignoring text that might be navigation related. \
Respond in markdown."

def summarize(url):
    site = Website(url)
    model = genai.GenerativeModel('gemini-2.0-flash')

    '''
    messages = [
        {"role": "user", "parts": get_message_parts(site)}
    ]
    '''
    response = model.generate_content(contents=[get_content(site), "Summarize the document."])
                                          #config = genai.GenerateContentConfig(system_instruction=system_instruction))
    #display(Markdown(summary))
    for chunk in response:
        #print(chunk.text)
        display(chunk.text)

load_dotenv(override=True)
api_key = os.getenv('GOOGLE_GENERATIVE_AI_API_KEY')

if not api_key:
    print("No API key was found - please head over to the troubleshooting notebook in this folder to identify & fix!")
elif not api_key.startswith("AI"):
    print("An API key was found, but it doesn't start AI; please check you're using the right key - see troubleshooting notebook")
elif api_key.strip() != api_key:
    print("An API key was found, but it looks like it might have space or tab characters at the start or end - please remove them - see troubleshooting notebook")
else:
    print("API key found and looks good so far!")

genai.configure(api_key=api_key)

summarize("https://huggingface.co")

'''
This document is a snapshot of the Hugging Face platform. Hugging Face positions itself as "The AI community building the future," a collaborative platform for machine learning.  It highlights its core offerings:

*   **Models:** Over 1 million models available, showcasing trending models from various organizations.
*   **Datasets:** Over 250k datasets available.
*   **Spaces:**  A place to explore and run AI applications, highlighting trending and actively running Spaces.
*   **Collaboration:** Promotes collaboration, open-source tools, and building a machine learning portfolio.
*   **Compute & Enterprise Solutions:** Offers paid compute resources (Inference Endpoints, GPU Spaces) and enterprise solutions with added features.
*   **Organizations on Hugging Face:** Lists prominent organizations using the platform.
*   **Open Source:** Showcases popular open-source libraries like Transformers, Diffusers, and Datasets.
*   **Links:**  Provides links to various aspects of the platform, including website, documentation, social media, and company information.

In essence, the document serves as a landing page emphasizing Hugging Face's role as a central hub for the machine learning community, providing tools, resources, and a platform for collaboration and innovation.
This document is a snapshot of the Hugging Face website, a platform for the machine learning community. It highlights:
'''