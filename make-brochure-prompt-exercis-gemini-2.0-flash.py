import os
import requests
import json
from dotenv import load_dotenv
from bs4 import BeautifulSoup
from IPython.display import Markdown, display, update_display
import google.generativeai as genai

class Website:
    url: str
    title: str
    links: []

    def __init__(self, url):
        self.url = url

        headers = {
 "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/117.0.0.0 Safari/537.36"
}
        response = requests.get(url, headers=headers)
        self.body = response.content
        soup = BeautifulSoup(self.body, 'html.parser')
        self.title = soup.title.string if soup.title else "No title found"
        if soup.body:
            for irrelevant in soup.body(["script", "style", "img", "input"]):
                irrelevant.decompose()
            self.text = soup.body.get_text(separator="\n", strip=True)
        else:
            self.text = ""
        links = [link.get('href') for link in soup.find_all('a')]
        self.links = [link for link in links if link]

    def get_contents(self):
        return f"Webpage Title:\n{self.title}\nWebpage Contents:\n{self.text}\n\n"

system_instruction = "You are an assistant that analyzes the contents of a website \
and provides a short summary, ignoring text that might be navigation related. \
Respond in markdown."

link_system_prompt = "You are provided with a list of links found on a webpage. \
You are able to decide which of the links would be most relevant to include in a brochure about the company, \
such as links to an About page, or a Company page, or Careers/Jobs pages.\n"
link_system_prompt += "You should respond in JSON as in this example:"
link_system_prompt += """
{
    "links": [
        {"type": "about page", "url": "https://full.url/goes/here/about"},
        {"type": "careers page": "url": "https://another.full.url/careers"}
    ]
}
"""
def get_links(url):
    site = Website(url)
  
    #print(site.links)
    response = model.generate_content(contents=["\n".join(site.links), link_system_prompt])
    json_str = response.text
    json_str = json_str.replace("```json", "").replace("```", "")

    return json.loads(json_str)

def get_all_details(url):
    result = "Landing page:\n"
    result += Website(url).get_contents()
    links = get_links(url)
    #print("Found links:", links)
    for link in links["links"]:
        result += f"\n\n{link['type']}\n"
        result += Website(url + link["url"]).get_contents()
    return result

brochure_prompt = "You are an assistant that analyzes the contents of several relevant pages from a company website \
and creates a short brochure about the company for prospective customers, investors and recruits. Respond in markdown.\
Include details of company culture, customers and careers/jobs if you have the information."
def create_brochure(company_name, url):
    user_prompt = f"You are looking at a company called: {company_name}\n"
    user_prompt += f"Here are the contents of its landing page and other relevant pages; use this information to build a short brochure of the company in markdown.\n"
    response = model.generate_content(contents=[get_all_details(url), user_prompt], stream=True)
    for chunk in response:
        display(chunk.text)

def prompt_exercise():
    _prompt =  """
Please explain what this code does and why:
yield from {book.get("author") for book in books if book.get("author")}
"""
    response = model.generate_content(contents=[_prompt], stream=True)
    for chunk in response:
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
model = genai.GenerativeModel('gemini-2.0-flash')

#create_brochure("HuggingFace", "https://huggingface.co")
#create_brochure("anthropic", "https://www.anthropic.com")
prompt_exercise()
