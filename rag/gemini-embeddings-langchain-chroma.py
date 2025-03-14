from datasets import load_dataset
from pathlib import Path
from langchain_community.embeddings.huggingface import HuggingFaceInferenceAPIEmbeddings
from langchain.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.schema import Document
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
import numpy as np
from sklearn.manifold import TSNE
import plotly.graph_objects as go
from huggingface_hub import login
import os
from dotenv import load_dotenv
import json
import glob
from langchain_google_genai import GoogleGenerativeAIEmbeddings

def load_dataset_to_local():
    dataset = load_dataset(path="dvilasuero/finepersonas-v0.1-tiny", split="train")
    #Path("data").mkdir(parents=True, exist_ok=True)
    for i, persona in enumerate(dataset):
        labels = json.loads(persona["summary_label"])
        folder = Path("data") / labels[0];
        folder.mkdir(parents=True, exist_ok=True)
        with open(folder / f"persona_{i}.txt", "w") as f:
            f.write(persona["persona"])

load_dotenv(override=True)
token = os.getenv('HF_TOKEN')
google_api_key = os.getenv('GOOGLE_GENERATIVE_AI_API_KEY')
os.environ["GOOGLE_API_KEY"] = google_api_key
login(token)
db_name = "vector_db"

#load_dataset_to_local()

folders = glob.glob("./w5/data/*")
text_loader_kwargs = {'encoding': 'utf-8'}
documents = []
for folder in folders:
    doc_type = os.path.basename(folder)
    loader = DirectoryLoader(folder, glob="**/*.txt", loader_cls=TextLoader, loader_kwargs=text_loader_kwargs)
    folder_docs = loader.load()
    for doc in folder_docs:
        doc.metadata["doc_type"] = doc_type
        documents.append(doc)

embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")

if os.path.exists(db_name):
    Chroma(persist_directory=db_name, embedding_function=embeddings).delete_collection()
vectorstore = Chroma.from_documents(documents=documents, embedding=embeddings, persist_directory=db_name)
print(f"Vectorstore created with {vectorstore._collection.count()} documents")

# Get one vector and find how many dimensions it has
collection = vectorstore._collection
sample_embedding = collection.get(limit=1, include=["embeddings"])["embeddings"][0]
dimensions = len(sample_embedding)
print(f"The vectors have {dimensions:,} dimensions")

result = collection.get(limit=100, include=['embeddings', 'documents', 'metadatas'])
vectors = np.array(result['embeddings'])
documents = result['documents']
doc_types = [metadata['doc_type'] for metadata in result['metadatas']]
#colors = [['blue', 'green', 'red', 'orange'][['Academia', 'Sports', 'Neuroscience', 'Healthcare'].index(t)] for t in doc_types]
colors = [['blue', 'green', 'red', 'orange', 'yellow'] [ i%5 ] for i, t in enumerate(doc_types)]
print("visalizing")

tsne = TSNE(n_components=2, random_state=42)
reduced_vectors = tsne.fit_transform(vectors)

# Create the 2D scatter plot
fig = go.Figure(data=[go.Scatter(
    x=reduced_vectors[:, 0],
    y=reduced_vectors[:, 1],
    mode='markers',
    marker=dict(size=5, color=colors, opacity=0.8),
    text=[f"Type: {t}<br>Text: {d[:100]}..." for t, d in zip(doc_types, documents)],
    hoverinfo='text'
)])

fig.update_layout(
    title='2D Chroma Vector Store Visualization',
    scene=dict(xaxis_title='x',yaxis_title='y'),
    width=800,
    height=600,
    margin=dict(r=20, b=10, l=10, t=40)
)

fig.show()