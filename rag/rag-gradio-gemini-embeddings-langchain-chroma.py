from datasets import load_dataset
from pathlib import Path
from langchain_community.embeddings.huggingface import HuggingFaceInferenceAPIEmbeddings
from langchain.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.schema import Document
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_core.callbacks import StdOutCallbackHandler
import numpy as np
from sklearn.manifold import TSNE
import plotly.graph_objects as go
from huggingface_hub import login
import os
from dotenv import load_dotenv
import json
import glob
import gradio as gr
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI

def chat(question, history):
    result = conversation_chain.invoke({"question": question})
    return result["answer"]

load_dotenv(override=True)
token = os.getenv('HF_TOKEN')
google_api_key = os.getenv('GOOGLE_GENERATIVE_AI_API_KEY')
os.environ["GOOGLE_API_KEY"] = google_api_key
login(token)
db_name = "vector_db"

# dvilasuero/finepersonas-v0.1-tiny
#load_dataset_to_local()
embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")

vectorstore = Chroma(persist_directory=db_name, embedding_function=embeddings)
#print(f"Vectorstore created with {vectorstore._collection.count()} documents")

# Get one vector and find how many dimensions it has
collection = vectorstore._collection
##sample_embedding = collection.get(limit=1, include=["embeddings"])["embeddings"][0]
#dimensions = len(sample_embedding)
#print(f"The vectors have {dimensions:,} dimensions")


llm = ChatGoogleGenerativeAI(temperature=0.7, model="gemini-2.0-flash")
# set up the conversation memory for the chat
memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
# the retriever is an abstraction over the VectorStore that will be used during RAG
retriever = vectorstore.as_retriever()

# putting it together: set up the conversation chain with the GPT 3.5 LLM, the vector store and memory
conversation_chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever=retriever, memory=memory, callbacks=[StdOutCallbackHandler()])

view = gr.ChatInterface(chat, type="messages").launch(inbrowser=True)