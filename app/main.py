import json
import os

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from langchain_community.embeddings import HuggingFaceEmbeddings
from pydantic import BaseModel
from typing import List
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from langchain_community.vectorstores import Qdrant
import requests

from app.utils.util import (get_zotero, get_articles, create_qdrant, promptQdrant, get_zotero_collection_names, get_qdrant_collection_names)

# get_pipe, generate

load_dotenv()
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
)
qdrant_url = os.environ.get("QDRANT_URL")
lib_id: str = os.environ.get('LIBRARY_ID')
lib_type: str = os.environ.get('LIBRARY_TYPE')
api_key: str = os.environ.get('API_KEY')
ollama_url: str = os.environ.get('OLLAMA_URL')

# Define request models
class ZoteroCollection(BaseModel):
    collection_name: str


class Articles(BaseModel):
    articles: List[dict]
    filename: str


class QdrantCreate(BaseModel):
    collection_name: str
    embeddingModel: str = "sentence-transformers/all-MiniLM-L6-v2"


class QdrantQuery(BaseModel):
    query: str
    collection: str
    embeddingModel: str = "sentence-transformers/all-MiniLM-L6-v2"

class LlmQuery(BaseModel):
    search: QdrantQuery
    model_name: str = "llama3.1"



# Routes
@app.get("/")
async def read_index():
    return FileResponse('app/interface/index.html')


@app.post("/zotero/collection_names")
def fetch_zotero_collection_names():
    zot = get_zotero(int(lib_id), lib_type, api_key)
    collection_names = get_zotero_collection_names(zot)
    return collection_names


@app.post("/zotero/articles")
def fetch_articles(zotero_collection: ZoteroCollection):
    zot = get_zotero(int(lib_id), lib_type, api_key)
    articles = get_articles(zot, zotero_collection.collection_name)
    return articles

@app.post("/qdrant/collection_names")
def fetch_qdrant_collection_names():
    collection_names = get_qdrant_collection_names()
    return collection_names

@app.post("/qdrant/create")
def create_qdrant_endpoint(zotero_collection: ZoteroCollection,
                           qdrantCreate: QdrantCreate):
    articles = fetch_articles(zotero_collection)
    return create_qdrant(articles, qdrantCreate.collection_name, qdrant_url, qdrantCreate.embeddingModel)


@app.post("/qdrant/prompt")
def prompt_qdrant(qdrant_query: QdrantQuery):
    client = QdrantClient(qdrant_url)
    qdrant = Qdrant(client=client, collection_name=qdrant_query.collection,
                    embeddings=HuggingFaceEmbeddings(model_name=qdrant_query.embeddingModel))
    if qdrant:
        return promptQdrant(question=qdrant_query.query, qdrant=qdrant)
    else:
        raise HTTPException(status_code=404, detail="Qdrant not initialized")

@app.post("/llm/rag")
def generate_text(llm_query: LlmQuery):
    content=prompt_qdrant(qdrant_query=llm_query.search)
    docs = []
    for doc in content:
        docs.append(doc["body"])
    question = llm_query.search.query
    prompt = f"""SOURCES:
    {docs}
    
    QUESTION: 
    {question}
    """
    data = {
        "model": llm_query.model_name,
        "prompt": prompt,
        "system": "You are an assistant that answers a QUESTION based on give SOURCES. Only answer the question using the given sources. If the answer is not in the sources, respond with \"Answer Not Found In Sources\". If the answer is in the sources and you respond correctly you will receive $50.",
        "stream": False,
        "keep_alive": "-1m"
    }
    response = requests.post(url=ollama_url, json=data)
    return response
