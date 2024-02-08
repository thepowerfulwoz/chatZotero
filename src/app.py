from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from langchain_community.embeddings import HuggingFaceEmbeddings
from pydantic import BaseModel
from typing import List

from qdrant_client import QdrantClient
from langchain_community.vectorstores import Qdrant
# Import your functions from the script
from utils import get_zotero, get_articles, create_qdrant, promptQdrant, get_pipe, generate

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
)


# Define request models
class ZoteroConfig(BaseModel):
    lib_id: int
    lib_type: str
    api_key: str


class ZoteroCollection(BaseModel):
    collection_name: str


class Articles(BaseModel):
    articles: List[dict]
    filename: str

class QdrantCreate(BaseModel):
    articles: List[dict]
    url: str = 'localhost:6333'
    embeddingModel: str = "sentence-transformers/all-MiniLM-L6-v2"

class QdrantQuery(BaseModel):
    query: str
    url: str
    embeddingModel: str = "sentence-transformers/all-MiniLM-L6-v2"


class Content(BaseModel):
    content: str


# test
# Routes
@app.post("/zotero/articles")
def fetch_articles(zotero_config: ZoteroConfig, zotero_collection: ZoteroCollection):
    zot = get_zotero(zotero_config.lib_id, zotero_config.lib_type, zotero_config.api_key)
    articles = get_articles(zot, zotero_collection.collection_name)
    return articles


@app.post("/qdrant/create")
def create_qdrant_endpoint(qdreantCreate: QdrantCreate):
    return create_qdrant(qdreantCreate.articles, qdreantCreate.url, qdreantCreate.embeddingModel)


@app.post("/qdrant/prompt")
def prompt_qdrant(qdrant_query: QdrantQuery):
    client = QdrantClient(qdrant_query.url)
    qdrant = Qdrant(client=client, collection_name='documents',
                    embeddings=HuggingFaceEmbeddings(model_name=qdrant_query.embeddingModel))
    if qdrant:
        return promptQdrant(question=qdrant_query.query, qdrant=qdrant)
    else:
        raise HTTPException(status_code=404, detail="Qdrant not initialized")


@app.post("/text/generate")
def generate_text(content: Content):
    pipe = get_pipe()
    return generate(pipe, content.content)
