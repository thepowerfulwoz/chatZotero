from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.params import Depends
from starlette.responses import FileResponse
from langchain_community.embeddings import HuggingFaceEmbeddings
from pydantic import BaseModel
from typing import List
from typing_extensions import Annotated

from functools import lru_cache

from qdrant_client import QdrantClient
from langchain_community.vectorstores import Qdrant

from utils import get_zotero, get_articles, create_qdrant, promptQdrant, get_pipe, generate
import config


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
    collection_name: str
    embeddingModel: str = "sentence-transformers/all-MiniLM-L6-v2"


class QdrantQuery(BaseModel):
    query: str
    collection: str
    embeddingModel: str = "sentence-transformers/all-MiniLM-L6-v2"


class Content(BaseModel):
    content: str


@lru_cache
def get_settings():
    return config.Settings()


# Routes
@app.get("/")
async def read_index():
    return FileResponse('../interface/index.html')


@app.post("/zotero/articles")
def fetch_articles(settings: Annotated[config.Settings, Depends(get_settings)], zotero_collection: ZoteroCollection):
    zot = get_zotero(int(settings.library_id), settings.library_type, settings.api_key)
    articles = get_articles(zot, zotero_collection.collection_name)
    return articles


@app.post("/qdrant/create")
def create_qdrant_endpoint(settings: Annotated[config.Settings, Depends(get_settings)], zotero_collection: ZoteroCollection,
                           qdrantCreate: QdrantCreate):
    articles = fetch_articles(settings, zotero_collection)
    return create_qdrant(articles, qdrantCreate.collection_name, settings.qdrant_url, qdrantCreate.embeddingModel)


@app.post("/qdrant/prompt")
def prompt_qdrant(settings: Annotated[config.Settings, Depends(get_settings)], qdrant_query: QdrantQuery):
    client = QdrantClient(settings.qdrant_url)
    qdrant = Qdrant(client=client, collection_name=qdrant_query.collection,
                    embeddings=HuggingFaceEmbeddings(model_name=qdrant_query.embeddingModel))
    if qdrant:
        return promptQdrant(question=qdrant_query.query, qdrant=qdrant)
    else:
        raise HTTPException(status_code=404, detail="Qdrant not initialized")


@app.post("/text/generate")
def generate_text(content: Content):
    pipe = get_pipe()
    return generate(pipe, content.content)
