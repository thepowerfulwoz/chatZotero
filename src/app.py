from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List

# Import your functions from the script
import utils

app = FastAPI()

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

class Content(BaseModel):
    content: str

# Routes
@app.post("/zotero/articles")
def fetch_articles(zotero_config: ZoteroConfig, zotero_collection: ZoteroCollection):
    zot = utils.get_zotero(zotero_config.lib_id, zotero_config.lib_type, zotero_config.api_key)
    articles = utils.get_articles(zot, zotero_collection.collection_name)
    return articles

@app.post("/articles/output")
def output_articles(articles: Articles):
    utils.articles_output(articles.articles, articles.filename)
    return {"message": "Articles written to file successfully"}

@app.post("/articles/input")
def input_articles(filename: str):
    return utils.articles_input(filename)

@app.post("/documents/jsonToDoc")
def convert_json_to_doc(articles: Articles):
    return utils.jsonToDoc(articles.articles)

@app.post("/documents/chunk")
def chunk_documents(documents: List[dict]):
    return utils.chunkDocs(documents)

@app.post("/qdrant/create")
def create_qdrant_endpoint(directory: str, remove_dir: bool = True, filename: str = None, embeddingModel: str = "sentence-transformers/all-MiniLM-L6-v2"):
    return utils.create_qdrant(directory, remove_dir, filename, embeddingModel)

@app.post("/qdrant/index")
def index_data_endpoint(directory: str, remove_dir: bool = True, filename: str = None):
    return utils.index_data(directory, remove_dir, filename)

@app.post("/qdrant/prompt")
def prompt_qdrant(query: str, directory: str):
    qdrant = utils.create_qdrant(directory)
    if qdrant:
        return utils.promptQdrant(query, qdrant)
    else:
        raise HTTPException(status_code=404, detail="Qdrant not initialized")

@app.post("/text/generate")
def generate_text(content: Content):
    pipe = utils.get_pipe()
    return utils.generate(pipe, content.content)