import os
import os.path

import json
import shutil

from pyzotero import zotero, zotero_errors

from langchain.schema import Document
from qdrant_client import QdrantClient
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Qdrant
from langchain.embeddings import HuggingFaceEmbeddings
import torch

LIBRARY_ID = 13561077
LIBRARY_TYPE = 'user'
API_KEY = 'EoIgL0JDHoLru84VrUDnS1la'


def get_zotero(lib_id, lib_type, api_key):
    return zotero.Zotero(lib_id, lib_type, api_key)


def get_articles(zot: zotero.Zotero):
    collections = zot.collections()
    class_articles = zot.collection_items(collections[0]['data']['key'])
    articles = []
    for article in class_articles:
        if article['data']['itemType'] == 'attachment' and article['data']['contentType'] == 'application/pdf':
            article_title = zot.item(article['data']['parentItem'])['data']['title']
            print(article['data']['filename'])
            print(article['data']['key'])
            print(article_title)
            print('-' * 10)
            try:
                articles.append({
                    'title': article_title,
                    'body': zot.fulltext_item(article['data']['key'])['content']
                })
            except zotero_errors.ResourceNotFound:
                print(f"Could not get text for: {article_title}")
                print("-" * 10)
                continue
    return articles


def articles_output(articles: list, filename: str):
    with open(filename, "w") as final:
        json.dump(articles, final)


def articles_input(filename):
    f = open(filename)
    return json.load(f)


def jsonToDoc(articles: list[dict]):
    documents = []
    for article in articles:
        title = article['title']
        body = article['body']

        document = Document(page_content=body, metadata={'title': title})
        documents.append(document)
    return documents


def chunkDocs(documents: list[Document], chunkSize=1000, chunkOverlap=200):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunkSize, chunk_overlap=chunkOverlap)
    chunked_articles = text_splitter.split_documents(documents)
    return chunked_articles


def loadQdrantFromDir(directory, embeddingModel):
    client = QdrantClient(path=directory, prefer_grpc=True)
    qdrant = Qdrant(client=client, collection_name="documents",
                    embeddings=HuggingFaceEmbeddings(model_name=embeddingModel))
    return qdrant


def docsToQdrant(docs, remove_dir: bool, directory, embeddingModel):
    if remove_dir and os.path.isdir(directory):
        shutil.rmtree(directory)
    embedddings: HuggingFaceEmbeddings = HuggingFaceEmbeddings(model_name=embeddingModel)
    qdrant = Qdrant.from_documents(
        docs, embedddings, path=directory, collection_name='documents'
    )
    return qdrant


def create_qdrant(directory, remove_dir: bool, filename=None, embeddingModel="sentence-transformers/all-MiniLM-L6-v2"):
    if directory and not remove_dir:
        if os.path.isdir(directory):
            return loadQdrantFromDir(directory, embeddingModel)
    if filename is None:
        print("must specify filename on first run")
        return None
    f = open(filename)

    # returns JSON object as
    # a dictionary
    data = json.load(f)
    documents = jsonToDoc(data)
    chunked_docs = chunkDocs(documents)
    qdrant = docsToQdrant(chunked_docs, remove_dir, directory, embeddingModel)

    return qdrant
