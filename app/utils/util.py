import json
import shutil

import transformers
from pyzotero import zotero, zotero_errors

from transformers import pipeline

from langchain.schema import Document
from qdrant_client import QdrantClient
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
# import torch
from typing import List


def get_zotero(lib_id: int, lib_type: str, api_key: str):
    return zotero.Zotero(lib_id, lib_type, api_key)


def get_zotero_collection_names(zot: zotero.Zotero):
    collections = zot.collections()
    names = [collection["data"]["name"] for collection in collections]
    return names


def get_qdrant_collection_names(url: str = "http://qdrant:6333"):
    client = QdrantClient(url=url)
    collections_list = []
    collections = client.get_collections()
    print("COLLECTIONS: ", list(collections))
    for c in list(collections)[0][1]:
        collections_list.append(c.name)
    return collections_list


def get_articles(zot: zotero.Zotero, collection_name: str):
    collections = zot.collections()
    print(collections)
    print("-" * 10)
    class_articles = []
    for x, collection in enumerate(collections):
        if collection['data']['name'] == collection_name:
            class_articles = zot.collection_items(collections[x]['data']['key'])
            break
        if x == len(collections) - 1:
            raise ValueError('Collection not found, please specify a different collection and try again.')
    articles = []
    for article in class_articles:
        if article['data']['itemType'] == 'attachment' and article['data']['contentType'] == 'application/pdf':
            if 'parentItem' in article['data'].keys():
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


def articles_input(filename: str):
    f = open(filename)
    return json.load(f)


def jsonToDoc(articles: List[dict]):
    documents = []
    for article in articles:
        title = article['title']
        body = article['body']

        document = Document(page_content=body, metadata={'title': title})
        documents.append(document)
    return documents


def chunkDocs(documents: List[Document], chunkSize=1000, chunkOverlap=200):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunkSize, chunk_overlap=chunkOverlap)
    chunked_articles = text_splitter.split_documents(documents)
    return chunked_articles


# def loadQdrantFromDir(directory: str, embeddingModel: str):
#     client = QdrantClient(path=directory, prefer_grpc=True)
#     qdrant = Qdrant(client=client, collection_name="documents",
#                     embeddings=HuggingFaceEmbeddings(model_name=embeddingModel))
#     return qdrant


def docsToQdrant(docs, url, embeddingModel: str, collection_name: str):
    ###IF ADDING Back, add remove_dir parameter
    # if remove_dir and os.path.isdir(directory):
    #     shutil.rmtree(directory)
    embeddings: HuggingFaceEmbeddings = HuggingFaceEmbeddings(model_name=embeddingModel)
    qdrant = QdrantVectorStore.from_documents(
        docs, embeddings, url=url, collection_name=collection_name, prefer_grpc=True
    )
    return qdrant


def create_qdrant(docs, collection_name, url: str = "localhost:6333",
                  embeddingModel="sentence-transformers/all-MiniLM-L6-v2"):
    print("loaded data")
    documents = jsonToDoc(articles=docs)
    print("chunkingDocs")
    chunked_docs = chunkDocs(documents)
    docsToQdrant(chunked_docs, url=url, embeddingModel=embeddingModel, collection_name=collection_name)
    print("created qdrant")
    return {"response": "Created Qdrant"}


def promptQdrant(question: str, qdrant, k: int = 3):
    found_docs = qdrant.similarity_search_with_score(query=question, k=k, score_threshold=.01)
    # print(found_docs)
    content = []
    for doc in found_docs:
        content.append({"title": doc[0].metadata["title"], "body": doc[0].page_content, "score": doc[1]})
    return content

