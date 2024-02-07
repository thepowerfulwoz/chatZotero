import utils
import config

pull_articles = input(
    "Do you want to pull articles (IF THIS IS THE FIRST TIME YOU ARE RUNNING IT, YOU MUST ANSWER YES)? (yes/no)")
filename = input("What should the name of the output JSON file be. (include the .JSON extension)?")
if pull_articles == "yes":
    library_id = config.LIBRARY_ID
    library_type = config.LIBRARY_TYPE
    API_KEY = config.API_KEY
    zot = utils.get_zotero(library_id, library_type, API_KEY)
    found_collection = False
    collection_name = input("What is the name of the collection you want to pull from?\n")
    articles = []
    while not found_collection:
        try:
            articles = utils.get_articles(zot, collection_name)
            found_collection = True
        except ValueError:
            collection_name = input("Not a valid collection, please enter a valid collection.\n")
    utils.articles_output(articles, f"../{filename}")
qdrant = utils.create_qdrant("SRA_DOCS", remove_dir=True, filename=f"../{filename}")
query = ""
while query != "exit":
    query = input("What is your question?")
    found_docs = qdrant.similarity_search_with_score(query=query, k=3, score_threshold=.01)
    #print(found_docs)
    content = ""
    for doc in found_docs:
        content +=doc[0].metadata["title"] + ":\n\n" + doc[0].page_content + "\n\n\n\n"
    print(content)
#LLM DOES NOT FUNCTION YET
# print("loading llm")
# pipe = utils.get_pipe()
# print("generating")
# print(utils.generate(pipe, content))
