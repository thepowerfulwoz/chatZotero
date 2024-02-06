import utils


LIBRARY_ID = 13561077
LIBRARY_TYPE = 'user'
API_KEY = 'EoIgL0JDHoLru84VrUDnS1la'

pull_articles = input("Do you want to pull articles (IF THIS IS THE FIRST TIME YOU ARE RUNNING IT, YOU MUST ANSWER YES)? (yes/no)")
filename = input("What should the name of the output JSON file be. (include the .JSON extension)?")
if pull_articles == "yes":
    library_id = input("What is the library id?\n")
    library_type = input("What is the library type?\n")
    API_KEY = input("What is the API key?")
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
qdrant = utils.create_qdrant("SRA_DOCS", filename=f"../{filename}")
found_docs = qdrant.similarity_search_with_score(query = "What are the critically important things in the life cycle of a terrirst group", k=3,score_threshold=.01)

