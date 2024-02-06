import utils

#TODO
#make article gathering modular so that you can pull from any collection as long as you know the name


qdrant = utils.create_qdrant(directory="SRA_DOCS", remove_dir=True, filename='../SRA211_2-5-24.json')
found_docs = qdrant.similarity_search_with_score(query = "What are the critically important things in the life cycle of a terrirst group", k=3,score_threshold=.01)

