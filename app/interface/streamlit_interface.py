import requests
import streamlit as st


def get_zotero_collection_names():
    response = requests.post(url="http://backend:8000/zotero/collection_names")
    collections = eval(response.text)
    st.session_state["zotero_collections"] = collections

def get_qdrant_collection_names():
    response = requests.post(url="http://backend:8000/qdrant/collection_names")
    collections = eval(response.text)
    st.session_state["qdrant_collections"] = collections

title = st.title('chatZotero Interface')
st.button(label="Refresh Zotero Collections", on_click=get_zotero_collection_names())
with st.form(key="Create Qdrant"):
    create_qdrant_header = st.header('Create Qdrant')
    zotero_collection_name = st.selectbox("Zotero Collection", st.session_state["zotero_collections"])
    qdrant_collection_name = st.text_input("Qdrant Collection Name:")
    create_qdrant_embedding_model = st.text_input(label="Embedding Model:",
                                                  value="sentence-transformers/all-MiniLM-L6-v2")
    create_qdrant_button = st.form_submit_button("Create Qdrant")
    if create_qdrant_button:
        response = requests.post(url="http://backend:8000/qdrant/create",
                                 json={"zotero_collection": {'collection_name': zotero_collection_name},
                                       "qdrantCreate": {'collection_name': qdrant_collection_name,
                                                        'embeddingModel': create_qdrant_embedding_model}}).json()
        print(response)
        st.write(response)

st.button(label="Refresh Qdrant Collections", on_click=get_qdrant_collection_names())
with st.form(key="Prompt Qdrant"):
    prompt_qdrant_header = st.header("Prompt Qdrant")
    prompt_qdrant_query = st.text_input("Query")
    prompt_qdrant_collection_name = st.selectbox("Qdrant Collection:", st.session_state["qdrant_collections"])
    prompt_qdrant_embedding_model = st.text_input(label="Embedding Model:",
                                                  value="sentence-transformers/all-MiniLM-L6-v2")
    prompt_qdrant_button = st.form_submit_button("Prompt Qdrant")
    if prompt_qdrant_button:
        prompt_response = requests.post(url="http://backend:8000/qdrant/prompt",
                                        json={'query': prompt_qdrant_query,
                                              'collection': prompt_qdrant_collection_name,
                                              'embeddingModel': prompt_qdrant_embedding_model}).json()
        st.write(prompt_response)
