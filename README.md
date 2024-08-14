# ChatZotero: Chat with your Zotero Documents
This program allows you to chat with any of your zotero documents that have their full text indexed. (PDFs and text files)

## Installation

Run the command:
```git clone https://github.com/thepowerfulwoz/chatZotero.git```

## Configuration
After cloning the repo, you must create a `.env` file with the values `LIBRARY_ID` `LIBRARY_TYPE` `API_KEY` and `QDRANT_URL` in the root directory of the project.
### For personal libraries:
The `LIBRARY_ID` can be found [here](https://www.zotero.org/settings/keys) if it is a personal library. 

The `LIBRARY_TYPE` is `user`.
### For group libraries:
For group libraries, the ID can be found by opening the group's page: https://www.zotero.org/groups/groupname, and hovering over the group settings link. The ID is the integer after `/groups/`. 

The `LIBRARY_TYPE` is `group`.

The `API_KEY` can also be created [here](https://www.zotero.org/settings/keys). 

# Running The Program
The program runs in a docker container with docker compose. (If docker is not installed, install docker)

After docker has been installed, navigate to the directory the repo was cloned into and run `docker compose up --build`

The web app is at `{HOST_URL}:8501`.

The dashboard of qdrant, if desired, is at `{HOST_URL}:6333/dashboard`


