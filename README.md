# ChatZotero: Chat with your Zotero Documents
This program allows you to chat with any of your zotero documents that have their full text indexed. (PDFs and text files)

## Installation

Run the command:
```git clone https://github.com/thepowerfulwoz/chatZotero.git```

To install dependencies: ```pip install -r requirements.txt```

## Configuration
After cloning the repo, you must edit the three values in the `config.py` file.
### For personal libraries:
The `LIBRARY_ID` can be found [here](https://www.zotero.org/settings/keys) if it is a personal library. 

The `LIBRARY_TYPE` is `user`.
### For group libraries:
For group libraries, the ID can be found by opening the group's page: https://www.zotero.org/groups/groupname, and hovering over the group settings link. The ID is the integer after `/groups/`. 

The `LIBRARY_TYPE` is `group`.

# Running The Program
Navigate to the program directory and run the command ```python3 /src/main.py``` 


