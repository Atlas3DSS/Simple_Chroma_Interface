##Simple ChromaDB Interface for Collection creation, management, Embedding and built in QnA Tool
This is a Python script that allows you to create a collection of text documents, add documents to the collection, and query the collection using OpenAI's GPT-3.5 model. The script uses ChromaDB as the document database and OpenAI's API to generate embeddings and perform question-answering.

Getting Started
To use this tool, you will need to have the following:

An OpenAI API key
A ChromaDB database
Python 3 installed on your computer
You will also need to install the following Python packages:

chromadb
openai
python-dotenv
You can install these packages by running pip install -r requirements.txt.

You will also need to fill out the .env file in the root directory of this project you only need to add the following variables for this script:
OPENAI_API_KEY=<your OpenAI API key>

Usage
To use this tool, run the following command in your terminal:
simple_chroma_interface.py

This will launch the tool and display a menu with several options. You can create a new collection, add documents to an existing collection, query a collection, and delete a collection. You can also load an existing collection and perform actions on it.

When querying a collection, you will be prompted to enter a query text and the number of results to return. The tool will generate embeddings for the documents in the collection using OpenAI's API, and return the top search results.

Configuration can be handled in the interface
![image](https://user-images.githubusercontent.com/89653506/232256119-846cd869-b961-46e3-ba9a-82b9b75a5c32.png)
![image](https://user-images.githubusercontent.com/89653506/232256145-a9cc671f-57b2-4eed-85be-cbc573fb2974.png)
![image](https://user-images.githubusercontent.com/89653506/232256172-5936acea-6e2a-4acd-a396-0302f45881a2.png)

