##Simple ChromaDB Interface for Collection creation, management, Embedding and built in QnA Tool
This is a Python script that allows you to create a collection of text documents, add documents to the collection, and query the collection using OpenAI's GPT-3.5 model. The script uses ChromaDB as the document database and embedding generator and uses OpenAI's API to perform chat style question-answering.

Getting Started
To use this tool, you will need to have the following:

An OpenAI API key
A ChromaDB database
Python 3 installed on your computer
You will also need to install the following Python packages:

chromadb
openai
bard api
python-dotenv
You can install these packages by running pip install -r requirements.txt.

You will also need to fill out the .env file in the root directory of this project you only need to add the following variables for this script:
OPENAI_API_KEY=<your OpenAI API key>

Usage
To use this tool, run the following command in your terminal:
main.py

This will launch the tool and display a menu with several options. You can create a new collection, add documents to an existing collection, query a collection, and delete a collection. You can also load an existing collection and perform actions on it. 

Configuration can be handled in the interface

  bard integration - instructions on how to get a token here https://github.com/Ai-Austin/BardVoice/tree/main
  
