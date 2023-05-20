from Bard import Chatbot
from os import system
import warnings
import sys
import os
from dotenv import load_dotenv
from colorama import Fore, Style, init
import openai
from PyPDF2 import PdfReader
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import Counter
from chromadb.config import Settings
from prompts import atlas, biology, chemistry, physics, math, history, art, economics, literature, culinary_arts, generic

init(autoreset=True)  # Automatically reset to default color after each print
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
openai.api_key = OPENAI_API_KEY
bard_token = os.getenv("BARD_TOKEN")
token = bard_token
chatbot = Chatbot(token)

print("Your API key is:", openai.api_key)
print("BARD_TOKEN:", bard_token)
unwanted_strings = "william_tatum"

##CHROMA FUNCTIONS##
def create_collection(chroma_client):
    """Create a collection with a name specified by the user."""
    while True:
        name = input("Enter a name for the collection: ")
        if name in chroma_client.list_collections():
            print(f"Collection {name} already exists.")
            continue
        else:
            break
    collection = chroma_client.create_collection(name=name)
    print(f"Collection {name} created.")
    return collection, name

def load_collection(chroma_client):
    current_databases = chroma_client.list_collections()
    if len(current_databases) == 0:
        print("No collections found.")
        return
    print("Current collections:")
    # Print just the names of the collections
    for collection in current_databases:
        print(collection)
    name = input("Enter the name of the collection to load: ")
    try:
        collection = chroma_client.get_or_create_collection(name=name)
        print(f"Collection {name} loaded.")
    except:
        print("An error occurred while loading the collection.")

def add_documents_from_folder(collection, folder_path):
    """Add all text files in the specified folder to the collection."""
    documents = []
    metadatas = []
    ids = []
    for filename in os.listdir(folder_path):
        if not filename.endswith(".txt"):
            continue
        with open(os.path.join(folder_path, filename), "r", encoding="utf-8") as f:
            document = f.read()
        documents.append(document)
        metadata = {"filename": filename}
        metadatas.append(metadata)
        ids.append(filename)
    collection.add(documents=documents, metadatas=metadatas, ids=ids)
    print(f"{len(documents)} documents added to the collection.")

def add_document_from_file_path(collection, file_path):
    """Add all text files in the specified folder to the collection."""
    documents = []
    metadatas = []
    ids = []
    for filename in os.listdir(file_path):
        if not filename.endswith(".txt"):
            continue
        with open(os.path.join(file_path, filename), "r", encoding="utf-8") as f:
            document = f.read()
        documents.append(document)
        metadata = {"filename": filename}
        metadatas.append(metadata)
        ids.append(filename)
    collection.add(documents=documents, metadatas=metadatas, ids=ids)
    print(f"{len(documents)} documents added to the collection.")    

def remove_unwanted_strings(text, unwanted_strings):
    for string in unwanted_strings:
        text = text.replace(string, '')
    return text

def get_text_from_pdf(pdf_path, unwanted_strings):
    txt_file_name = pdf_path.replace('.pdf', '.txt')
    if os.path.isfile(txt_file_name):
        print("Text file already exists, skipping text extraction.")
        with open(txt_file_name, 'r', encoding='utf-8', errors='replace') as f:
            text = f.read()
    else:
        print("Getting text from pdf...")
        with open(pdf_path, 'rb') as f:
            pdf = PdfReader(f)
            number_of_pages = len(pdf.pages)
            text = ''
            for page in range(number_of_pages):
                text += pdf.pages[page].extract_text()
            text = remove_unwanted_strings(text, unwanted_strings)
            with open(txt_file_name, 'w', encoding='utf-8', errors='replace') as f:
                f.write(text)
            print("Text file saved.")
    return text

def process_chunk(chunk, i, book_title, root_folder):
    cleaned_chunk = clean_athena_(chunk, book_title)
    keywords = extract_keywords_athena(cleaned_chunk)
    summary = ask_summarize_athena(cleaned_chunk, book_title)

    with open(os.path.join(root_folder, f"chunk_{i + 1}.txt"), "w", encoding='utf-8', errors='replace') as f:
        f.write(cleaned_chunk)
    with open(os.path.join(root_folder, f"keywords_{i + 1}.txt"), "w", encoding='utf-8', errors='replace') as f:
        f.write(keywords)
    with open(os.path.join(root_folder, f"summary_{i + 1}.txt"), "w", encoding='utf-8', errors='replace') as f:
        f.write(summary)

def chunk_text(text: str, max_chunk: int = 750, min_chunk: int = 150, overlap: int = 300):
    print("Chunking text...")
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + max_chunk
        if end < len(words) and len(words[end:]) > min_chunk:
            end -= overlap
        chunks.append(' '.join(words[start:end]))
        start = end
    print("Text chunked.")
    return chunks

def process_text(txt_path, book_title):
    txt_file_name = os.path.basename(txt_path).replace('.txt', '')
    root_folder = f"{txt_file_name}_processed"
    if not os.path.exists(root_folder):
        os.makedirs(root_folder)

    with open(txt_path, 'r', encoding='utf-8', errors='replace') as f:
        text = f.read()

    chunks = chunk_text(text)
    workers = input("How many workers would you like to use? ")
    #convert to int
    max_workers = int(workers)
    with ThreadPoolExecutor(max_workers) as executor:
        futures = []
        for i, chunk in enumerate(chunks):
            futures.append(executor.submit(process_chunk, chunk, i, book_title, root_folder))

        for future in as_completed(futures):
            future.result()

    # Collect keywords from all keyword files
    keyword_list = []
    for file in os.listdir(root_folder):
        if file.startswith("keywords_"):
            with open(os.path.join(root_folder, file), 'r', encoding='utf-8', errors='replace') as f:
                for line in f.readlines():
                    keyword_list.append(line.strip())
    # Remove duplicates
    keyword_list = list(set(keyword_list))

    # Save the unique keywords to a file
    with open(os.path.join(root_folder, "unique_keywords.txt"), "w", encoding='utf-8', errors='replace') as f:
        for keyword in keyword_list:
            f.write(keyword + "\n")

    return root_folder

def keyword_collection(folder_path):
    keyword_list = []
    for file in os.listdir(folder_path):
        if not file.startswith("keywords_"):
            continue
        with open(os.path.join(folder_path, file), 'r', encoding='utf-8', errors='replace') as f:
            for line in f.readlines():
                keyword_list.append(line.strip())
    keyword_list = list(set(keyword_list))
    return keyword_list

##OPENAI FUNCTIONS##
def evaluator(text_prompt):
    response = openai.ChatCompletion.create(
        model='gpt-3.5-turbo',
        messages=[
            {
                'role': 'system',
                'content': (
                    "You are a highly trained classifier tool. You're capable of determining which subject any given question pertains to. You categorize the subjects as follows: "
                    "1 - Atlas, 2 - Biology, 3 - Chemistry, 4 - Physics, 5 - Math, 6 - History, 7 - Art, 8 - Economics, 9 - Literature, 10 - English, 11 - Culinary, 12 - Other or unknown. "
                    "For each input, you should indicate the corresponding subject number. For all things 3D printing or having the word 'atlas' choose 1. If you dont know, or otherwise can't answer choose 12. Response should be a number between 1 and 12. subject': subject_number "
                )
            },
            {'role': 'user', 'content': "What are the main characteristics of Baroque art?"},
            {'role': 'assistant', 'content': "7"},
            {'role': 'user', 'content': "Who is Atlas?"},
            {'role': 'assistant', 'content': "1"},
            {'role': 'user', 'content': text_prompt},
        ],
    )
    #print(response)
    subject_number_str = response.choices[0].message["content"]
    # Check if subject_number_str is a valid integer. If not, default to 12.
    try:
        subject_number = int(subject_number_str)
    except ValueError:
        #print(f"Invalid subject number received from AI: {subject_number_str}. Defaulting to 12.")
        subject_number = 12
    return {'subject': subject_number}

def get_prompts(subject_number):
    # Print the received subject number
    print(f"Executor received subject number: {subject_number}")  

    # Prepare a dictionary with subject prompts
    prompts_dict = {
        1: atlas,  
        2: biology,  
        3: chemistry,
        4: physics,
        5: math,  
        6: history,
        7: art,  
        8: economics,
        9: literature,
        10: literature,
        11: culinary_arts,
        12: generic  
    }

    # If the subject number is not in the dictionary, set it to 'generic' by default
    subject_prompt = prompts_dict.get(subject_number, generic)    
    # Return the response
    return subject_prompt


def prompt_athena(subject_prompts, combined_text):
    # Copy the subject prompts
    messages = subject_prompts.copy()
    # Append the user's prompt
    messages.append({'role': 'user', 'content': f"{combined_text}"})

    response = openai.ChatCompletion.create(
        model='gpt-3.5-turbo',
        messages=messages
    )
    return response.choices[0].message["content"]

## UTILITY FUNCTIONS AND PROMPTS##
def extract_keywords_athena(cleaned_chunk):
    response = openai.ChatCompletion.create(
        model='gpt-3.5-turbo',
        temperature=0.0,
        messages=[
            {'role': 'system', 'content': f'Extract keywords from {cleaned_chunk} Identify and return 10 keywords combinations from the chunk.'},
        ],
    )
    return response.choices[0].message["content"]

def clean_athena_(chunk, book_title):
    response = openai.ChatCompletion.create(
        model='gpt-3.5-turbo',
        temperature=0.0,
        messages=[
            {'role': 'system', 'content': "You are a diligent editor who meticulously cleans and formats text excerpts from books. You improve the readability of the text by fixing any formatting, grammar, or spelling errors, and ensure that the text follows a clear and elegant style. Your task is not to add or remove any content or alter the meaning, but to present the existing text in the best possible way."},
            {'role': 'user', 'content': chunk},
            {'role': 'assistant', 'content': f"From the book '{book_title}': {chunk}"},
        ],
    )  

    return response.choices[0].message["content"]

def ask_summarize_athena(chunk, book_title):
    book_title = book_title
    response = openai.ChatCompletion.create(
        model='gpt-3.5-turbo',
        messages=[
            {'role': 'system', 'content': 'You are a knowledgeable assistant who summarizes sections of text from books. You must create an accurate summary of at least two paragraphs that is easy for a beginner to understand, avoiding unnecessary jargon but including relevant technical terms when necessary. Each response should include the summary itself and a glossary of any technical terms used, with their definitions. Please ensure your response is clear, well-formatted, and free from grammatical or spelling errors.'},
            {'role': 'user', 'content': chunk},
            {'role': 'assistant', 'content': f"From the book '{book_title}': {chunk}"},
        ],
    )  

    return response.choices[0].message["content"]

def get_context(collection, prompt_text):
    """Query the specified collection."""
    query_texts = [prompt_text]
    n_results = 1
    results = collection.query(query_texts=query_texts, n_results=n_results)
    print(results)
    
    # Check if distances are present and extract them
    if "distances" in results and results["distances"][0][0] > 0.95:
        context = context_bard(prompt_text)
    else:
        # Initialize the texts variable
        texts = []
        
        # Extract the texts from the results
        if "documents" in results:
            for document in results["documents"]:
                texts.append(document[0])

        # Concatenate the top search results into a single context string
        context = "\n".join(texts)

    # Combine the context and the last query_text in the list
    combined_text = f"Please use this relevant info: {context} in response to this: {query_texts[-1]} "

    # Return the response
    return combined_text

##GOOGLE BARD FUNCTIONS##
def critical_bard(prompt_text, response):
    modified_prompt = f"Athena was asked {prompt_text} and she responded with {response}. Be critical if she is wrong or add more nuance if she is right."
    bard_response = chatbot.ask(modified_prompt)
    return bard_response['content']

def context_bard(prompt_text):
    modified_prompt = f"Athena was asked {prompt_text} please help her with some context."
    bard_response = chatbot.ask(modified_prompt)
    return bard_response['content']