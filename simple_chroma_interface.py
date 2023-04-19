import chromadb
import os
import openai
from PyPDF2 import PdfReader
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed
from autoresearcher import literature_review
from collections import Counter

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
openai.api_key = OPENAI_API_KEY
print("OPENAI_API_KEY:", OPENAI_API_KEY)
print("Welcome to the Embedding and QnA tool!")

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


def concatenate_keywords(keyword_path):
    """Concatenate all keywords from the specified folder into one list."""
    keywords = set()
    for filename in os.listdir(keyword_path):
        if not filename.startswith("keywords_"):
            continue
        with open(os.path.join(keyword_path, filename), "r", encoding="utf-8") as f:
            keyword = f.read().strip()
        keywords.add(keyword)

    # Save the unique keywords in a CSV txt file called all_keywords.txt
    with open(os.path.join(keyword_path, "all_keywords.txt"), "w", encoding="utf-8") as f:
        f.write(','.join(keywords))

    print(f"{len(keywords)} unique keywords added to the collection.")

    # Find the top 15 most common words and save them to most_common_keywords.txt
    word_counts = Counter(','.join(keywords).split(','))
    most_common_keywords = word_counts.most_common(15)

    with open(os.path.join(keyword_path, "most_common_keywords.txt"), "w", encoding="utf-8") as f:
        for word, count in most_common_keywords:
            f.write(f"{word}\n")

    return list(keywords)


unwanted_strings = ["This ebook belongs to William Tatum (info@atlas3dss.com),", "purchased on 14/04/2023"]
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
    max_workers = input("How many workers would you like to use? ")
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


##ATHENA FUNCTIONS AND PROMPTS##
def extract_keywords_athena(cleaned_chunk):
    response = openai.ChatCompletion.create(
        model='gpt-3.5-turbo',
        temperature=0.0,
        messages=[
            {'role': 'system', 'content': f'Extract keywords from {cleaned_chunk} Identify and return 10 keywords combinations from the chunk.'},
        ],
    )

    return response.choices[0].message["content"]

def clean_athena_(chunk,book_title):
    response = openai.ChatCompletion.create(
        model='gpt-3.5-turbo',
        
        temperature=0.0,
        messages=[
            {'role': 'system', 'content': f'You clean chunks of text from books. Please clean the following: {chunk} from the following book {book_title}. Clean up the formatting of the chunk delivered to you. Return the chunk with clean, clear, elegant formatting, grammar, and spelling corrections. Add nothing new. Do not alter the meaning of the text.'},    
        ],
    )  

    return response.choices[0].message["content"]

def ask_summarize_athena(chunk, book_title):
    book_title = book_title
    response = openai.ChatCompletion.create(
        model='gpt-3.5-turbo',
        
        messages=[
            {'role': 'system', 'content': f'"You summarize snippets of text from books. Summarize the following: {chunk} from the following book {book_title}. Verbosity is preferred over brevity. Summarize as a subject matter expert would when speaking to a junior mentee. Avoid jargon, but include  relevant technical terms. Response should have 1-2 paragraphs of summary, Summary: Content. Include a Glossary of terms at the end of the summary. Glossary: Term, Definition. Return the summary with clean, clear, elegant formatting, grammar, and spelling corrections.'},    
        ],
    )  

    return response.choices[0].message["content"]


def query_collection(collection):
    """Query the specified collection."""
    query_texts = []
    while True:
        query_text = input("Enter a query text (or enter 'done' to finish): ")
        if query_text.lower() == "done":
            break
        query_texts.append(query_text)
    n_results = 10
    results = collection.query(query_texts=query_texts, n_results=n_results)
        
    # Initialize the texts variable
    texts = []
    
    # Extract the texts from the results
    if "documents" in results:
        for document in results["documents"]:
            texts.append(document[0])

    # Concatenate the top search results into a single context string
    context = "\n".join(texts)

    # Combine the context and the last query_text in the list
    combined_text = f"{query_texts[-1]} {context}"
    
    # Create a prompt for Athena
    athena_prompt = f"Athena, you are a wise and helpful goddess. Please help me understand the following: {combined_text}"

    # Pass the athena_prompt to the ask_athena function
    athena_response = ask_athena(athena_prompt)
    
    # Print the response
    print(f"Athena's response: {athena_response}")

def ask_athena(athena_prompt):
    response = openai.ChatCompletion.create(
        model='gpt-3.5-turbo',
        
        messages=[
            {'role': 'system', 'content': f'You are Athena, a helpful and wise AI assistant. You are a subject matter expert in all domains.Please respond to the following prompt: {athena_prompt}'},
        ],
    )
    ##make a folder called athena_responses
    if os.path.isdir('athena_responses'):
        pass
    else:
        os.mkdir('athena_responses')
    ##check and see if any files are in the folder if so get the last file number and add 1 to it
    ##if not start at 1
    if len(os.listdir('athena_responses')) == 0:
        file_number = 1
    else:
        file_number = int(os.listdir('athena_responses')[-1].replace('.txt', '')) + 1
    ##save the response to a file in the athena_responses folder
    with open(os.path.join('athena_responses', f'{file_number}.txt'), 'w', encoding='utf-8', errors='replace') as f:
        f.write(response.choices[0].message["content"])    
    return response.choices[0].message["content"]

##SPECIAL MENU FOR COLLECTIONS##
def collection_menu(collection):
    while True:
        print(f"\nCollection '{collection.name}' Menu\n-----------------------------")
        print("1. Query the collection")
        print("2. Add documents to the collection")
        print("3. Delete the collection")
        print("4. Return to main menu")
        choice = input("Enter your choice: ")
        if choice == "1":
           while True: 
            query_collection(collection)            
        elif choice == "2":
            ##get folder path pass to add documents from folder function do not create a folder if its not there throw error
            folder_path = input("Enter the path to the folder containing the text files: ")
            if not os.path.isdir(folder_path):
                print("Error: the specified path is not a directory.")
                continue
            add_documents_from_folder(collection, folder_path)
        elif choice == "3":
            try:
                collection.client.delete_collection(name=collection.name)
                print(f"Collection '{collection.name}' deleted.")
                break
            except ValueError as e:
                print(f"Error: {e}")
        elif choice == "4":
            break
        else:
            print("Invalid choice, please try again.")

##MAIN FUNCTION##
def main():
    chroma_client = chromadb.Client(chromadb.config.Settings(
        chroma_db_impl="duckdb+parquet",
        persist_directory="ChromaDB"
    ))
    while True:
        print("\nChromaDB Menu\n-------------")
        print("1. Create a collection")
        print("2. Add documents from a folder to a collection")
        print("3. Query a collection")
        print("4. Load a collection")
        print("5. Process a PDF")
        print("6. Process a TXT")
        print("7. Delete a collection")
        print("8. Keyword merge")
        print("9. Generate a lit review")
        print("10. Exit")
        choice = input("Enter your choice: ")
        if choice == "1":
            name = input("Enter a name for the collection: ")
            if name in chroma_client.list_collections():
                collection = chroma_client.load_collection(name=name)
                print(f"Collection {name} loaded.")
            else:
                collection = chroma_client.create_collection(name=name)
                print(f"Collection {name} created.")
                os.makedirs(os.path.join("ChromaDB", name))
        elif choice == "2":
            if not "collection" in locals():
                print("Error: no collection selected.")
                continue
            folder_path = input("Enter the path to the folder containing the text files: ")
            if not os.path.isdir(folder_path):
                print("Error: the specified path is not a directory.")
                continue
            add_documents_from_folder(collection, folder_path)
        elif choice == "3":
            if not "collection" in locals():
                print("Error: no collection selected.")
                continue
            query_collection(collection)
        elif choice == "4":
            current_databases = chroma_client.list_collections()
            if len(current_databases) == 0:
                print("No collections found.")
                continue
            print("Current collections:")
            ##print just the names of the collections
            for collection in current_databases:
                print(collection)
            name = input("Enter the name of the collection to load: ")
            try:
                collection = chroma_client.get_or_create_collection(name=name)
                print(f"Collection {name} loaded.")
                collection_menu(collection)
            except ValueError as e:
                print(f"Error: {e}")
        elif choice == "5":
            pdf_path = input("Enter the path to the PDF: ")
            get_text_from_pdf(pdf_path, unwanted_strings)
        elif choice == "6":
            txt_path = input("Enter the path to the TXT: ")
            book_title = input("Enter the title of the book: ")
            process_text(txt_path,book_title)
        elif choice == "7":
            name = input("Enter the name of the collection to delete: ")
            try:
                chroma_client.delete_collection(name=name)
                print(f"Collection {name} deleted.")
            except ValueError as e:
                print(f"Error: {e}")
        elif choice == "8":
            keyword_path = input("Enter the path to the folder containing the keywords files: ")
            concatenate_keywords(keyword_path)
        elif choice == "9":            
            research_question = input("Enter the research question: ")
            output_filename = f"my_{research_question.replace(' ', '_')}_review.txt"
            researcher = literature_review(research_question, output_file=output_filename)

        elif choice == "10":
            break

if __name__ == "__main__":
    main()
