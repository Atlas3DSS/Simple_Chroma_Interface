import chromadb
import os
import openai
from PyPDF2 import PdfReader
from dotenv import load_dotenv

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

def chunk_text(text: str, max_chunk: int = 1500, min_chunk: int = 150, overlap: int = 300):
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
    

def process_txt(txt_path):
    txt_file_name = os.path.basename(txt_path).replace('.txt', '')
    chunked_folder = f"chunked_{txt_file_name}"
    if not os.path.exists(chunked_folder):
        os.makedirs(chunked_folder)
    with open(txt_path, 'r', encoding='utf-8', errors='replace') as f:
        text = f.read()
    chunks = chunk_text(text)
    for i, chunk in enumerate(chunks):
        with open(os.path.join(chunked_folder, f"chunk_{i + 1}.txt"), "w", encoding='utf-8', errors='replace') as f:
            f.write(chunk)
    return chunked_folder


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

##ATHENA FUNCTIONS AND PROMPTS##
def extract_keywords(chunks, book_title):
    extracted_keywords_folder = f"extracted_keywords_{book_title}"
    if not os.path.exists(extracted_keywords_folder):
        os.makedirs(extracted_keywords_folder)

    for i, chunk in enumerate(chunks):
        keywords = extract_keywords_athena(chunk, book_title)
        with open(os.path.join(extracted_keywords_folder, f"keywords_{i + 1}.txt"), "w", encoding='utf-8', errors='replace') as f:
            f.write(keywords)

    return extracted_keywords_folder


def extract_keywords_athena(chunk, book_title):
    response = openai.ChatCompletion.create(
        model='gpt-3.5-turbo',
        max_tokens=4000,
        messages=[
            {'role': 'system', 'content': f'You are Athena, and you extract keywords from chunks of text from books. At the start of your extraction, state "This keyword extraction corresponds to {book_title}." Now, please extract keywords from the following chunk: {chunk} from the book {book_title}. For this task, you will identify and return the most important keywords or key phrases from the chunk. At the end of your extraction, always state "This keyword extraction corresponds to {book_title}."'},
        ],
    )

    return response.choices[0].message["content"]


def chunk_cleaning(chunks, book_title):
    clean_chunks = f"clean_chunks_{book_title}"
    if not os.path.exists(clean_chunks):
        os.makedirs(clean_chunks)

    for i, chunk in enumerate(chunks):
        cleaned_chunk = clean_athena_(chunk, book_title)
        with open(os.path.join(clean_chunks, f"clean_chunk_{i + 1}.txt"), "w", encoding='utf-8', errors='replace') as f:
            f.write(cleaned_chunk)

    return clean_chunks


def clean_athena_(chunk,book_title):
    response = openai.ChatCompletion.create(
        model='gpt-3.5-turbo',
        max_tokens=4000,
        temperature=0.0,
        messages=[
            {'role': 'system', 'content': f'You are Athena, you clean chunks of text from books. At the start of your cleaning state "This cleaning corresponds to {book_title}. Now, please clean the following: {chunk} from the following book {book_title}. For this task you will clean up the formatting of the chunk delivered to you. You will return the chunk with only formatting, grammar, or spelling changes. You wil not add new content or otherwise alter the meaning of the text. At the end of your cleaning always state "This cleaning corresponds to {book_title}."'},    
        ],
    )  

    return response.choices[0].message["content"]

def summarize_chunks(summarization_path, book_title):
   
    summary_folder_name = f"summarized_{os.path.basename(summarization_path)}"
    summary_folder_path = os.path.join(summarization_path, summary_folder_name)
    if not os.path.exists(summary_folder_path):
        os.makedirs(summary_folder_path)

    chunks = []
    for file in os.listdir(summarization_path):
        if file.endswith(".txt"):
            with open(os.path.join(summarization_path, file), 'r', encoding='utf-8', errors='replace') as f:
                chunk = f.read()
                chunks.append(chunk)

    for i, chunk in enumerate(chunks):
        summary = ask_summarize_athena(chunk, book_title)
        with open(os.path.join(summary_folder_path, f"summary_{i + 1}.txt"), "w", encoding='utf-8', errors='replace') as f:
            f.write(summary)

    return summary_folder_path,book_title


def query_collection(collection):
    """Query the specified collection."""
    query_texts = []
    while True:
        query_text = input("Enter a query text (or enter 'done' to finish): ")
        if query_text.lower() == "done":
            break
        query_texts.append(query_text)
    n_results = int(input("Enter the number of results to return: "))
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


def ask_summarize_athena(chunk, book_title):
    book_title = book_title
    response = openai.ChatCompletion.create(
        model='gpt-3.5-turbo',
        max_tokens=4000,
        messages=[
            {'role': 'system', 'content': f'"You are Athena, you summarize snippets of text from books. At the start of your summaries state "This summary corresponds to {book_title}. Now, please summarize the following: {chunk} from the following book {book_title}. Verbosity is preferred over brevity. Use as many words as you need to summarize the chunk. Avoid jargon but include any relevant technical terms. Specifically pull out any definitions, keywords, or explanations of terms that are relevant to the chunk. At the end of your summary always state "This summary corresponds to {book_title}."'},    
        ],
    )  

    return response.choices[0].message["content"]

def ask_athena(athena_prompt):
    response = openai.ChatCompletion.create(
        model='gpt-3.5-turbo',
        max_tokens=4000,
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
        print("7. Summarize Chunks of TXT")
        print("8. Delete a collection")
        print("9. Clean Chunks of TXT")
        print("10. Extract Keywords from Chunks of TXT")
        print("11. Exit")
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
            process_txt(txt_path)
        elif choice == "7":
            summarization_path = input("Enter the path to the folder containing the chunks of TXT to be summarized: ")
            book_title = input("Enter the title of the book: ")
            if not os.path.isdir(summarization_path):
                print("Error: the specified path is not a directory.")
                continue
            summarize_chunks(summarization_path, book_title)
            
        elif choice == "8":
            name = input("Enter the name of the collection to delete: ")
            try:
                chroma_client.delete_collection(name=name)
                print(f"Collection {name} deleted.")
            except ValueError as e:
                print(f"Error: {e}")

        elif choice == "9":
            chunks_folder = input("Enter the path to the folder containing the chunks of TXT to be cleaned: ")
            book_title = input("Enter the title of the book: ")
            if not os.path.isdir(chunks_folder):
                print("Error: the specified path is not a directory.")
                continue
            chunks = []
            for file in os.listdir(chunks_folder):
                if file.endswith(".txt"):
                    with open(os.path.join(chunks_folder, file), 'r', encoding='utf-8', errors='replace') as f:
                        chunk = f.read()
                        chunks.append(chunk)
            clean_chunks_folder = chunk_cleaning(chunks, book_title)
            print(f"Clean chunks saved in the folder: {clean_chunks_folder}")

        elif choice == "10":
            chunks_folder = input("Enter the path to the folder containing the chunks of TXT for keyword extraction: ")
            book_title = input("Enter the title of the book: ")
            if not os.path.isdir(chunks_folder):
                print("Error: the specified path is not a directory.")
                continue
            chunks = []
            for file in os.listdir(chunks_folder):
                if file.endswith(".txt"):
                    with open(os.path.join(chunks_folder, file), 'r', encoding='utf-8', errors='replace') as f:
                        chunk = f.read()
                        chunks.append(chunk)
            extracted_keywords_folder = extract_keywords(chunks, book_title)
            print(f"Extracted keywords saved in the folder: {extracted_keywords_folder}")

        elif choice == "11":
            break



if __name__ == "__main__":
    main()
