from prompts import generic, atlas, biology, history, math, literature, art, culinary_arts, physics, chemistry, economics
from utils import prompt_athena, evaluator, add_document_from_file_path, add_documents_from_folder, get_context, get_text_from_pdf, process_chunk, unwanted_strings, process_text, get_prompts, critical_bard
import openai
from dotenv import load_dotenv
import os
from chromadb import Client
from chromadb.config import Settings
import chromadb
from colorama import Fore, Style, init

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
openai.api_key = OPENAI_API_KEY
bard_token = os.getenv("BARD_TOKEN")
token = bard_token

print(f"Your API key is: {Fore.GREEN}{openai.api_key}{Style.RESET_ALL}")
print(f"BARD_TOKEN: {Fore.GREEN}{bard_token}{Style.RESET_ALL}")
print(f"Welcome to {Fore.CYAN}Athena{Style.RESET_ALL}, the AI that can answer any question you have about any subject.\n")

chroma_client = chromadb.Client(chromadb.config.Settings(
    chroma_db_impl="duckdb+parquet",
    persist_directory="ChromaDB"
))

while True:
    print("\nMain Menu\n-------------")
    print("1. Create a collection")
    print("2. Add documents from a folder to a collection")
    print("3. Load a collection")
    print("4. Process a PDF")
    print("5. Process a TXT")
    print("6. Delete a collection")
    print("7. Add document from file path")
    print("8. Exit")
    print("9. Chat with Athena")

    choice = input("Enter your choice: ")

    if choice == "1":
        name = input("Enter a name for the collection: ")
        if name in chroma_client.list_collections():
            collection = chroma_client.load_collection(name=name)
            print(f"{Fore.YELLOW}Collection {name} loaded.{Style.RESET_ALL}")
        else:
            collection = chroma_client.create_collection(name=name)
            print(f"{Fore.YELLOW}Collection {name} created.{Style.RESET_ALL}")
            os.makedirs(os.path.join("ChromaDB", name))

    elif choice == "2":
        if not "collection" in locals():
            print(f"{Fore.RED}Error: no collection selected.{Style.RESET_ALL}")
            continue
        folder_path = input("Enter the path to the folder containing the text files: ")
        if not os.path.isdir(folder_path):
            print(f"{Fore.RED}Error: the specified path is not a directory.{Style.RESET_ALL}")
            continue
        add_documents_from_folder(collection, folder_path)

    elif choice == "3":
        current_databases = chroma_client.list_collections()
        if len(current_databases) == 0:
            print("No collections found.")
            continue
        print(f"{Fore.YELLOW}Current collections:{Style.RESET_ALL}")
        for collection in current_databases:
            print(collection)
        name = input("Enter the name of the collection to load: ")
        try:
            collection = chroma_client.get_or_create_collection(name=name)
            print(f"{Fore.YELLOW}Collection {name} loaded.{Style.RESET_ALL}")
        except ValueError as e:
            print(f"{Fore.RED}Error: {e}{Style.RESET_ALL}")

    elif choice == "4":
        pdf_path = input("Enter the path to the PDF: ")
        get_text_from_pdf(pdf_path, unwanted_strings)

    elif choice == "5":
        txt_path = input("Enter the path to the TXT: ")
        book_title = input("Enter the title of the book: ")
        process_text(txt_path, book_title)

    elif choice == "6":
        name = input("Enter the name of the collection to delete: ")
        try:
            chroma_client.delete_collection(name=name)
            print(f"{Fore.YELLOW}Collection {name} deleted.{Style.RESET_ALL}")
        except ValueError as e:
            print(f"{Fore.RED}Error: {e}{Style.RESET_ALL}")

    elif choice == "7":
        if not "collection" in locals():
            print(f"{Fore.RED}Error: no collection selected.{Style.RESET_ALL}")
            continue
        file_path = input("Enter the path to the file: ")
        if not os.path.isfile(file_path):
            print(f"{Fore.RED}Error: the specified path is not a file.{Style.RESET_ALL}")
            continue
        add_document_from_file_path(collection, file_path)

    elif choice == "9":
        print("Chatting with Athena...")
        while True:
            if not "collection" in locals():
                print(f"{Fore.RED}Error: no collection selected.{Style.RESET_ALL}")
                continue
            prompt_text = input("What would you like to ask Athena?\n(type 'exit' to quit) ")
            if prompt_text.lower() == "exit":
                break
            # Call your classifier function here
            subject_number = evaluator(prompt_text)['subject']
            subject_prompts = get_prompts(subject_number)
            combined_text = get_context(collection, prompt_text)
            print(f"{Fore.GREEN}{combined_text}{Style.RESET_ALL}")
            response_text = prompt_athena(subject_prompts, combined_text)
            print(f"{Fore.RED}{response_text}{Style.RESET_ALL}")
            response=response_text
            print(f"{Fore.YELLOW}{critical_bard(prompt_text, response)}{Style.RESET_ALL}")


    elif choice == "8":
        print("Exiting...")
        break

    else:
        print(f"{Fore.RED}Invalid option, please try again.{Style.RESET_ALL}")
