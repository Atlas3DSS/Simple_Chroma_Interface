import os
from PyPDF2 import PdfReader

def remove_unwanted_strings(text, unwanted_strings):
    for string in unwanted_strings:
        text = text.replace(string, '')
    return text

print("Welcome to the pdf processing tool!")
pdf_path = input("What is the path to the pdf? ")

unwanted_strings = ["This ebook belongs to William Tatum (info@atlas3dss.com),", "purchased on 14/04/2023"]

def get_text_from_pdf(pdf_path):
    txt_file_name = os.path.join(chunked_folder, pdf_path.replace('.pdf', '.txt'))
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


def chunk_text(text: str, max_chunk: int = 1000, min_chunk: int = 250, overlap: int = 150):
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

pdf_file_name = os.path.basename(pdf_path).replace('.pdf', '')
chunked_folder = f"chunked_{pdf_file_name}"
text_file_name = pdf_file_name + '.txt'
if not os.path.exists(chunked_folder):
    os.makedirs(chunked_folder)

text_path = os.path.join(chunked_folder, text_file_name)
# Rest of the code remains the same

if not os.path.isfile(text_path):
    text = get_text_from_pdf(pdf_path)
    with open(text_path, "w", encoding='utf-8', errors='replace') as f:
        f.write(text)
else:
    with open(text_path, "r", encoding='utf-8', errors='replace') as f:
        text = f.read()

chunks = []

chunk_files = [f for f in os.listdir(chunked_folder) if f.endswith(".txt") and "chunk_" in f]

if chunk_files:
    last_chunk_file = sorted(chunk_files)[-1]
    with open(os.path.join(chunked_folder, last_chunk_file), "r", encoding='utf-8', errors='replace') as f:
        last_chunk = f.read()

    last_chunk_words = last_chunk.split()
    text_words = text.split()

    if last_chunk_words and text_words and last_chunk_words[-1] == text_words[-1]:
        print("Text already chunked.")
        for chunk_file in chunk_files:
            with open(os.path.join(chunked_folder, chunk_file), "r", encoding='utf-8', errors='replace') as f:
                chunk = f.read()
            chunks.append(chunk)
    else:
        chunks = chunk_text(text)
        for i, chunk in enumerate(chunks):
            with open(os.path.join(chunked_folder, f"chunk_{i + 1}.txt"), "w", encoding='utf-8', errors='replace') as f:
                f.write(chunk)
else:
    chunks = chunk_text(text)
    for i, chunk in enumerate(chunks):
        with open(os.path.join(chunked_folder, f"chunk_{i + 1}.txt"), "w", encoding='utf-8', errors='replace') as f:
            f.write(chunk)


