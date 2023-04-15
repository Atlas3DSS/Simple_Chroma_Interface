import openai
from dotenv import load_dotenv
from tiktoken import Tokenizer
from tiktoken.encoding import Encoding
import os

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
openai.api_key = OPENAI_API_KEY
print("OPENAI_API_KEY:", OPENAI_API_KEY)
print("Welcome to the summarizer tool!")

chunk_dir_path = input("What is the path to the directory of chunks to be summarized? ")
summarized_path = chunk_dir_path.replace('chunked', 'summarized')
if not os.path.exists(summarized_path):
    os.makedirs(summarized_path)
##chunks can be found in the chunked folder from input create a function to load those chunks for get_chatGPT_response

def load_chunks(chunk_dir_path):
    chunks = []
    for chunk_file in os.listdir(chunk_dir_path):
        with open(os.path.join(chunk_dir_path, chunk_file), 'r', encoding="utf-8", errors="replace") as f:
            chunk = f.read()
        chunks.append(chunk)
    return chunks


def get_chatGPT_response(chunk: str, previous_summary: str) -> str:
    tokenizer = Tokenizer()
    enc = Encoding.for_model("gpt-3.5-turbo")
    combined_text = f'{previous_summary} {chunk}'
    
    token_count = len(tokenizer.tokenize(combined_text, enc))
    max_tokens = 4096 - 10  # Subtract some tokens for the system message and formatting

    if token_count > max_tokens:
        tokens = tokenizer.tokenize(combined_text, enc)[:max_tokens]
        trimmed_text = enc.detokenize(tokens)
    else:
        trimmed_text = combined_text

    messages = [
        {
            'role': 'system',
            'content': f'You are a highly intelligent AI language model. Please summarize the following text in a clear and verbose manner. Your summaries will be used to determine if a given chunk is relevant to a user, so please be as verbose as possible. Also, you are summarizing a chunk of text, not a question. There is overlap between chunks, so please do not repeat yourself, check your previous summary to make sure you are not repeating yourself: {previous_summary} {chunk}'
        },
        {
            'role': 'user',
            'content': trimmed_text
        }
    ]
    response = openai.ChatCompletion.create(
        model='gpt-3.5-turbo',
        messages=messages,
    )

    return response.choices[0].message.content


previous_summary = ""
for chunk_file in os.listdir(chunk_dir_path):
    with open(os.path.join(chunk_dir_path, chunk_file), 'r', encoding="utf-8", errors="replace") as f:
        chunk = f.read()

    summary = get_chatGPT_response(chunk, previous_summary)
    previous_summary = summary

    # Update the output file name to include the "summarized_" prefix
    summarized_chunk_file = chunk_file.replace("chunk_", "summarized_chunk_")

    with open(os.path.join(summarized_path, summarized_chunk_file), 'w', encoding="utf-8", errors="replace") as f:
        f.write(summary)

print("Summarization complete!")