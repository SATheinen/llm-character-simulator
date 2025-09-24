from llama_cpp import Llama
import chromadb
from chromadb.utils import embedding_functions
import datetime
import os
import re
import numpy as np
import textwrap
import json

import rag_module
print(rag_module.__file__)

# Get the ragmodule from another python file
from rag_module import RAG_MODULE

llama_cli_path="/Users/silas/work/projects/mistral/llama.cpp/build/bin/llama-cli"
MODEL_PATH="/Users/silas/work/projects/mistral/llama.cpp/models/mistral/"
MODEL="mistral-7b-instruct-v0.1.Q4_K_M.gguf"

# Path to the character information with RAGable content
CHARACTER_INFORMATION_PATH="./char_info/jack_sparrow"

# Chat history save file
LOG_FILE=f"{CHARACTER_INFORMATION_PATH}/chat_history.txt"
            
# Delete old Logfile before starting a fresh chatbot run
if os.path.exists(LOG_FILE):
    print("Deleted old Chat History!")
    os.remove(LOG_FILE)

# Create clean log file
with open(LOG_FILE, "w", encoding="utf-8") as f:
    f.write("Conversation Start \n\n")

# Store the whole coversation
chat_history = []

# Get basic information of the character
base_info_path = f"{CHARACTER_INFORMATION_PATH}/base_information.json"
with open(base_info_path, 'r', encoding="utf-8") as f:
    base_info = json.load(f)

character_name = base_info["name"]
char_description = base_info["simple_description"]

# Initialize a rag_module 
rag_module = RAG_MODULE(CHARACTER_INFORMATION_PATH, char_description, character_name)

# Get Mistral
llm = Llama(
    model_path=f"{MODEL_PATH}/{MODEL}",
    n_ctx=8192, # context size
    n_gpu_layers=-1,
    verbose=False
)

# Define chatbot function
def response(prompt: str) -> str:

    global chat_history
    
    # How many turns to store in chat history [User:, Chatbot:, ... max_turns times]
    max_turns = 1
    if (len(chat_history) // 2) > max_turns:
        chat_history = chat_history[2:]

    # Create an enriched prompt with specific instructions for the llm
    enriched_prompt = rag_module.augment_prompt(prompt, chat_history)

    print()
    print("################################")
    print(f"Enriched Prompt:\n{enriched_prompt}")
    print("################################")
    print()

    # Call mistral
    llm_output = llm(
        f"{enriched_prompt}",
        max_tokens=400,
        temperature=0.7,
        top_p=0.9,
        repeat_penalty=1.3,   # discourage repeating exact tokens
        frequency_penalty=0.9,  # discourage repeating phrases
        presence_penalty=0.6,    # encourage introducing new ideas
        stop=["\n","\n\n"],  # stop sequences where generation cuts off
    )

    reply = llm_output["choices"][0]["text"].strip()

    # Explicitly mark User and Chatbot responses
    chat_history.append(f"{prompt}")
    chat_history.append(f"{character_name}: {reply}")
    
    # Store Chatbot reply
    with open(LOG_FILE, "a") as f:
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        f.write(f"User: {prompt}\n")
        f.write(f"{character_name}: {reply}\n")

    return reply

# Start conversation
if __name__ == "__main__":
    print("Chatbot ready. Type 'exit' to quit.\n")
    while True:
        prompt = input("You: ")
        if prompt == "exit":
            print("Bye")
            break
        else:
            answer = response(prompt)
            print(f"{character_name}: {answer}\n")
