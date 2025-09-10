from llama_cpp import Llama
import chromadb
from chromadb.utils import embedding_functions
import datetime
import os
import re
import numpy as np
import textwrap

llama_cli_path="/Users/silas/work/projects/mistral/llama.cpp/build/bin/llama-cli"
MODEL_PATH="/Users/silas/work/projects/mistral/llama.cpp/models/mistral/"
MODEL="mistral-7b-instruct-v0.1.Q4_K_M.gguf"

# Path to the RAGable content
CHARACTER_INFORMATION_PATH="./character_information"

# Chat history save file
LOG_FILE="chat_log.txt"

class RAG_MODULE():
    def __init__(self, CHARACTER_INFORMATION_PATH,
                  char_description, character_name):

        # Name and simple char description
        self.character_name = character_name
        self.char_description = char_description

        # Path containing the character information files
        self.CHARACTER_INFORMATION_PATH = CHARACTER_INFORMATION_PATH
        # Create chroma for RAG interactions
        self.client = chromadb.PersistentClient(path="./chroma") 

        # Embed functions into vector representation
        self.embedder = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"  # small, 384-dim, works offline
        )

        # create empty collection
        self.collection = self.client.get_or_create_collection(
            name="Information",
            embedding_function = self.embedder
        )

        # add information from CHARACTER_INFORMATION_PATH to collection
        self._add_information_to_collection()

    # Chunk information from text files
    def _sentence_chunk(self, text: str, max_len: int) -> str:
        sentences = re.split(r'(?<=[.!?])\s+|\n+', text)
        chunks, current = [], ""

        for sentence in sentences:
            # continue adding sentences
            if len(current) + len(sentence) < max_len:
                current += " " + sentence
            else:
                # Add overlap and save to chunks
                current += " " + sentence
                chunks.append(current)

                # create next chunk
                current = ""
                current += " " + sentence
        
        return chunks

    def _add_information_to_collection(self):
        # Add all character information to be accessible for the chatbot
        for i, file in enumerate(os.listdir(self.CHARACTER_INFORMATION_PATH)):
            # Only add text files
            if file.endswith(".txt"):
                print(f"{file} found and added to context")
                # Search for all available files 
                file_path = os.path.join(self.CHARACTER_INFORMATION_PATH, file)
                # Chunk the text
                with open(file_path, "r", encoding="utf-8") as f:
                    chunks = self._sentence_chunk(f.read().strip(), max_len=100)

                # Add the chunks to the database
                for i, chunk in enumerate(chunks):
                    self.collection.add(
                        documents=[chunk],      # actual text, not the filename
                        ids=[f"char_{i}"],        # unique id for each file
                        metadatas=[{"source": file}]  # optional: keep track of source file
                    )
        return None

    # Get Most relevant information
    def _prompt_result(self, prompt: str, n_results: int):
        prpt_res = self.collection.query(query_texts=[prompt],
                                    n_results=n_results*5, include=["documents", "distances"])
        prpt_docs = prpt_res["documents"][0]
        prpt_dists = prpt_res["distances"][0]
        return prpt_docs, prpt_dists
    
    # return chosen chunks from collection
    def _RAG_information(self, prompt: str, n_results: int, temp=1.0) -> list:
        # Always include basic character behavior
        base_prompt = f"""
            {self.char_description}
            """
        # Get specific behavior dependent on user prompt
        user_prompt = f"""
            {prompt}
            """
    
        # Get relevant information dependent on base prompt and user input
        base_prpt_docs, base_prpt_dists = self._prompt_result(f"{base_prompt}", n_results=n_results*5)
        user_prpt_docs, user_prpt_dists = self._prompt_result(f"{user_prompt}", n_results=n_results*5)    

        # forge together most relevant information in [(doc, dist)] pairs
        pairs = (list(zip(base_prpt_docs, base_prpt_dists))
                + list(zip(user_prpt_docs, user_prpt_dists)))
        
        # sort by distance and reduce to n_results
        sorted_pairs = sorted(pairs, key=lambda x: x[1])
        most_relevant_pairs = sorted_pairs[:n_results*5]

        # Zip back to dist and docs
        docs, dists = zip(*most_relevant_pairs)
        dists = np.array(list(dists))
        docs = list(docs)

        # More relevant information has lower distances
        relevance = 1 / dists

        # Softmax relevances and get probability dist
        probs = np.exp(relevance / temp) / np.sum(np.exp(relevance / temp))
        print(probs)

        # Sample from probability distribution
        chosen_indices = np.random.choice(
            len(docs),        # population size
            size=n_results,   # how many samples you want
            replace=False,    # don’t pick the same doc twice
            p=probs           # probability distribution
        )

        chosen_docs = [docs[i] for i in chosen_indices]
        unique_docs = np.unique(chosen_docs).tolist()

        text = " ".join(unique_docs)
        
        return text

    def augment_prompt(self, prompt: str, chat_history: str, n_results: int) -> str:

        # get relevant text dependent on user prompt
        retrieved_text = self._RAG_information(prompt, n_results, temp=1.0)

        # return intruction text with relevant RAGed information
        augmented_prompt = textwrap.dedent(f"""

{self.char_description} {retrieved_text}

Always answer in character as {self.character_name}, using the traits above to guide your style of speech. 
Do not repeat the traits directly. You only speak as {self.character_name}. Instead, embody them in how you speak.

Here is the ongoing conversation:
{chat_history}

This is the current prompt:
{prompt}

{character_name}:

        """)
        
        return augmented_prompt
            
# Delete old Logfile before starting a fresh chatbot run
if os.path.exists(LOG_FILE):
    print("Deleted old Chat History!")
    os.remove(LOG_FILE)

# Store the whole coversation
chat_history = []

character_name = "Jack Sparrow"
char_description = "You are Jack Sparrow, a pirate captain"
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
    max_turns = 2
    if (len(chat_history) // 2) > max_turns:
        chat_history = chat_history[2:]
    print(f"Chat History length: {len(chat_history)}")

    # Convert chat historyies list of text to one big text
    plain_txt_chat_history = "\n\n".join(chat_history).strip()
    # Create an enriched prompt with specific instructions for the llm
    enriched_prompt = rag_module.augment_prompt(prompt, plain_txt_chat_history, n_results=1)

    print()
    print("################################")
    print(f"Enriched Prompt:\n{enriched_prompt}")
    print("################################")
    print()

    # Call mistral
    llm_output = llm(
        f"{enriched_prompt}",
        max_tokens=200,
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
