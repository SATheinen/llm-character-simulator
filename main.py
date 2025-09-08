from llama_cpp import Llama
import chromadb
from chromadb.utils import embedding_functions
import datetime

llama_cli_path="/Users/silas/work/projects/mistral/llama.cpp/build/bin/llama-cli"
MODEL_PATH="/Users/silas/work/projects/mistral/llama.cpp/models/mistral/"
MODEL="mistral-7b-instruct-v0.1.Q4_K_M.gguf"

# Chat history save file
LOG_FILE="chat_log.txt"

# Get Mistral
llm = Llama(
    model_path=f"{MODEL_PATH}/{MODEL}",
    n_ctx=8192, # context size
    n_gpu_layers=-1, 
)

# Create chroma for RAG interactions
client = chromadb.PersistentClient(path="./chroma")

# Embed functions into vector representation
sentence_embedder = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"  # small, 384-dim, works offline
)

collection = client.get_or_create_collection(
    name="Information",
    embedding_function = sentence_embedder
)

docs = [
    "Ubisoft released Assassin’s Creed Valhalla in 2020.",
    "Far Cry 6 was released in 2021.",
    "Watch Dogs: Legion also launched in 2020."
]

collection.add(
    documents=docs,
    ids=[f"doc_{i}" for i in range(len(docs))]
)

# Store the whole coversation
chat_history = []

# Define chatbot function
def response(prompt: str) -> str:

    global chat_history
    
    # First get context from prompt
    context = collection.query(query_texts=[prompt], n_results=2) 
    
    # Explicitly mark User and Chatbot responses
    chat_history.append(f"{context} User: {prompt}")

    conversation = "\n".join(chat_history) + "\nAssistant:"    

    # Call mistral
    llm_output = llm(
        f"{conversation}",
        max_tokens=200,
        temperature=0.5,
        top_p=0.9,
    )

    reply = llm_output["choices"][0]["text"].strip()

    chat_history.append(f"Assistant: {reply}")

    # Store Chatbot reply
    with open(LOG_FILE, "a") as f:
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        f.write(f"User: {prompt}\n")
        f.write(f"Assistant: {reply}\n")

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
            print(f"Assistant response: {answer}\n")
