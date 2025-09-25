import numpy as np
import textwrap
import chromadb
from chromadb.utils import embedding_functions
import os
import re

class SEMANTIC_SEARCH_MODULE:
    def __init__(self, CHARACTER_INFORMATION_PATH,
                  simple_char_description, character_name):

        # char Name and simple char description
        self.character_name = character_name
        self.simple_char_description = simple_char_description

        # Path containing the character information files
        self.CHARACTER_INFORMATION_PATH = CHARACTER_INFORMATION_PATH
        # Create chroma to endable document search
        self.client = chromadb.Client() 

        # Embed functions into vector representation
        self.embedder = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"  # small, 384-dim, works offline
        )

        # Store different collections here
        self.collection_dict = {}

        # Create collections and fill them with data from the person files
        self._build_collections()

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
    
    def _create_empty_collection(self, collection_name):

        # create empty collection
        self.collection_dict[collection_name] = self.client.get_or_create_collection(
            name=f"{collection_name}",
            embedding_function = self.embedder
            )
        
        return None

    def _add_information_to_collection(self, collection_name, information):

        # Chunk the text
        chunks = self._sentence_chunk(information, max_len=100)
        
        # Search the matching collection
        collection = self.collection_dict[collection_name]

        # Add the chunks to the database
        for i, chunk in enumerate(chunks):
            collection.add(
                documents=[chunk],      # actual text, not the filename
                ids=[f"{collection_name}_chunk_{i}"],        # unique id for each file
                metadatas=[{"source": collection_name}]  # keep track of source file
            )

        return None

    # Get Most relevant information
    def _prompt_result(self, prompt: str, collection_name, n_results: int):

        # Get the collection to search from
        collection = self.collection_dict[collection_name]

        # Search in chromadb
        prpt_res = collection.query(query_texts=[prompt],
                                    n_results=n_results*5, include=["documents", "distances"])
        prpt_docs = prpt_res["documents"][0]
        prpt_dists = prpt_res["distances"][0]
        return prpt_docs, prpt_dists
    
    def _build_collections(self):

        # Add all character information to be accessible for the chatbot
        for file in os.listdir(self.CHARACTER_INFORMATION_PATH):
            # get filename without .txt ending
            filename = file.split(".")[0]

            # Create and add information for each text file
            if file.endswith(".txt"):

                file_path = os.path.join(self.CHARACTER_INFORMATION_PATH, f"{filename}.txt")
                # Get information from file
                with open(file_path, "r", encoding="utf-8") as f:
                    information = f.read().strip()

                self._create_empty_collection(filename)
                self._add_information_to_collection(collection_name=filename, information=information)
                print(f"{file} found and added to context")

        return None
    
    # return chosen chunks from collection
    def _search_information(self, prompt: str, collection_name: str, rng_temp=None, threshold=np.inf, n_max_results=2) -> list:

        # Get specific behavior dependent on user prompt
        search_prompt = f"""
            {prompt}
            """
    
        # Get relevant information dependent on base prompt and user input
        prompt_docs, prompt_dists = self._prompt_result(f"{search_prompt}", collection_name, n_max_results*5)

        # Use numpy for advanced array operations
        prompt_dists = np.array(prompt_dists)

        # Remove data with too high distance
        prompt_dists = np.where(prompt_dists > threshold, np.inf, prompt_dists)

        # Number of docs who survived the threshold
        num_relevant_docs = np.sum(prompt_dists < threshold)

        # Break early if no doc survived the threshold
        if num_relevant_docs == 0:
            text = " "

        # Probabilistic search if rng has a temperature
        elif rng_temp != None:
            # Softmax relevances to get probability dist
            probs = np.exp(-prompt_dists / rng_temp) / np.sum(np.exp(-prompt_dists / rng_temp))

            # Sample from probability distribution
            chosen_indices = np.random.choice(
                len(prompt_docs),        # population size
                size=n_max_results,   # how many samples you want
                replace=False,    # don’t pick the same doc twice
                p=probs           # probability distribution
            )

            chosen_docs = [prompt_docs[i] for i in chosen_indices]

            text = " ".join(chosen_docs)

        # Deterministic search
        elif rng_temp == None:
            
            # Decrease search results when threshold cancels too many items
            if num_relevant_docs <= n_max_results:
                n_results = num_relevant_docs
            else:
                n_results = n_max_results

            if n_results > 0:
                text = " ".join(prompt_docs[:n_results])
            else:
                text = " "
        
        return text

    def augment_prompt(self, prompt: str, chat_history: str) -> str:

        # Convert chat historyies list of text to one big text
        plain_txt_chat_history = "\n\n".join(chat_history).strip()

        # Add new chat details to chromadb
        self._add_information_to_collection("chat_history", "\n\n".join(chat_history[:-2]))

        # get relevant text dependent on user prompt
        retrieved_personality = self._search_information(prompt + "\n" + self.simple_char_description, "personality", rng_temp=1.0, threshold=np.inf, n_max_results=2)
        retrieved_surroundings = self._search_information(prompt, "surroundings", rng_temp=None, threshold=0.8, n_max_results=2)
        retrieved_chathistory = self._search_information(prompt, "chat_history", rng_temp=None, threshold=0.8, n_max_results=2)

        # return instruction text with relevant search information
        augmented_prompt = textwrap.dedent(f"""

{self.simple_char_description} {retrieved_personality}

Always answer in character as {self.character_name}, using the traits above to guide your style of speech. 
Do not repeat the traits directly. You only speak as {self.character_name}. Instead, embody them in how you speak.
Only use information that existed in the temporal context of the person!

{retrieved_surroundings}

Here is the ongoing conversation:
{retrieved_chathistory}

{plain_txt_chat_history}

This is the current prompt:
{prompt}

{self.character_name}:

        """)
        
        return augmented_prompt