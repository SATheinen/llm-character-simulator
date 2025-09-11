import numpy as np
import textwrap
import chromadb
from chromadb.utils import embedding_functions
import os
import re

class RAG_MODULE:
    def __init__(self, CHARACTER_INFORMATION_PATH,
                  simple_char_description, character_name):

        # char Name and simple char description
        self.character_name = character_name
        self.simple_char_description = simple_char_description

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
                        ids=[f"{file}_chunk_{i}"],        # unique id for each file
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
            {self.simple_char_description}
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

{self.simple_char_description} {retrieved_text}

Always answer in character as {self.character_name}, using the traits above to guide your style of speech. 
Do not repeat the traits directly. You only speak as {self.character_name}. Instead, embody them in how you speak.
Only use information that existed in the temporal context of the person!

Here is the ongoing conversation:
{chat_history}

This is the current prompt:
{prompt}

{self.character_name}:

        """)
        
        return augmented_prompt