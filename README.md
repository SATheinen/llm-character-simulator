# Introduction
In the past few years LLM's gained broad attention and people rapidly adapted to integrate LLM's into their lifes. Since ChatGPT became famous, basically over night, many people and compaies pushed forward to create their own language model, inventing new techiques and fine tuning already existing models. But its not only important to enhance your tool, but also how to use it. Prompt engineering is a technique allowing to create better LLM responses using the same language model – starting with simple ideas like structuring your prompt better, prompt engineering fastly develveloped into using more complex approaches including external files as memory. Being more flexible and lightweight than fine tuning, enabling
different Chat personalities using the same language model, thats prompt engineering. Prompt engineering is an interesting, yet broadly applicable technique. For this reason I want to learn by doing and exactly that is what this repository is, a playground.

# Installation
1. clone the repository `git clone git@github.com:SATheinen/mistral_prompt_engineering.git`
2. Install llama.cpp inside `mistral_prompt_engineering`
3. Install mistral 7B instruct (or another llm model) inside
mistral_prompt_engineering/llama.cpp/models/mistral
4. Initialise a python venv inside `mistral_prompt_engineering` by typing `python -m venv mistral_env` in your terminal
5. activate the venv `source mistral_env/bin/activate`
6. and install the python packages `pip install -r requirements.txt`

# Usage
Now you can run the model with `python main.py` and write with the chatbot in the Terminal.

You can switch to another character after creating the required files by changing the variable `CHARACTER_INFORMATION_PATH="./char_info/jack_sparrow"` inside `main.py`. 

**Example**
You: who are you?
Jack Sparrow: “Ahoy there, matey! I be Captain Jack Sparrow of the Black Pearl. Aye, a pirate’s life may not always bring you riches or honor but it sure does make for an exciting adventure.”

You: May I join your crew?
Jack Sparrow: “Aye, matey! The Black Pearl always be looking for new blood to join our crew. But ye must prove yourself worthy first.”

You: How should I prove myselfe to you?
Jack Sparrow: “Aye, matey! Prove yourself to me by demonstrating your wit and courage. Show that ye can handle the rough seas of being a pirate with grace and charm.”

You: Nothing more easy then that. Lets start our adventure! Where do we go?
Jack Sparrow: “Aye, matey! Now that ye have proven yourself worthy to join my crew, we be setting sail on a grand adventure across the seven seas. Our destination? The legendary island of Skull Isle – home to treasure beyond imagination and danger at every turn.”

# Adding a new character
Adding a different character is easily possble with a minimal setup.

You can always follow the example of the `project/char_info/jack_sparrow` folder. First you need to create a new folder in the `project/char_info` directory, for example `johnny_depp`. Then you need to create a `base_information.json` file that contains a the characters name (Johhny Depp) and a small sentence to descripe this character (Johnny Depp an actor who played in many famous movies) in json format. Then finally you need to create a `personality.txt` file, just dump descriptive text about the character inside, no special format needed.

# Features summary
1. Prompt-engineered character behavior
2. RAG using ChromaDB and sentence-transformers
3. In-character conversation with dynamic context
4. Easily add your own characters
5. Fully local setup using llama.cpp and Mistral-7B-Instruct
6. Modular, extendable Python code

# Prompt Engineering Design
Currently this project only features character personalities. However, the code is easily exdentable to handle different kinds of RAGable information such as background story, the conversation history or locational information. 

All character information is stored externally in text form. The RAG_MODULE reads this data in and chunks the information. These chunks build the database from which we later RAG the important parts based on the user prompt if relevant. If the user prompt doesn't relate to anything from the character personality, we RAG by using the simple character description, for example: "You are Jack Sparrow, a pirate captain". To spice things a bit up (and cover the limitations of a small language model) we transform the most relevant chunks into a probability distribution adn draw the chunks from it. However, if we for example want to RAG locational information this trick wouldn't work, because sometimes the prompt doesnt't relate to the location. In this case we need to set a threshold and compare it with the distance metric to check wether the information should be included or not. In general RAG is very powerfull to gain controll about what the language model has access to and how it behaves. Especially for small language models RAG can outsource most information and helps to focus only on the important part, while lowering computational cost.

# File structure overview
>├── char_info
>│   └── jack_sparrow
>├── chat_log.txt
>├── chroma
>├── llama.cpp
>│   └── models
>│       └── mistral
>│           └── mistral-7b-instruct-v0.1.Q4_K_M.gguf
>├── main.py
>├── mistral-env
>│   ├── bin
>│   ├── include
>│   ├── lib
>│   ├── pyvenv.cfg
>│   └── share
>├── rag_module.py
>├── README.md
>└── requirements.txt

# How it works
The RAG module first gets the plain text file input from the character information. Then it creates a Chromadb with embedder function. Now the character information is chunked and saved in chromadb. We can call augment_prompt now from the main script and give it the raw user prompt together with the chat history and the number of chunks we want to retrieve from the character information. augment_prompt will forge everything together for us creating the prompt we feed to the llm.

# Dependencies & Acknowledgments

This project uses the following open-source tools:

- [Mistral-7B-Instruct](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.1) – Licensed under Apache 2.0
- [llama.cpp](https://github.com/ggerganov/llama.cpp) – Licensed under MIT
- [ChromaDB](https://github.com/chroma-core/chroma) – Licensed under Apache 2.0
- [sentence-transformers](https://www.sbert.net/) – Licensed under Apache 2.0

These tools are not distributed in this repo; users must install/download them separately.

## 👤 Author

**Silas Theinen**  
Prompt Engineer & LLM Enthusiast  
🔗 [GitHub](https://github.com/SATheinen)  
📫 Reach me on [LinkedIn](https://www.linkedin.com/in/silas-theinen-058977358)  