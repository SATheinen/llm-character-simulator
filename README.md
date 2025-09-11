# 🧠 LLM Character Simulator – A Playground for Prompt Engineering

This repository is a hands-on exploration of **prompt engineering** using local LLMs. It simulates dynamic, persona-consistent conversations with fictional characters like **Jack Sparrow** using:

- 🧠 Instruction-based prompt design
- 📚 Retrieval-Augmented Generation (RAG)
- 💬 Local inference with **Mistral 7B Instruct** and **llama.cpp**

---

## 📌 Introduction

In recent years, LLMs have gained massive attention and quickly found their way into daily life. Since ChatGPT’s breakout success, individuals and companies alike have raced to build and fine-tune their own models.

But improving the model is only part of the equation — how you **use** the model is just as important.

**Prompt engineering** is the craft of structuring inputs to guide language models toward better, more controllable outputs. It started with basic tricks like prompt formatting but has evolved into advanced techniques such as:
- RAG (Retrieval-Augmented Generation)
- Context injection via external files
- Instruction tuning and persona grounding

Because prompt engineering is **flexible** and **lightweight** it opens the door to customizing behavior without costly fine-tuning.

This project is my personal playground to experiment with these techniques — especially for creating **dynamic, in-character conversations** for gaming and narrative use cases.

---

## ⚙️ Installation

1. Clone the repository  
```bash
git clone git@github.com:SATheinen/llm_character_simulator.git
cd llm_character_simulator
```
2. Install llama.cpp from here
```bash
git@github.com:ggml-org/llama.cpp.git
```

3. Install mistral 7B instruct (or another llm model) in GGUF format
```bash
cd llama.cpp/models
mkdir mistral
cd mistral
```
place mistral 7B instruct inside, you can get it from here:
```
https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF
```

4. Initialise and activate a python venv inside `llm_character_simulator` 
```bash
`python -m venv mistral_env`
`source mistral_env/bin/activate`
`pip install -r requirements.txt`
```

## Usage
Now you can run the model with python main.py and write with the chatbot in the Terminal.

You can switch to another character after creating the required files by changing the following variable inside main.py.
```python
`CHARACTER_INFORMATION_PATH = "./char_info/jack_sparrow"` 
```

**Example**
You: who are you?
Jack Sparrow: “Ahoy there, matey! I be Captain Jack Sparrow of the Black Pearl. Aye, a pirate’s life may not always bring you riches or honor but it sure does make for an exciting adventure.”

You: May I join your crew?
Jack Sparrow: “Aye, matey! The Black Pearl always be looking for new blood to join our crew. But ye must prove yourself worthy first.”

You: How should I prove myself to you?
Jack Sparrow: “Aye, matey! Prove yourself to me by demonstrating your wit and courage. Show that ye can handle the rough seas of being a pirate with grace and charm.”

You: Nothing more easy then that. Lets start our adventure! Where do we go?
Jack Sparrow: “Aye, matey! Now that ye have proven yourself worthy to join my crew, we be setting sail on a grand adventure across the seven seas. Our destination? The legendary island of Skull Isle – home to treasure beyond imagination and danger at every turn.”

### Adding a new character
Adding a different character is easily possible with a minimal setup.

```
char_info/
└── johnny_depp/
    ├── base_information.json
    └── personality.txt
```

You can always follow the example of the project/char_info/jack_sparrow folder. 

1. add a new character folder in char_info
2. add a base_information.json file
```json
{
  "name": "Johnny Depp",
  "describtion": "Johnny Depp is an actor known for his diverse roles in movies like Pirates of the Caribbean and Edward Scissorhands."
}
```
3. add a personality.txt text file with sentences describing your character, no specific format needed
4. change character path inside `main.py`
```python
`CHARACTER_INFORMATION_PATH = "./char_info/new_character"` 
```

## ⭐ Features Summary
🎭 Prompt-engineered character behavior
🔁 RAG using ChromaDB + Sentence Transformers
💬 In-character conversations with contextual memory
🧱 Modular and extensible Python architecture
📦 Fully local, private, and offline-capable

## Prompt Engineering Design
This prototype focuses on character personality and conversation style. All character information is stored externally in plain .txt files. The RAG_MODULE handles:
Chunking personality data
Embedding and storing in a ChromaDB collection
Retrieving the most relevant chunks based on the user’s prompt
If a prompt doesn’t match any specific personality chunk, the fallback is the base character description, such as:
"You are Jack Sparrow, a pirate captain..."
To simulate probabilistic memory recall, we convert relevance distances into a softmax distribution and randomly sample from it — useful for diversity in small models like Mistral-7B.
For other types of context (e.g. location-based memory), a slightly different RAG strategy using thresholds might be necessary to avoid injecting irrelevant information.
Overall, RAG allows small language models to “outsource” long-term memory and behavioral consistency efficiently.

## File structure overview
```
├── char_info
│   └── jack_sparrow
├── chat_log.txt
├── chroma
├── llama.cpp
│   └── models
│       └── mistral
│           └── mistral-7b-instruct-v0.1.Q4_K_M.gguf
├── main.py
├── mistral-env
│   ├── bin
│   ├── include
│   ├── lib
│   ├── pyvenv.cfg
│   └── share
├── rag_module.py
├── README.md
└── requirements.txt
```

## Dependencies & Acknowledgments

This project uses the following open-source tools:

- [Mistral-7B-Instruct](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.1) – Licensed under Apache 2.0
- [llama.cpp](https://github.com/ggerganov/llama.cpp) – Licensed under MIT
- [ChromaDB](https://github.com/chroma-core/chroma) – Licensed under Apache 2.0
- [sentence-transformers](https://www.sbert.net/) – Licensed under Apache 2.0

These tools are not distributed in this repo; users must install/download them separately.

## 👤 Author

**Silas Theinen**
🔗 [GitHub](https://github.com/SATheinen)  
📫 Reach me on [LinkedIn](https://www.linkedin.com/in/silas-theinen-058977358)  