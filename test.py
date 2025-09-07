from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Model repo (official from Hugging Face)
model_id = "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"

# Load tokenizer + model
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="mps",       # Use Apple Silicon GPU
    torch_dtype=torch.float16
)

# Test prompt
prompt = "You are a teacher. Explain prompt engineering to a beginner in 3 sentences. Keep it simple and clear."

# Tokenize + run on model
inputs = tokenizer(prompt, return_tensors="pt").to("mps")
outputs = model.generate(
    **inputs,
    max_new_tokens=150,
    do_sample=True,
    temperature=0.7,
    top_p=0.9
)

print(tokenizer.decode(outputs[0], skip_special_tokens=True))
