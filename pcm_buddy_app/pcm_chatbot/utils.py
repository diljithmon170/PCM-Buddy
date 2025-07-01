# utils.py
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("google/gemma-2b-it")
model = AutoModelForCausalLM.from_pretrained(
    "google/gemma-2b-it",
    device_map="auto",
    torch_dtype=torch.float16,
    )

def get_answer(query, subject):
    prompt = f"Answer this class 10 {subject} question in a simple way:\n\n{query}\n\nAnswer:"
    
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=200,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            temperature=0.7
        )

    answer = tokenizer.decode(output[0], skip_special_tokens=True)
    return answer.split("Answer:")[-1].strip()
