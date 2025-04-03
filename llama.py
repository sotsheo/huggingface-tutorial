from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from huggingface_hub import login
login(token="")

# Kiểm tra thiết bị
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Tải model
model_name = "google/gemma-2b-it"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map=device
)

def ask(prompt, max_tokens=200):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_tokens,
        temperature=0.7
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Demo
print(ask("Cách pha cà phê ngon?"))
