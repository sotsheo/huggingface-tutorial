from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Cấu hình low-memory
model_name = "google/gemma-2b-it"  # Dùng bản 2B thay vì 7B
tokenizer = AutoTokenizer.from_pretrained(model_name)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto",  # Tự động nhận M1/M2 GPU
    low_cpu_mem_usage=True
)

# Hàm chat tối giản
def ask(prompt, max_tokens=200):
    inputs = tokenizer(prompt, return_tensors="pt").to("mps")  # MPS = Metal Performance Shaders
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_tokens,
        temperature=0.7
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Demo
print(ask("Cách pha cà phê ngon?"))