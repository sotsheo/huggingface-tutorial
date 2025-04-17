# -*- coding: utf-8 -*-
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import time
import psutil

# 1. Kiểm tra hệ thống
def system_check():
    print("=== Thông tin hệ thống ===")
    print(f"CPU: {psutil.cpu_count()} cores")
    print(f"RAM: {psutil.virtual_memory().total / (1024**3):.2f} GB")
    print(f"PyTorch version: {torch.__version__}")
    print(f"MPS available: {torch.backends.mps.is_available()}")
    print(f"Device: {torch.device('mps')}")

system_check()

# 2. Tải model và tokenizer (Phi-2 2.7B)
MODEL_NAME = "microsoft/phi-2"

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True
).to('mps')

# 3. Hàm generate tối ưu cho Phi-2
def phi2_generate(prompt, max_length=200):
    inputs = tokenizer(prompt, return_tensors="pt").to('mps')
    
    start_time = time.time()
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=max_length,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    generation_time = time.time() - start_time
    output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Benchmark
    tokens = len(tokenizer.encode(output_text))
    print(f"\n[Performance] {tokens/generation_time:.2f} tokens/s | {generation_time:.2f}s")
    
    return output_text

# 4. Chatbot interface
def chat():
    print("\n=== Phi-2 Chatbot trên Mac ===")
    print("Nhập 'exit' để thoát\n")
    
    while True:
        user_input = input("You: ")
        if user_input.lower() == 'exit':
            break
            
        # Thêm prompt engineering cho Phi-2
        prompt = f"""Instruct: {user_input}
Output:"""
        
        response = phi2_generate(prompt)
        print("\nAI:", response.split("Output:")[-1].strip())
        print("-" * 50)

# 5. Giám sát tài nguyên
def monitor():
    import threading
    
    def resource_monitor():
        while True:
            cpu = psutil.cpu_percent()
            mem = psutil.virtual_memory().percent
            print(f"\n[Monitor] CPU: {cpu}% | RAM: {mem}%")
            time.sleep(3)
    
    thread = threading.Thread(target=resource_monitor, daemon=True)
    thread.start()

if __name__ == "__main__":
    monitor()
    chat()