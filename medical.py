# -*- coding: utf-8 -*-
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import warnings
warnings.filterwarnings("ignore")

# 1. Load dữ liệu từ CSV
def load_data(csv_path):
    df = pd.read_csv(csv_path)
    qa_pairs = {row['question']: row['answer'] for _, row in df.iterrows()}
    return qa_pairs

# 2. Load model (phi-2 nhẹ cho CPU/GPU yếu)
def load_model():
    model_name = "microsoft/phi-2"
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    return model, tokenizer

# 3. Tìm câu trả lời phù hợp
def find_best_answer(question, qa_pairs, model, tokenizer):
    # Ưu tiên tìm trong CSDL trước
    for q, a in qa_pairs.items():
        if question.lower() in q.lower():
            return a
    
    # Nếu không có thì dùng model generate
    prompt = f"""Bạn là bác sĩ ảo. Hãy trả lời ngắn gọn, tự nhiên:
Câu hỏi: {question}
Trả lời:"""
    
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        max_length=200,
        temperature=0.7,
        do_sample=True
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True).split("Trả lời:")[-1].strip()

# 4. Giao diện command line
def chat_interface(csv_path):
    print("Đang tải dữ liệu y tế...")
    qa_pairs = load_data(csv_path)
    
    print("Đang khởi động model AI...")
    model, tokenizer = load_model()
    
    print("\n=== CHATBOT Y TẾ ===")
    print("(Nhập 'exit' để thoát)\n")
    
    while True:
        try:
            user_input = input("Bạn: ")
            if user_input.lower() == 'exit':
                break
                
            answer = find_best_answer(user_input, qa_pairs, model, tokenizer)
            print("\nBot:", answer)
            print("-" * 50)
            
        except KeyboardInterrupt:
            break

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Cách dùng: python medical_chatbot.py [đường_dẫn_file.csv]")
    else:
        chat_interface(sys.argv[1])