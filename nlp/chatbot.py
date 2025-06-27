import gradio as gr
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load mô hình
model_name = "microsoft/DialoGPT-small"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Biến toàn cục
chat_history_ids = None

# Hàm phản hồi
def respond(user_input, history):
    global chat_history_ids

    new_input_ids = tokenizer.encode(user_input + tokenizer.eos_token, return_tensors='pt')

    bot_input_ids = torch.cat([chat_history_ids, new_input_ids], dim=-1) if chat_history_ids is not None else new_input_ids

    # Cảnh báo attention_mask có thể bỏ qua hoặc thêm vào nếu cần chính xác
    chat_history_ids = model.generate(
        bot_input_ids,
        max_length=1000,
        pad_token_id=tokenizer.eos_token_id,
        do_sample=True,
        top_k=50,
        top_p=0.95
    )

    bot_response = tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)

    return bot_response

# Giao diện Gradio
iface = gr.ChatInterface(
    fn=respond,
    title="🤖 Chatbot GPT (DialoGPT)",
    examples=["Hello!", "Tell me a joke", "Who are you?", "I feel sad..."]
)

# Chạy
if __name__ == "__main__":
    iface.launch(share=True)  # thêm share=True để tạo public link
