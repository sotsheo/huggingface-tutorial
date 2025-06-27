import gradio as gr
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load mô hình và tokenizer từ Hugging Face
model_name = "microsoft/DialoGPT-small"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Tạo biến toàn cục để lưu lịch sử hội thoại
chat_history_ids = None

def respond(user_input, chat_history=[]):
    global chat_history_ids

    # Tokenize user input
    new_input_ids = tokenizer.encode(user_input + tokenizer.eos_token, return_tensors='pt')

    # Ghép với lịch sử chat nếu có
    if chat_history_ids is not None:
        bot_input_ids = torch.cat([chat_history_ids, new_input_ids], dim=-1)
    else:
        bot_input_ids = new_input_ids

    # Sinh phản hồi từ model
    chat_history_ids = model.generate(
        bot_input_ids,
        max_length=1000,
        pad_token_id=tokenizer.eos_token_id,
        do_sample=True,
        top_k=50,
        top_p=0.95
    )

    # Decode response
    bot_response = tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)

    # Thêm vào lịch sử
    chat_history.append((user_input, bot_response))
    return chat_history, chat_history

# Tạo giao diện với Gradio
chatbot = gr.ChatInterface(
    fn=respond,
    title="🤖 Chatbot GPT (DialoGPT)",
    theme="compact",
    examples=["Hi", "Tell me a joke", "What's your name?", "I feel sad..."],
    retry_btn="🔁 Thử lại",
    undo_btn="↩️ Quay lại",
    clear_btn="🗑️ Xoá"
)

# Khởi chạy
if __name__ == "__main__":
    chatbot.launch()
