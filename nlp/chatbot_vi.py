import gradio as gr
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# BEST OPTIONS - Nhẹ và tốt hơn GPT2

# OPTION 1: Arcee-VyLinh 3B (MỚI NHẤT 2024 - KHUYÊN DÙNG)
# model_name = "arcee-ai/Arcee-VyLinh"  # 3B params, rất tốt cho tiếng Việt

# OPTION 2: Qwen2.5 0.5B (Rất nhẹ, đa ngôn ngữ tốt)
model_name = "Qwen/Qwen2.5-0.5B-Instruct"

# OPTION 3: Qwen2.5 1.5B (Cân bằng tốt)
# model_name = "Qwen/Qwen2.5-1.5B-Instruct"

# OPTION 4: Microsoft Phi-3 Mini (3.8B - nhẹ và mạnh)
# model_name = "microsoft/Phi-3-mini-4k-instruct"

# OPTION 5: TinyLlama (1.1B - rất nhẹ)
# model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

print(f"Loading model: {model_name}")
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_name, 
    torch_dtype=torch.float16,  # Giảm RAM
    trust_remote_code=True
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

def respond_vi(prompt, history):
    # Format theo chuẩn chat template
    messages = [
        {"role": "system", "content": "Bạn là một trợ lý AI thông minh, hữu ích và thân thiện. Hãy trả lời bằng tiếng Việt một cách tự nhiên và chính xác."},
        {"role": "user", "content": prompt}
    ]
    
    # Áp dụng chat template nếu có
    if hasattr(tokenizer, 'apply_chat_template'):
        formatted_prompt = tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
    else:
        formatted_prompt = f"System: Bạn là một trợ lý AI thông minh, hữu ích và thân thiện. Hãy trả lời bằng tiếng Việt một cách tự nhiên và chính xác.\nUser: {prompt}\nAssistant:"
    
    input_ids = tokenizer.encode(formatted_prompt, return_tensors="pt")
    
    # Generation với parameters tối ưu
    with torch.no_grad():
        output = model.generate(
            input_ids,
            max_new_tokens=100,      # Chỉ sinh token mới
            do_sample=True,
            top_k=40,
            top_p=0.85,
            temperature=0.7,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    # Chỉ lấy phần response mới
    new_tokens = output[0][len(input_ids[0]):]
    result = tokenizer.decode(new_tokens, skip_special_tokens=True)
    
    # Clean up response
    result = result.strip()
    
    # Loại bỏ các artifacts thường gặp
    cleanup_patterns = ["Assistant:", "User:", "System:", "<|im_end|>", "<|im_start|>"]
    for pattern in cleanup_patterns:
        if pattern in result:
            result = result.split(pattern)[0].strip()
    
    return result

# UI tương thích với Gradio cũ hơn
demo = gr.ChatInterface(
    fn=respond_vi,
    title="🤖 Chatbot Việt Thông Minh",
    description="Sử dụng model hiện đại, nhẹ và tối ưu cho tiếng Việt",
    examples=[
        "Xin chào! Bạn có thể giúp tôi gì?",
        "Giải thích về trí tuệ nhân tạo", 
        "Cho tôi lời khuyên học tiếng Anh",
        "Kể một câu chuyện ngắn hay",
        "Thời tiết hôm nay thế nào?"
    ]
)

if __name__ == "__main__":
    demo.launch(
        share=True,
        show_error=True
    )