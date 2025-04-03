from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Tải mô hình GPT-2 và tokenizer từ Hugging Face
model_name = "gpt2"
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

# Đặt mô hình vào chế độ đánh giá
model.eval()

# Hàm xử lý câu hỏi và tạo câu trả lời
def generate_answer(input_text):
    # Mã hóa văn bản đầu vào
    inputs = tokenizer.encode(input_text, return_tensors="pt")

    # Sinh ra văn bản tiếp theo từ mô hình
    output = model.generate(inputs, max_length=100, num_return_sequences=1, no_repeat_ngram_size=2)

    # Giải mã và trả về văn bản đầu ra
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    return generated_text

# Chạy chatbot
def chatbot():
    print("Chatbot: Chào bạn! Tôi có thể giúp gì cho bạn?")
    
    while True:
        # Nhận câu hỏi từ người dùng
        user_input = input("Bạn: ")
        
        # Kiểm tra nếu người dùng muốn thoát
        if user_input.lower() == "thoát":
            print("Chatbot: Tạm biệt!")
            break
        
        # Sinh câu trả lời từ mô hình
        response = generate_answer(user_input)
        
        # In câu trả lời chatbot
        print(f"Chatbot: {response}")

# Khởi chạy chatbot
chatbot()
