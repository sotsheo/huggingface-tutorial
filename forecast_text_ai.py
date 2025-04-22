# Sử dụng lượng tử hóa để tăng tốc độ
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch.quantization

model = GPT2LMHeadModel.from_pretrained("NlpHUST/gpt2-vietnamese")
tokenizer = GPT2Tokenizer.from_pretrained("NlpHUST/gpt2-vietnamese")

# Lượng tử hóa mô hình
quantized_model = torch.quantization.quantize_dynamic(
    model, {torch.nn.Linear}, dtype=torch.qint8
)

# Sử dụng mô hình đã lượng tử hóa
input_text = "Công nghệ AI đang"
inputs = tokenizer(input_text, return_tensors="pt")
with torch.no_grad():
    outputs = quantized_model.generate(**inputs, max_length=50)

print(tokenizer.decode(outputs[0], skip_special_tokens=True))