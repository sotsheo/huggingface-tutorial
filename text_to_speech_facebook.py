import torch
from transformers import VitsModel, AutoTokenizer
import soundfile as sf

# Load model and tokenizer
model = VitsModel.from_pretrained("facebook/mms-tts-vie")
tokenizer = AutoTokenizer.from_pretrained("facebook/mms-tts-vie")

# Input text
text = "RAM APACER NOX 16GB DDR5 Bus 5200Mhz – Black là mẫu RAM cao cấp hội tụ mọi yếu tố từ thiết kế ấn tượng tới hiệu năng mạnh để trở thành sự lựa chọn hoàn hảo cho những bộ PC Gaming hiệu năng cao. Có dung lượng lên tới 16GB chuẩn DDR5 Bus 5200Mhz, mang tới cho bộ máy tính của bạn khả năng đa nhiệm mượt mà, nhanh chóng cũng như tăng lượng fps trong game của bạn lên đáng kể. Không chỉ là ngoại hình hầm hố chuẩn Gaming , RAM APACER NOX 16GB DDR5 Bus 5200Mhz – Black với lớp vỏ nhôm tản nhiệt còn giúp duy trì sự mát mẻ, ổn định hiệu năng hoạt động ngay cả khi sử dụng những tác vụ nặng, chơi Game liên tục trong thời gian dài."

# Tokenize and generate speech
inputs = tokenizer(text, return_tensors="pt")
with torch.no_grad():
    output = model(**inputs).waveform

# Save as WAV file
sf.write("output.wav", output.numpy().squeeze(), model.config.sampling_rate)
print("Speech saved as 'output.wav'")