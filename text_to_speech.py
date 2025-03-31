import torch
from transformers import AutoProcessor, AutoModel
import scipy.io.wavfile

# Load model
device = "cuda" if torch.cuda.is_available() else "cpu"
processor = AutoProcessor.from_pretrained("suno/bark")
model = AutoModel.from_pretrained("suno/bark").to(device)

# Vietnamese text
text = "RAM APACER NOX 16GB DDR5 Bus 5200Mhz là mẫu RAM cao cấp cho game thủ."

# Process input with attention mask
inputs = processor(
    text,
    voice_preset="v2/en_speaker_0",
    return_tensors="pt"
).to(device)

# Tạo attention mask thủ công
inputs["attention_mask"] = torch.ones_like(inputs["input_ids"]).to(device)

# Generate audio with proper pad_token_id
with torch.no_grad():
    audio = model.generate(
        **inputs, 
        pad_token_id=processor.tokenizer.eos_token_id
    )

# Convert tensor to numpy
audio_np = audio.cpu().numpy().squeeze()

# Save output
scipy.io.wavfile.write(
    "output_bark.wav",
    rate=model.generation_config.sample_rate,
    data=audio_np
)

print("✅ File 'output_bark.wav' đã được lưu thành công!")
