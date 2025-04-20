from audiocraft.models import MusicGen
model = MusicGen.get_pretrained('facebook/musicgen-small')
model.set_generation_params(duration=30)  # Tạo nhạc 30 giây

descriptions = ["Nhạc piano nhẹ nhàng", "EDM sôi động với bass mạnh"]
wav = model.generate(descriptions)  # Tạo file âm thanh