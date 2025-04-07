from diffusers import AutoPipelineForText2Image
import torch

# 1. Sử dụng model công khai Stable Diffusion XL thay thế
pipeline = AutoPipelineForText2Image.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",  # Model công khai
    torch_dtype=torch.float16,  # Giảm VRAM (dùng bfloat16 nếu GPU hỗ trợ)
    variant="fp16",
    use_safetensors=True
).to("cuda")

# 2. Tải LoRA thay thế (nếu cần)
pipeline.load_lora_weights(
    "ostris/ikea-instructions-lora-sdxl",  # LoRA công khai
    weight_name="ikea_instructions_xl_v1_5.safetensors"  # Tên file
)

# 3. Tạo ảnh
prompt = """
a boy and a girl looking out of a window with a cat perched on the window sill. 
There is a bicycle parked in front of them and a plant with flowers to the right side of the image. 
The wall behind them is visible in the background.
"""
image = pipeline(prompt, num_inference_steps=30).images[0]
image.save("my_image.png")