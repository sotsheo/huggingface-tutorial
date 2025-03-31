from diffusers import AutoPipelineForText2Image
import torch

pipe = AutoPipelineForText2Image.from_pretrained(
    "kandinsky-community/kandinsky-2-2-decoder",
    torch_dtype=torch.float16,
).to("cuda" if torch.cuda.is_available() else "cpu")

prompt = "Một bức tranh phong cảnh Việt Nam với đồng lúa vàng, hoàng hôn"
image = pipe(prompt).images[0]
image.save("kandinsky_output.png")