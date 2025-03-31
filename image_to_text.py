from transformers import pipeline
from PIL import Image

captioner = pipeline("image-to-text", model="Salesforce/blip-image-captioning-large")
image = Image.open("18.jpg")
result = captioner(image)
print(result)