from transformers import pipeline
from PIL import Image
import requests
from io import BytesIO

captioner = pipeline("image-to-text", model="Salesforce/blip-image-captioning-large")
# image = Image.open("public/18.jpg")
url = 'https://demos.themeselection.com/sneat-bootstrap-html-admin-template/assets/img/backgrounds/event.jpg'
# Tải ảnh từ URL
response = requests.get(url)

# Kiểm tra nếu tải thành công
if response.status_code == 200:
    image = Image.open(BytesIO(response.content))
    # image.show()  # hoặc img.save("local.jpg")
else:
    print("Không thể tải ảnh từ URL.")
result = captioner(image)
print(result)