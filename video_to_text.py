from transformers import pipeline
from PIL import Image
import cv2
import numpy as np
import torch
from googletrans import Translator

# Tạo pipeline mô tả hình ảnh bằng mô hình BLIP
captioner = pipeline("image-to-text", model="Salesforce/blip-image-captioning-large")

# Tạo đối tượng dịch tự động sang tiếng Việt
translator = Translator()

# Đường dẫn đến video bạn muốn xử lý
video_path = "public/mixkit-chopping-fruit-for-a-salad-43913-hd-ready.mp4"  # đổi thành tên file video của bạn

# Mở video bằng OpenCV
cap = cv2.VideoCapture(video_path)

# Cấu hình: mỗi X giây lấy 1 khung hình
frame_rate = 1  # lấy 1 khung hình mỗi giây
frame_id = 0
fps = cap.get(cv2.CAP_PROP_FPS)
frame_interval = int(fps * frame_rate)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break  # nếu không còn frame thì thoát

    if frame_id % frame_interval == 0:
        # Chuyển từ BGR (OpenCV) sang RGB (chuẩn của PIL)
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
        # Mô tả hình ảnh bằng tiếng Anh
        result = captioner(image)
        english_caption = result[0]['generated_text']

        # Dịch sang tiếng Việt
        translated = translator.translate(english_caption, src='en', dest='vi').text

        # In kết quả ra màn hình
        print(f"🖼️ Khung hình {frame_id}: {translated} (EN: {english_caption})")

    frame_id += 1

cap.release()
