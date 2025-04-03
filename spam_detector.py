# -*- coding: utf-8 -*-
import pandas as pd
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split  # ĐÃ THÊM DÒNG NÀY
import pickle
from fastapi import FastAPI
import uvicorn
from pydantic import BaseModel

# ================== PHẦN 1: TẠO DATASET ==================
def create_dataset():
    brands = ["Dell", "HP", "Asus", "Acer", "Lenovo", "Apple", "MSI"]
    products = ["laptop", "PC gaming", "máy tính bảng", "màn hình", "RAM", "SSD"]
    
    messages = []
    # Tin hợp lệ (ham)
    for _ in range(100):
        msg = random.choice([
            f"{random.choice(brands)} {random.choice(products)} giá bao nhiêu?",
            f"Còn hàng {random.choice(brands)} {random.choice(products)} không?",
            f"Tư vấn giúp mình {random.choice(products)} {random.choice(['văn phòng', 'đồ họa', 'gaming'])}",
            f"Bảo hành {random.choice(products)} bao lâu?"
        ])
        messages.append([msg, "ham"])
    
    # Tin spam (spam)
    for _ in range(100):
        spam_type = random.choice([
            f"{random.choice(['GIẢM SỐC', 'KHUYẾN MÃI', 'FLASH SALE'])} {random.randint(30,90)}% {random.choice(brands)} {random.choice(products)}",
            f"{random.choice(['TRÚNG THƯỞNG', 'QUAY SỐ MAY MẮN'])} khi mua {random.choice(products)}",
            f"{random.choice(['THANH LÝ', 'XẢ KHO'])} {random.choice(brands)} chỉ {random.randint(5,20)} triệu",
            f"CLICK NGAY ĐỂ NHẬN VOUCHER {random.randint(1,5)} TRIỆU"
        ])
        messages.append([spam_type, "spam"])
    
    df = pd.DataFrame(messages, columns=["text", "label"])
    df.to_csv("computer_spam.csv", index=False)
    print("Đã tạo dataset: computer_spam.csv")
    return df

# ================== PHẦN 2: HUẤN LUYỆN MODEL ==================
def preprocess(text):  # ĐƯA HÀM NÀY RA NGOÀI ĐỂ DÙNG CHUNG
    text = text.lower()
    text = ''.join([c for c in text if c.isalpha() or c.isspace()])
    return text

def train_model():
    try:
        df = pd.read_csv("computer_spam.csv")
    except:
        df = create_dataset()
    
    # Tiền xử lý
    df['processed_text'] = df['text'].apply(preprocess)
    
    # Chia dữ liệu
    X_train, X_test, y_train, y_test = train_test_split(
        df['processed_text'], 
        df['label'], 
        test_size=0.3,
        random_state=42
    )
    
    # Vector hóa
    vectorizer = TfidfVectorizer(max_features=1000)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    
    # Huấn luyện
    model = SVC(kernel='linear', probability=True)
    model.fit(X_train_vec, y_train)
    
    # Đánh giá
    y_pred = model.predict(X_test_vec)
    print("\nBáo cáo đánh giá:\n", classification_report(y_test, y_pred))
    
    # Lưu model
    with open('spam_model.pkl', 'wb') as f:
        pickle.dump(model, f)
    with open('vectorizer.pkl', 'wb') as f:
        pickle.dump(vectorizer, f)
    
    print("\nĐã lưu model vào: spam_model.pkl và vectorizer.pkl")
    return model, vectorizer

# ================== PHẦN 3: API DỰ ĐOÁN ==================
app = FastAPI()

class TextRequest(BaseModel):
    text: str

try:
    with open('spam_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('vectorizer.pkl', 'rb') as f:
        vectorizer = pickle.load(f)
except:
    print("Không tìm thấy model, bắt đầu huấn luyện...")
    model, vectorizer = train_model()

@app.post("/predict")
def predict(request: TextRequest):
    processed_text = preprocess(request.text)
    text_vec = vectorizer.transform([processed_text])
    prediction = model.predict(text_vec)[0]
    confidence = model.predict_proba(text_vec).max()
    
    return {
        "text": request.text,
        "processed_text": processed_text,
        "prediction": prediction,
        "confidence": float(confidence),
        "is_spam": prediction == "spam"
    }

# ================== CHẠY ỨNG DỤNG ==================
if __name__ == "__main__":
    # Tạo dataset nếu chưa có
    try:
        pd.read_csv("computer_spam.csv")
    except:
        create_dataset()
    
    # Huấn luyện model nếu chưa có
    try:
        with open('spam_model.pkl', 'rb') as f:
            model = pickle.load(f)
    except:
        train_model()
    
    # Khởi động API
    print("\nKhởi động API tại: http://localhost:8000")
    print("Truy cập http://localhost:8000/docs để test API")
    uvicorn.run(app, host="0.0.0.0", port=8000)