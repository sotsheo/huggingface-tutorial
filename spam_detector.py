# -*- coding: utf-8 -*-
import pandas as pd
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
import pickle
from fastapi import FastAPI, Query
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

# ================== PHẦN 2: TIỀN XỬ LÝ VÀ MODEL ==================
def preprocess(text):
    text = text.lower()
    text = ''.join([c for c in text if c.isalpha() or c.isspace()])
    return text

def train_model():
    try:
        df = pd.read_csv("computer_spam.csv")
    except:
        df = create_dataset()
    
    df['processed_text'] = df['text'].apply(preprocess)
    
    X_train, X_test, y_train, y_test = train_test_split(
        df['processed_text'], 
        df['label'], 
        test_size=0.3,
        random_state=42
    )
    
    vectorizer = TfidfVectorizer(max_features=1000)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    
    model = SVC(kernel='linear', probability=True)
    model.fit(X_train_vec, y_train)
    
    y_pred = model.predict(X_test_vec)
    print("\nBáo cáo đánh giá:\n", classification_report(y_test, y_pred))
    
    with open('spam_model.pkl', 'wb') as f:
        pickle.dump(model, f)
    with open('vectorizer.pkl', 'wb') as f:
        pickle.dump(vectorizer, f)
    
    print("\nĐã lưu model vào: spam_model.pkl và vectorizer.pkl")
    return model, vectorizer

# ================== PHẦN 3: API CHÍNH ==================
app = FastAPI()

# Load model
try:
    with open('spam_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('vectorizer.pkl', 'rb') as f:
        vectorizer = pickle.load(f)
except:
    print("Không tìm thấy model, bắt đầu huấn luyện...")
    model, vectorizer = train_model()

# ================== PHẦN 4: API CHATBOT ĐƠN GIẢN ==================
@app.get("/chat_reply")
async def chat_reply(message: str = Query(..., min_length=2)):
    """API trả lời tự động chỉ cần message, không cần user info"""
    # Kiểm tra spam
    processed_text = preprocess(message)
    text_vec = vectorizer.transform([processed_text])
    is_spam = model.predict(text_vec)[0] == "spam"
    
    if is_spam:
        return {"reply": "Xin lỗi, chúng tôi không hỗ trợ tin nhắn quảng cáo/spam"}
    
    # Logic trả lời đơn giản
    if any(keyword in processed_text for keyword in ["giá", "bao nhiêu"]):
        product = next((p for p in ["dell", "macbook", "asus"] if p in processed_text), None)
        if product:
            prices = {"dell": "25 triệu", "macbook": "32 triệu", "asus": "18 triệu"}
            return {"reply": f"{product.title()} có giá {prices[product]}"}
        return {"reply": "Vui lòng cho biết tên sản phẩm cụ thể (Dell, Macbook, Asus...)"}
    
    elif "còn hàng" in processed_text:
        return {"reply": "Sản phẩm vẫn còn hàng, có thể đặt ngay trên website"}
    
    elif "bảo hành" in processed_text:
        return {"reply": "Bảo hành chính hãng 24 tháng tại tất cả chi nhánh"}
    
    else:
        return {"reply": "Bạn cần thông tin về giá, tình trạng hàng hay bảo hành ạ?"}

# ================== CHẠY ỨNG DỤNG ==================
if __name__ == "__main__":
    # Khởi động API
    print("\nAPI Chatbot Spam Detector đã sẵn sàng")
    print("Truy cập các endpoint sau:")
    print("- GET /chat_reply?message=<nội dung> - Chatbot tự động")
    print("- POST /predict - API phát hiện spam chi tiết")
    uvicorn.run(app, host="0.0.0.0", port=8000)