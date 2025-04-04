# -*- coding: utf-8 -*-
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import re

# ================== CẤU HÌNH HỆ THỐNG ==================
app = FastAPI()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ================== PHẦN 1: ĐỌC VÀ XỬ LÝ DỮ LIỆU ==================
class ProductDatabase:
    def __init__(self, csv_path="public/laptop_sales_large.csv"):  # Thêm giá trị mặc định
        try:
            self.df = pd.read_csv(csv_path)
            self._preprocess_data()
            print("✅ Đã tải xong dữ liệu sản phẩm!")
        except Exception as e:
            print(f"❌ Lỗi khi đọc file CSV: {str(e)}")
            self.df = pd.DataFrame()

    def _preprocess_data(self):
        """Chuẩn hóa dữ liệu"""
        if not self.df.empty:
            self.df['Product'] = self.df['Product'].str.lower().fillna('')
            self.df['Brand'] = self.df['Brand'].str.lower().fillna('')
            self.df['Price'] = pd.to_numeric(self.df['Price'], errors='coerce').fillna(0)

    def get_product_info(self, product_name: str) -> dict:
        """Lấy thông tin sản phẩm theo tên"""
        if self.df.empty:
            return None
            
        product_name = product_name.lower()
        matched = self.df[self.df['Product'].str.contains(product_name, case=False, na=False)]
        
        if not matched.empty:
            product = matched.iloc[0]
            return {
                "name": product['Product'],
                "brand": product['Brand'],
                "price": f"{product['Price']:,.0f} VNĐ",
                "quantity": product['Quantity'],
                "last_sale": product.get('Date', 'N/A')
            }
        return None

# ================== PHẦN 2: HỆ THỐNG HỎI ĐÁP ==================
class QASystem:
    def __init__(self, db):
        self.db = db
        try:
            self.tokenizer = AutoTokenizer.from_pretrained("vinai/bartpho-syllable")
            self.model = AutoModelForSeq2SeqLM.from_pretrained("vinai/bartpho-syllable").to(device)
            print("✅ Đã tải xong mô hình ngôn ngữ!")
        except Exception as e:
            print(f"❌ Lỗi khi tải mô hình: {str(e)}")
            self.model = None

    def extract_info(self, question: str) -> dict:
        """Trích xuất thông tin từ câu hỏi"""
        question = question.lower()
        print((p for p in self.db.df['Product'].unique() if p in question))
        return {
            "product": next((p for p in self.db.df['Product'].unique() if p in question), None),
            "brand": next((b for b in self.db.df['Brand'].unique() if b in question), None),
            "question_type": ("price" if "giá" in question else 
                            "specs" if "cấu hình" in question else
                            "stock" if "còn hàng" in question else
                            "general")
        }

    def generate_response(self, question: str) -> str:
        """Tạo câu trả lời"""
        if not self.model:
            return "Xin lỗi, hệ thống đang bảo trì"
            
        info = self.extract_info(question)
        
        # Lấy thông tin từ database
        context = ""
        if info['product']:
            product_info = self.db.get_product_info(info['product'])
            if product_info:
                context = f"""
                Sản phẩm: {product_info['name']}
                Hãng: {product_info['brand']}
                Giá: {product_info['price']}
                Số lượng: {product_info['quantity']}
                """
        
        # Tạo câu trả lời
        inputs = self.tokenizer(question, context, return_tensors="pt", truncation=True).to(device)
        outputs = self.model.generate(**inputs, max_length=150)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

# ================== PHẦN 3: API ==================
# Khởi tạo hệ thống
try:
    db = ProductDatabase()  # Sửa thành đường dẫn file CSV thực tế của bạn
    qa_system = QASystem(db)
except Exception as e:
    print(f"❌ Lỗi khởi tạo hệ thống: {str(e)}")
    qa_system = None

class QuestionRequest(BaseModel):
    question: str

@app.post("/ask")
async def ask_question(request: QuestionRequest):
    if not qa_system or not qa_system.model:
        raise HTTPException(status_code=500, detail="Hệ thống chưa sẵn sàng")
    
    try:
        answer = qa_system.generate_response(request.question)
        return {"answer": answer}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Lỗi xử lý câu hỏi: {str(e)}")

# ================== PHẦN 4: CHẠY ỨNG DỤNG ==================
if __name__ == "__main__":
    print("\n🤖 AI CHATBOT THÔNG TIN SẢN PHẨM")
    print("Các endpoint chính:")
    print("- POST /ask - Gửi câu hỏi về sản phẩm")
    print("\nAPI đang chạy tại: http://localhost:8000")
    
    uvicorn.run(app, host="0.0.0.0", port=8000)