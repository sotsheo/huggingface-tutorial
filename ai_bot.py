# -*- coding: utf-8 -*-
import pandas as pd
import pickle
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch


# ================== CẤU HÌNH HỆ THỐNG ==================
app = FastAPI()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ================== PHẦN 1: LOAD MODEL NGÔN NGỮ ==================
class QAModel:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("vinai/bartpho-qa")
        self.model = AutoModelForSeq2SeqLM.from_pretrained("vinai/bartpho-qa").to(device)
        self.product_db = self._load_product_db()
    
    def _load_product_db(self):
        """Tải cơ sở dữ liệu sản phẩm (có thể thay bằng kết nối database thực)"""
        return {
            "laptop": {
                "Dell XPS 13": {"price": "25.990.000đ", "specs": "Core i7, 16GB RAM, SSD 512GB"},
                "MacBook Air M1": {"price": "23.500.000đ", "specs": "Apple M1, 8GB RAM, 256GB SSD"}
            },
            "pc": {
                "Asus ROG Strix": {"price": "35.000.000đ", "specs": "RTX 3080, 32GB RAM"}
            }
        }
    
    def generate_answer(self, question: str, context: str = None) -> str:
        """Sinh câu trả lời dựa trên câu hỏi và ngữ cảnh"""
        if not context:
            context = self._retrieve_context(question)
        
        inputs = self.tokenizer(question, context, return_tensors="pt", truncation=True).to(device)
        outputs = self.model.generate(**inputs, max_length=150)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    def _retrieve_context(self, question: str) -> str:
        """Truy xuất thông tin sản phẩm liên quan"""
        product_type = self._detect_product_type(question)
        product_name = self._detect_product_name(question)
        
        if product_type and product_name in self.product_db.get(product_type, {}):
            product_info = self.product_db[product_type][product_name]
            return f"""
            Thông tin sản phẩm {product_name}:
            - Giá bán: {product_info['price']}
            - Cấu hình: {product_info['specs']}
            - Bảo hành: 24 tháng
            """
        return "Sản phẩm máy tính với nhiều lựa chọn từ các hãng Dell, Asus, Apple..."

    def _detect_product_type(self, text: str) -> str:
        """Nhận diện loại sản phẩm"""
        text = text.lower()
        if "laptop" in text: return "laptop"
        if "pc" in text or "máy bàn" in text: return "pc"
        return None

    def _detect_product_name(self, text: str) -> str:
        """Nhận diện tên sản phẩm cụ thể"""
        text = text.lower()
        for product_type in self.product_db.values():
            for name in product_type.keys():
                if name.lower() in text:
                    return name
        return None

# Khởi tạo hệ thống AI
try:
    qa_system = QAModel()
    print("Đã tải xong model QA!")
except Exception as e:
    print(f"Lỗi khi tải model: {str(e)}")
    qa_system = None

# ================== PHẦN 2: API THÔNG MINH ==================
class QuestionRequest(BaseModel):
    question: str
    conversation_history: list = []

@app.post("/ask")
async def ask_ai(request: QuestionRequest):
    if not qa_system:
        raise HTTPException(status_code=500, detail="Hệ thống AI chưa sẵn sàng")
    
    # Xây dựng ngữ cảnh từ lịch sử hội thoại
    context = "\n".join(request.conversation_history[-3:])  # Lấy 3 tin nhắn gần nhất
    
    # Tạo câu trả lời
    answer = qa_system.generate_answer(
        question=request.question,
        context=context if context else None
    )
    
    return {
        "answer": answer,
        "suggestions": [
            "Bạn cần thêm thông tin gì về sản phẩm này?",
            "Bạn muốn so sánh với sản phẩm khác?"
        ]
    }

# ================== PHẦN 3: CHẠY ỨNG DỤNG ==================
if __name__ == "__main__":
    print("\n🤖 HỆ THỐNG AI BÁN HÀNG THÔNG MINH ĐÃ SẴN SÀNG")
    print("Các tính năng chính:")
    print("- Hiểu và trả lời câu hỏi tự nhiên về sản phẩm")
    print("- Duy trì ngữ cảnh hội thoại")
    print("- Gợi ý tiếp theo thông minh")
    
    uvicorn.run(app, host="0.0.0.0", port=8000)