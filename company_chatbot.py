from sentence_transformers import SentenceTransformer
import numpy as np
import re

class CompanyChatbot:
    def __init__(self, txt_file="company/company_info.txt"):
        """
        Khởi tạo chatbot với file txt chứa thông tin công ty
        Sử dụng model Sentence Transformer tiếng Việt
        """
        # Load model tiếng Việt
        self.model = SentenceTransformer('keepitreal/vietnamese-sbert')
        
        # Đọc và xử lý file txt
        with open(txt_file, 'r', encoding='utf-8') as f:
            text = f.read()
        
        # Tách thành các đoạn văn ngắn
        self.paragraphs = [p.strip() for p in re.split(r'\n\s*\n', text) if p.strip()]
        
        # Tạo embeddings cho các đoạn văn
        self.paragraph_embeddings = self.model.encode(self.paragraphs)
    
    def preprocess_question(self, question):
        """Chuẩn hóa câu hỏi tiếng Việt"""
        question = question.lower().strip()
        question = re.sub(r'[^\w\s]', '', question)  # Bỏ dấu câu
        return question
    
    def answer(self, question):
        """Trả lời câu hỏi dựa trên thông tin công ty"""
        question = self.preprocess_question(question)
        
        # Tìm đoạn văn liên quan nhất
        question_embedding = self.model.encode(question)
        scores = np.dot(self.paragraph_embeddings, question_embedding)
        best_idx = np.argmax(scores)
        
        # Ngưỡng similarity có thể điều chỉnh
        if scores[best_idx] > 0.6:  
            return self.paragraphs[best_idx]
        return "Xin lỗi, tôi không tìm thấy thông tin để trả lời câu hỏi này."

if __name__ == "__main__":
    bot = CompanyChatbot()
    print("Chatbot thông tin công ty (Nhập 'exit' để thoát)")
    
    while True:
        question = input("Bạn có câu hỏi gì về công ty? ")
        if question.lower() in ['exit', 'thoát', 'q']:
            break
        print("=>", bot.answer(question))
        print()  # In thêm dòng trống