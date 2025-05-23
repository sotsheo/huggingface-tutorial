from sentence_transformers import SentenceTransformer
from transformers import pipeline
import faiss
import numpy as np

# ============================
# 1. DỮ LIỆU CÔNG TY
# ============================

company_knowledge = [
    "Công ty ABC được thành lập vào ngày 12 tháng 3 năm 2015.",
    "Công ty có ba phòng ban chính: Phòng Nhân sự, Phòng Kỹ thuật, và Phòng Kinh doanh.",
    "Giám đốc điều hành hiện tại là ông Nguyễn Văn A.",
    "Phòng Nhân sự chịu trách nhiệm tuyển dụng và phúc lợi cho nhân viên.",
    "Phòng Kỹ thuật phụ trách phát triển phần mềm và hạ tầng công nghệ.",
    "Phòng Kinh doanh xử lý khách hàng và hợp đồng dịch vụ.",
    "Trụ sở chính của công ty nằm tại 123 đường ABC, Quận 1, TP. Hồ Chí Minh.",
    "Công ty có hơn 200 nhân viên tính đến năm 2024.",
    "Sứ mệnh của công ty là mang lại giải pháp công nghệ tối ưu cho doanh nghiệp Việt."
]

# ============================
# 2. TẠO EMBEDDING + FAISS
# ============================

print("Đang khởi tạo mô hình embedding...")
embedder = SentenceTransformer('all-MiniLM-L6-v2')
doc_embeddings = embedder.encode(company_knowledge, convert_to_numpy=True)

print("Đang tạo FAISS index...")
index = faiss.IndexFlatL2(doc_embeddings.shape[1])
index.add(doc_embeddings)

# ============================
# 3. TẢI MÔ HÌNH TRẢ LỜI
# ============================

print("Đang tải mô hình trả lời câu hỏi...")
qa_model = pipeline(
    "text2text-generation",
    model="VietAI/vit5-base",  # Có thể thay đổi nếu bạn dùng local
    device_map="auto", 
    model_kwargs={"torch_dtype": "auto"},
    max_new_tokens=200,
    temperature=0.2
)

# ============================
# 4. HÀM TRẢ LỜI CÂU HỎI
# ============================

def answer_question(question, top_k=3):
    # Bước 1: Encode câu hỏi
    q_embedding = embedder.encode([question], convert_to_numpy=True)

    # Bước 2: Tìm các câu nội dung liên quan nhất
    D, I = index.search(q_embedding, k=top_k)
    related_docs = [company_knowledge[i] for i in I[0]]

    # Bước 3: Tạo prompt
    context = "\n".join(related_docs)
    prompt = f"""Dưới đây là thông tin nội bộ công ty:

{context}

Dựa trên thông tin trên, hãy trả lời câu hỏi sau:
Câu hỏi: {question}
Trả lời:"""

    # Bước 4: Trả lời
    result = qa_model(prompt)[0]['generated_text']
    
    # Cắt phần trả lời nếu mô hình lặp lại prompt
    return result.split("Trả lời:")[-1].strip()

# ============================
# 5. Giao diện đơn giản
# ============================

if __name__ == "__main__":
    print("\n=== HỆ THỐNG HỎI ĐÁP VỀ CÔNG TY ===")
    while True:
        query = input("\nNhập câu hỏi của bạn (hoặc 'exit' để thoát): ")
        if query.lower().strip() == "exit":
            print("Tạm biệt!")
            break
        answer = answer_question(query)
        print(f"\n📌 Trả lời: {answer}")
