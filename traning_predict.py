import pickle
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# Bước 1: Load dữ liệu đã lưu
with open("product_data.pkl", "rb") as f:
    df = pickle.load(f)

product_embeddings = np.load("product_embeddings.npy")  # Load embeddings
index = faiss.read_index("faiss_index.bin")  # Load FAISS index
model = SentenceTransformer("all-MiniLM-L6-v2")  # Load lại mô hình

# Bước 2: Hàm tìm kiếm sản phẩm
def search_product(query, top_k=3, price_range=None, brand=None):
    query_embedding = model.encode([query], convert_to_numpy=True)
    
    # Tìm kiếm top_k sản phẩm gần nhất trong FAISS
    D, I = index.search(query_embedding, k=top_k)
    
    # Lấy thông tin sản phẩm từ DataFrame
    results = df.iloc[I[0]]

    # Lọc theo giá (nếu có)
    if price_range:
        min_price, max_price = price_range
        results = results[(results["Price"] >= min_price) & (results["Price"] <= max_price)]
    
    # Lọc theo thương hiệu (nếu có)
    if brand:
        results = results[results["Brand"].str.contains(brand, case=False, na=False)]
    
    return results[["Product", "Brand", "Price"]].to_dict(orient="records")

# Bước 3: Chạy chatbot trên CMD
print("Chatbot Sản phẩm - Nhập câu hỏi (gõ 'exit' để thoát):")
while True:
    query = input("Bạn: ")
    if query.lower() == "exit":
        break

    # Kiểm tra nếu có yêu cầu về giá
    price_range = None
    if "dưới" in query and "triệu" in query:
        max_price = int(query.split("dưới")[1].split("triệu")[0].strip()) * 1000000
        price_range = (0, max_price)
    
    # Kiểm tra nếu có yêu cầu về thương hiệu
    brand = None
    if "thương hiệu" in query:
        brand = query.split("thương hiệu")[-1].strip()

    results = search_product(query, top_k=5, price_range=price_range, brand=brand)
    
    if results:
        for res in results:
            print(f"- {res['Product']} | {res['Brand']} | {res['Price']} VND")
    else:
        print("Không tìm thấy sản phẩm phù hợp.")
