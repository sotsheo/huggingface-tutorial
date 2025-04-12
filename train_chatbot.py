import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re

# Load dữ liệu
df = pd.read_csv("public/laptop_sales_large.csv")
df["search_text"] = df["Product"] + " " + df["Brand"] + " " + df["Customer"].astype(str)

# Hệ thống gợi ý thông minh
class LaptopAdvisor:
    def __init__(self, data):
        self.data = data
        self.brand_keywords = ['asus', 'dell', 'hp', 'lenovo', 'apple', 'acer', 'msi']
        self.use_cases = {
            'học tập': ['vivobook', 'inspiron', 'ideapad', 'mỏng nhẹ', 'pin trâu'],
            'gaming': ['gaming', 'gtx', 'rtx', 'tuf', 'predator'],
            'đồ họa': ['ultrasharp', 'studio', 'creator', 'quadro', 'xps']
        }
    
    def detect_intent(self, query):
        query = query.lower()
        intent = {
            'brand': None,
            'price_range': None,
            'use_case': None,
            'newest': False,
            'cheapest': False
        }
        
        # Phát hiện hãng
        for brand in self.brand_keywords:
            if brand in query:
                intent['brand'] = brand
                break
        
        # Phát hiện khoảng giá
        price_ranges = {
            'dưới 10 triệu': (0, 10000000),
            'từ 10-15 triệu': (10000000, 15000000),
            'từ 15-20 triệu': (15000000, 20000000),
            'trên 20 triệu': (20000000, float('inf'))
        }
        
        for range_text, (min_p, max_p) in price_ranges.items():
            if range_text in query:
                intent['price_range'] = (min_p, max_p)
                break
        
        # Phát hiện mục đích sử dụng
        for use_case, keywords in self.use_cases.items():
            if use_case in query or any(kw in query for kw in keywords):
                intent['use_case'] = use_case
                break
        
        # Phát hiện yêu cầu đặc biệt
        if 'mới nhất' in query:
            intent['newest'] = True
        if 'rẻ nhất' in query or 'giá rẻ' in query:
            intent['cheapest'] = True
            
        return intent
    
    def generate_response(self, results, intent):
        if results.empty:
            return "Hiện chưa có sản phẩm nào phù hợp với yêu cầu của bạn ạ."
        
        # Tạo phần giới thiệu thông minh
        intro = self._generate_intro(intent)
        
        # Tạo danh sách sản phẩm
        products = []
        for _, row in results.iterrows():
            product_desc = (
                f"✨ {row['Brand']} {row['Product']}\n"
                f"   💵 Giá: {row['Price']:,} VNĐ\n"
                f"   ⏰ Ngày mua: {row['Date']}\n"
                f"   👤 Khách hàng: {row['Customer']}\n"
            )
            products.append(product_desc)
        
        # Thêm lời kết
        outro = (
            "\n💡 Mẹo nhỏ: Bạn có thể nói rõ hơn về nhu cầu như:\n"
            "- 'Tôi cần laptop Dell cho sinh viên'\n"
            "- 'Máy tính nào chơi game tốt dưới 20 triệu?'"
        )
        
        return intro + "\n\n" + "\n".join(products) + outro
    
    def _generate_intro(self, intent):
        intros = []
        
        if intent['brand']:
            intros.append(f"hãng {intent['brand'].title()}")
        if intent['price_range']:
            min_p, max_p = intent['price_range']
            intros.append(f"khoảng giá {min_p:,}-{max_p:,} VNĐ")
        if intent['use_case']:
            intros.append(f"cho {intent['use_case']}")
        if intent['newest']:
            intros.append("mới nhất")
        if intent['cheapest']:
            intros.append("giá tốt nhất")
            
        if intros:
            return f"Đây là các lựa chọn {', '.join(intros)} phù hợp với bạn:"
        return "Đây là các sản phẩm phù hợp với yêu cầu của bạn:"
    
    def search(self, query, top_n=5):
        intent = self.detect_intent(query)
        result_df = self.data.copy()
        
        # Áp dụng các bộ lọc
        if intent['brand']:
            result_df = result_df[result_df['Brand'].str.lower() == intent['brand']]
        
        if intent['price_range']:
            min_p, max_p = intent['price_range']
            result_df = result_df[(result_df['Price'] >= min_p) & (result_df['Price'] <= max_p)]
        
        if intent['use_case']:
            keywords = self.use_cases[intent['use_case']]
            pattern = '|'.join(keywords)
            result_df = result_df[result_df['Product'].str.contains(pattern, case=False)]
        
        if intent['newest']:
            result_df = result_df.sort_values('Date', ascending=False)
        elif intent['cheapest']:
            result_df = result_df.sort_values('Price')
        
        # Áp dụng TF-IDF nếu có từ khóa cụ thể
        if len(query.split()) > 2:  # Chỉ áp dụng khi query đủ dài
            try:
                vectorizer = TfidfVectorizer(min_df=1, stop_words=None)
                tfidf_matrix = vectorizer.fit_transform(result_df["search_text"])
                query_vec = vectorizer.transform([query])
                result_df['similarity'] = cosine_similarity(query_vec, tfidf_matrix).flatten()
                result_df = result_df.sort_values('similarity', ascending=False)
            except ValueError:
                pass
        
        results = result_df.head(top_n)
        return self.generate_response(results, intent)

# Chạy chatbot
if __name__ == "__main__":
    advisor = LaptopAdvisor(df)
    print("💻 Laptop Advisor - Hệ thống tư vấn laptop thông minh")
    print("Gõ 'exit' hoặc 'quit' để thoát\n")
    
    while True:
        try:
            user_input = input("👤 Bạn đang tìm laptop nào? ").strip()
            if user_input.lower() in ['exit', 'quit']:
                print("Cảm ơn bạn đã sử dụng dịch vụ!")
                break
            
            if not user_input:
                print("Vui lòng nhập yêu cầu của bạn ạ...")
                continue
                
            print("\n" + advisor.search(user_input) + "\n")
            
        except KeyboardInterrupt:
            print("\nHẹn gặp lại bạn!")
            break
        except Exception as e:
            print(f"Xin lỗi, có lỗi xảy ra: {str(e)}")