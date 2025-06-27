import gradio as gr
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import re
from difflib import SequenceMatcher
import os

class BizProChatbot:
    def __init__(self, company_info_file="company_info.txt"):
        print("🚀 Đang khởi tạo BizPro Chatbot...")
        
        # Đọc thông tin công ty từ file
        self.company_info = self._load_company_info(company_info_file)
        
        # Load model nhẹ
        model_name = "Qwen/Qwen2.5-0.5B-Instruct"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, 
            torch_dtype=torch.float16,
            trust_remote_code=True
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Tạo knowledge base từ thông tin công ty
        self.knowledge_base = self._create_knowledge_base()
        print("✅ BizPro Chatbot đã sẵn sàng!")
    
    def _load_company_info(self, file_path):
        """Đọc thông tin công ty từ file"""
        try:
            if os.path.exists(file_path):
                with open(file_path, 'r', encoding='utf-8') as file:
                    content = file.read()
                print(f"✅ Đã đọc thông tin công ty từ file: {file_path}")
                return content
            else:
                print(f"⚠️ Không tìm thấy file: {file_path}")
                # Fallback về thông tin mặc định
                return self._get_default_company_info()
        except Exception as e:
            print(f"❌ Lỗi khi đọc file {file_path}: {e}")
            return self._get_default_company_info()
    
    def _get_default_company_info(self):
        """Thông tin công ty mặc định nếu không đọc được file"""
        return ""

    def reload_company_info(self, file_path="company_info.txt"):
        """Tải lại thông tin công ty từ file (dùng để cập nhật real-time)"""
        old_info = self.company_info
        self.company_info = self._load_company_info(file_path)
        
        if old_info != self.company_info:
            # Cập nhật lại knowledge base
            self.knowledge_base = self._create_knowledge_base()
            print("🔄 Đã cập nhật thông tin công ty mới!")
            return True
        return False
    
    def _create_knowledge_base(self):
        """Tạo knowledge base từ thông tin công ty"""
        sections = {
            "thông tin chung": [
                "tên công ty", "địa chỉ", "website", "email", "hotline", "mã số doanh nghiệp", 
                "ngày thành lập", "tổng giám đốc", "người đại diện", "chi nhánh"
            ],
            "sản phẩm": [
                "bizhr", "bizsale", "bizfin", "phần mềm", "quản lý nhân sự", 
                "quản lý bán hàng", "kế toán", "crm"
            ],
            "dịch vụ": [
                "tư vấn", "triển khai", "bảo hành", "đào tạo", "hỗ trợ", "dùng thử"
            ],
            "giá cả": [
                "chi phí", "giá", "phí", "199000", "bao nhiêu tiền"
            ],
            "khách hàng": [
                "zenky", "hòa bình", "alphabooks", "việt smile", "á châu"
            ]
        }
        return sections
    
    def _find_relevant_info(self, query):
        """Tìm thông tin liên quan trong knowledge base"""
        query_lower = query.lower()
        relevant_sections = []
        
        # Tìm kiếm theo keywords
        for section, keywords in self.knowledge_base.items():
            for keyword in keywords:
                if keyword in query_lower:
                    relevant_sections.append(section)
                    break
        
        # Nếu không tìm thấy, tìm kiếm mờ
        if not relevant_sections:
            for section in self.knowledge_base.keys():
                similarity = SequenceMatcher(None, query_lower, section).ratio()
                if similarity > 0.3:
                    relevant_sections.append(section)
        
        # Trích xuất thông tin liên quan
        context = ""
        company_lines = self.company_info.split('\n')
        
        for line in company_lines:
            line_lower = line.lower()
            for section in relevant_sections:
                section_keywords = self.knowledge_base[section]
                if any(keyword in line_lower for keyword in section_keywords):
                    context += line + "\n"
                    break
        
        return context if context else self.company_info[:1000]  # Fallback
    
    def respond(self, query, history):
        """Trả lời câu hỏi về BizPro"""
        
        # Tìm thông tin liên quan
        relevant_context = self._find_relevant_info(query)
        
        # Tạo prompt với context
        system_prompt = f"""Bạn là chatbot chuyên biệt của Công ty BizPro Technology Solutions. 
Nhiệm vụ của bạn là trả lời chính xác các câu hỏi về công ty dựa trên thông tin được cung cấp.

THÔNG TIN CÔNG TY:
{relevant_context}

NGUYÊN TẮC:
- Chỉ trả lời dựa trên thông tin được cung cấp
- Trả lời ngắn gọn, chính xác và thân thiện
- Nếu không có thông tin, hãy nói "Tôi cần kiểm tra thêm thông tin này"
- Luôn giữ giọng điệu chuyên nghiệp nhưng gần gũi"""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Câu hỏi về BizPro: {query}"}
        ]
        
        # Format prompt
        if hasattr(self.tokenizer, 'apply_chat_template'):
            formatted_prompt = self.tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
            )
        else:
            formatted_prompt = f"System: {system_prompt}\nUser: {query}\nAssistant:"
        
        # Generate response
        input_ids = self.tokenizer.encode(formatted_prompt, return_tensors="pt")
        
        with torch.no_grad():
            output = self.model.generate(
                input_ids,
                max_new_tokens=200,
                do_sample=True,
                top_k=40,
                top_p=0.85,
                temperature=0.3,  # Thấp để đảm bảo thông tin chính xác
                repetition_penalty=1.1,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        
        # Decode response
        new_tokens = output[0][len(input_ids[0]):]
        result = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
        
        # Clean up
        result = result.strip()
        cleanup_patterns = ["Assistant:", "User:", "System:", "<|im_end|>", "<|im_start|>"]
        for pattern in cleanup_patterns:
            if pattern in result:
                result = result.split(pattern)[0].strip()
        
        # Thêm thông tin liên hệ nếu cần
        if any(word in query.lower() for word in ["liên hệ", "hotline", "email", "tư vấn"]):
            if "1900 633 686" not in result:
                result += "\n\n📞 Hotline: 1900 633 686 | 📧 Email: contact@bizpro.vn"
        
        return result

def create_sample_company_file():
    """Tạo file mẫu company_info.txt nếu chưa có"""
    if not os.path.exists("company_info.txt"):
        sample_content = """THÔNG TIN DOANH NGHIỆP

1. Thông tin chung
- Tên doanh nghiệp: Công ty Cổ phần Giải pháp Công nghệ BizPro
- Website: www.bizpro.vn
- Email: contact@bizpro.vn
- Hotline: 1900 633 686

2. Sản phẩm chính
- BizHR: Phần mềm quản lý nhân sự
- BizSale: Phần mềm quản lý bán hàng
- BizFin: Phần mềm tài chính kế toán

3. Dịch vụ
- Tư vấn miễn phí 24/7
- Triển khai nhanh 3-5 ngày
- Bảo hành miễn phí 12 tháng
"""
        
        with open("company_info.txt", 'w', encoding='utf-8') as f:
            f.write(sample_content)
        print("📝 Đã tạo file mẫu company_info.txt")

# Khởi tạo chatbot
print("🔄 Đang khởi tạo BizPro Chatbot...")

# Tạo file mẫu nếu chưa có
create_sample_company_file()

# Khởi tạo bot với file cấu hình
bizpro_bot = BizProChatbot("company_info.txt")

# Hàm wrapper để có thể reload thông tin
def respond_with_reload(query, history):
    # Kiểm tra nếu có yêu cầu reload
    if query.lower().strip() in ["reload", "cập nhật", "tải lại"]:
        if bizpro_bot.reload_company_info():
            return "✅ Đã cập nhật thông tin công ty mới từ file!"
        else:
            return "ℹ️ Thông tin công ty không có thay đổi."
    
    return bizpro_bot.respond(query, history)

# Tạo Gradio interface
demo = gr.ChatInterface(
    fn=respond_with_reload,
    title="🏢 BizPro Technology Solutions - Chatbot Hỗ Trợ",
    description="""
    Chào mừng bạn đến với BizPro! Tôi có thể trả lời mọi câu hỏi về:
    ✅ Thông tin công ty và dịch vụ
    ✅ Sản phẩm: BizHR, BizSale, BizFin  
    ✅ Giá cả và gói dịch vụ
    ✅ Hỗ trợ khách hàng
    
    💡 Gõ "reload" hoặc "cập nhật" để tải lại thông tin từ file company_info.txt
    """,
    examples=[
        "BizPro là công ty gì?",
        "Địa chỉ trụ sở chính ở đâu?",
        "Có những sản phẩm nào?",
        "Giá phần mềm bao nhiêu?",
        "BizHR có những tính năng gì?",
        "Làm sao để liên hệ tư vấn?",
        "Có dùng thử miễn phí không?",
        "Thời gian triển khai bao lâu?",
        "Khách hàng tiêu biểu là ai?",
        "Có hỗ trợ trên mobile không?",
        "reload"
    ],
    chatbot=gr.Chatbot(
        placeholder="<div style='text-align: center; color: #666;'>💬 Xin chào! Tôi là chatbot của BizPro. Bạn cần hỗ trợ thông tin gì?</div>"
    )
)

if __name__ == "__main__":
    demo.launch(
        share=True,
        show_error=True,
        server_name="0.0.0.0",
        server_port=7860
    )