import pandas as pd
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    TextDataset,
    DataCollatorForLanguageModeling,
    pipeline
)
import torch
import os

# === 1. Load CSV và chuẩn hóa dữ liệu ===
CSV_FILE = "public/products_10000.csv"

df = pd.read_csv(CSV_FILE)
df = df.dropna(subset=["name", "features", "description"])
df["input"] = "Thông tin sản phẩm: " + df["name"] + " | " + df["features"]
df["output"] = df["description"]

# === 2. Ghi dữ liệu huấn luyện vào file text ===
TRAIN_FILE = "train_data.txt"
with open(TRAIN_FILE, "w", encoding="utf-8") as f:
    for i, o in zip(df["input"], df["output"]):
        f.write(f"<s>Khách hỏi: {i}\nChatbot: {o}</s>\n")

# === 3. Load tokenizer và model ===
MODEL_NAME = "VietAI/gpt2-small"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

# === 4. Load tập dữ liệu huấn luyện ===
def load_dataset(file_path, tokenizer):
    return TextDataset(
        tokenizer=tokenizer,
        file_path=file_path,
        block_size=128,
    )

train_dataset = load_dataset(TRAIN_FILE, tokenizer)
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# === 5. Cấu hình và huấn luyện ===
training_args = TrainingArguments(
    output_dir="./chatbot_model",
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=4,
    save_steps=500,
    save_total_limit=1,
    fp16=False,
    logging_dir='./logs',
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset,
)

print("=== 🔧 Huấn luyện mô hình... ===")
trainer.train()
trainer.save_model("./chatbot_model")
tokenizer.save_pretrained("./chatbot_model")
print("✅ Huấn luyện xong!")

# === 6. Hàm tương tác với chatbot ===
print("🤖 Đang nạp mô hình huấn luyện xong để trò chuyện...")
chatbot = pipeline("text-generation", model="./chatbot_model", tokenizer=tokenizer)

def ask_chatbot(question):
    input_text = f"<s>Khách hỏi: {question}\nChatbot:"
    response = chatbot(input_text, max_length=150, do_sample=True, top_k=50)[0]["generated_text"]
    try:
        answer = response.split("Chatbot:")[1].strip().split("</s>")[0].strip()
        return answer
    except:
        return "Xin lỗi, tôi chưa hiểu rõ câu hỏi."

# === 7. Chat thử ===
print("\n=== 💬 Bắt đầu hỏi chatbot ===")
while True:
    q = input("Bạn: ")
    if q.lower() in ["exit", "quit", "bye"]:
        print("Chatbot: Tạm biệt bạn nhé!")
        break
    print("Chatbot:", ask_chatbot(q))
