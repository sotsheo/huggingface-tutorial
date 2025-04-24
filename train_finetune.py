import pandas as pd
from transformers import Trainer, TrainingArguments, AutoTokenizer, AutoModelForSequenceClassification
from datasets import Dataset
import torch

# ✅ Tải dữ liệu
df = pd.read_csv("public/data.csv").dropna()
dataset = Dataset.from_pandas(df)

# ✅ Sử dụng model nhẹ (có thể dùng distilbert-base-uncased hoặc bản offline)
model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# ✅ Tiền xử lý dữ liệu
def preprocess(example):
    return tokenizer(example['input'], truncation=True, padding='max_length', max_length=128)

encoded_dataset = dataset.map(preprocess, batched=True)
encoded_dataset = encoded_dataset.rename_column("output", "labels")
encoded_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])

# ✅ Tạo model (nhẹ)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=1)

# ✅ TrainingArguments
training_args = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=8,
    num_train_epochs=1,
    logging_dir="./logs",
    logging_steps=10,
    save_total_limit=1,
)

# ✅ Huấn luyện
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=encoded_dataset,
)

trainer.train()
