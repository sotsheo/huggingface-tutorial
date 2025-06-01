from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling

# 1. Load dữ liệu từ file JSONL
data = Dataset.from_json("company_qa.jsonl")

# 2. Dùng mô hình GPT-2 tiếng Việt
model_name = "minhtoan/gpt2-vietnamese"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token  # Thiết lập pad_token
model = AutoModelForCausalLM.from_pretrained(model_name)

# 3. Tiền xử lý dữ liệu
def preprocess(example):
    prompt = f"### Câu hỏi: {example['prompt']}\n### Trả lời:"
    response = f"{prompt} {example['completion']}"
    tokens = tokenizer(response, truncation=True, padding="max_length", max_length=256)
    tokens["labels"] = tokens["input_ids"].copy()
    return tokens

dataset = data.map(preprocess)

# 4. Cấu hình huấn luyện
training_args = TrainingArguments(
    output_dir="./finetuned-abc-vi-model",
    per_device_train_batch_size=2,
    num_train_epochs=5,
    logging_steps=10,
    save_steps=100,
    learning_rate=5e-5,
    weight_decay=0.01,
    fp16=False,
)

# 5. Huấn luyện mô hình
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    tokenizer=tokenizer,
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False)
)

trainer.train()
trainer.save_model("./finetuned-abc-vi-model")
tokenizer.save_pretrained("./finetuned-abc-vi-model")
