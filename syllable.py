from transformers import pipeline

model_name = "vinai/phobert-base"
nlp = pipeline("text-classification", model=model_name)

input_text = "Đây là một bài viết bằng tiếng Việt."
result = nlp(input_text)
print(result)
