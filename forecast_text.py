from transformers import pipeline

# Tạo pipeline cho tiếng Việt
unmasker = pipeline('fill-mask', model='xlm-roberta-base')

# Sử dụng với tiếng Việt
results = unmasker("tôi đang đói tôi muốn ăn <mask> .")

for result in results:
    print(f"Token: {result['token_str']}, Điểm: {result['score']:.4f}, Câu: {result['sequence']}")