from transformers import AutoTokenizer, AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("./finetuned-abc-vi-model")
tokenizer = AutoTokenizer.from_pretrained("./finetuned-abc-vi-model")
tokenizer.pad_token = tokenizer.eos_token

def ask(question: str):
    prompt = f"### Câu hỏi: {question}\n### Trả lời:"
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)

    outputs = model.generate(
        **inputs,
        max_new_tokens=100,
        pad_token_id=tokenizer.eos_token_id,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        repetition_penalty=1.2,
    )

    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return result.replace(prompt, "").strip()

print(ask("Công ty ABC bán gì?"))
