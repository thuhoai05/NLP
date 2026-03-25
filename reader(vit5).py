import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = AutoTokenizer.from_pretrained("./vit5_qa", use_fast=False)
model = AutoModelForSeq2SeqLM.from_pretrained("./vit5_qa")
print(f"📦 Đang tải mô hình Generative (ViT5) lên {device}...")


def answer_question(question, context): # Đổi contexts thành context
    # ✅ KHÔNG DÙNG JOIN NỮA
    input_text = f"question: {question} context: {context}"

    inputs = tokenizer(
        input_text,
        return_tensors="pt",
        max_length=512,
        truncation=True
    ).to(device)

    with torch.no_grad():
        outputs = model.generate(
            inputs.input_ids,
            max_length=64, # Cho nó dài thêm chút, 32 hơi ngắn
            num_beams=4,   # Đổi thành 4 cho nhanh mà vẫn mượt
            early_stopping=True
        )

    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return answer.strip(), 1.0
