# dành cho bert
import torch
from transformers import AutoTokenizer, AutoModelForQuestionAnswering

model_name = "nguyenvulebinh/vi-mrc-base"
device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"📦 Đang tải mô hình Extractive: {model_name} lên {device}...")
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForQuestionAnswering.from_pretrained(model_name).to(device)

def answer_question_extractive(question, context):
    inputs = tokenizer(question, context, return_tensors="pt", truncation=True, max_length=512).to(device)
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Lấy vị trí bắt đầu và kết thúc
    answer_start = torch.argmax(outputs.start_logits)
    answer_end = torch.argmax(outputs.end_logits) + 1
    
    # ⚠️ SỬA: Bắt lỗi nếu start > end (Mô hình ngáo)
    if answer_start >= answer_end:
        return "Không có câu trả lời", 0.0
        
    # ⚠️ SỬA: Decode mượt mà hơn để tránh lỗi Type Error
    answer_tokens = inputs.input_ids[0][answer_start:answer_end]
    answer = tokenizer.decode(answer_tokens, skip_special_tokens=True)
    
    answer = answer.strip()
    if not answer:
        return "Không có câu trả lời", 0.0
        
    return answer, 1.0