from datasets import load_dataset
from transformers import AutoTokenizer
import re

# 1. ĐỒNG NHẤT MODEL (Dùng vi-mrc-large như đã định)
model_checkpoint = "nguyenvulebinh/vi-mrc-large"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, use_fast=False)
max_length = 384

def clean_text(text):
    if text is None: return ""
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    return text

def preprocess(example):
    question = clean_text(example["question"])
    context = clean_text(example["context"])
    answer = clean_text(example["answer_text"])
    
    # Tokenize câu hỏi và ngữ cảnh
    inputs = tokenizer(
        question,
        context,
        max_length=max_length,
        truncation="only_second", # Chỉ cắt context, giữ nguyên question
        padding="max_length",
        return_offsets_mapping=True
    )

    offsets = inputs["offset_mapping"]
    start_char = context.find(answer)
    end_char = start_char + len(answer)

    # Mặc định nếu không tìm thấy là vị trí 0
    start_token_idx = 0
    end_token_idx = 0

    # Nếu tìm thấy đáp án trong context và không bị cắt mất do quá dài
    if start_char != -1:
        # Tìm token bắt đầu
        token_start_index = 0
        while token_start_index < len(offsets) and offsets[token_start_index][0] <= start_char:
            token_start_index += 1
        start_token_idx = token_start_index - 1

        # Tìm token kết thúc
        token_end_index = len(offsets) - 1
        while token_end_index >= 0 and offsets[token_end_index][1] >= end_char:
            token_end_index -= 1
        end_token_idx = token_end_index + 1

    inputs["start_positions"] = start_token_idx
    inputs["end_positions"] = end_token_idx

    # Xóa offset_mapping vì mô hình không nhận đầu vào này khi train
    inputs.pop("offset_mapping")
    return inputs

# --- CHẠY TIỀN XỬ LÝ ---
dataset = load_dataset("ntphuc149/ViSpanExtractQA")
tokenized_dataset = dataset.map(preprocess, batched=False) # Chạy từng mẫu để chính xác

# Lưu lại
tokenized_dataset.save_to_disk("processed_dataset")
print("✅ Tiền xử lý hoàn tất với Tokenizer chuẩn!")