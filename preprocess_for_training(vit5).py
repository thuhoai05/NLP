from datasets import load_dataset
from transformers import T5Tokenizer
import os

# SỬA Ở ĐÂY: Trỏ trực tiếp vào thư mục chứa 2 file bạn vừa tải
model_checkpoint = "./vit5_tokenizer"

try:
    # Load từ local để tránh lỗi cache hệ thống
    tokenizer = T5Tokenizer.from_pretrained(
        model_checkpoint, 
        use_fast=False,
        legacy=False
    )
    print("✅ CỰC KỲ TUYỆT VỜI! Tokenizer đã load thành công từ local.")
except Exception as e:
    print(f"❌ Lỗi cực nặng: {e}")
    print("Vui lòng đảm bảo bạn đã để file spiece.model vào thư mục vit5_tokenizer")
    exit()

max_input_length = 256
max_target_length = 64

def preprocess_function(examples):
    inputs = [
        f"question: {q} context: {c}"
        for q, c in zip(examples["question"], examples["context"])
    ]
    
    # ✅ FIX CHUẨN
    targets = [
        ans if ans and len(ans.strip()) > 0 else ""
        for ans in examples["answer_text"]
    ]

    model_inputs = tokenizer(
        inputs,
        max_length=256,
        truncation=True,
        padding="max_length"
    )

    labels = tokenizer(
        text_target=targets,
        max_length=64,
        truncation=True,
        padding="max_length"
    )

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs
# Load dataset
print("📡 Đang tải dataset từ Hugging Face...")
dataset = load_dataset("ntphuc149/ViSpanExtractQA")

# Map dữ liệu
tokenized_datasets = dataset.map(
    preprocess_function, 
    batched=True, 
    remove_columns=dataset["train"].column_names,
    desc="Đang xử lý dữ liệu cho ViT5..."
)

# Lưu dữ liệu
tokenized_datasets.save_to_disk("generative")
print("\n✅ Cuối cùng thì mọi thứ cũng xong! Bạn có thể chuyển sang bước Train.")

