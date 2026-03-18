from transformers import AutoModelForQuestionAnswering, TrainingArguments, Trainer, default_data_collator, AutoTokenizer
from datasets import load_from_disk
import torch
import numpy as np
import gc

# 1. Cấu hình thiết bị
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🚀 Đang chạy trên: {device.upper()}")

# 2. Load dataset
print("📂 Loading dataset...")
tokenized_dataset = load_from_disk("processed_dataset")
# Lấy một lượng nhỏ để train nhanh và tránh tràn RAM
train_sub = tokenized_dataset["train"].select(range(50))

# 3. Load model + tokenizer
model_name = "nguyenvulebinh/vi-mrc-large"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForQuestionAnswering.from_pretrained(model_name).to(device)

# 4. Training Arguments (Bản tối giản nhất để tránh lỗi tham số)
training_args = TrainingArguments(
    output_dir="./results",
    learning_rate=2e-5,
    per_device_train_batch_size=1, 
    num_train_epochs=1,
    weight_decay=0.01,
    save_strategy="no", # Không lưu checkpoint để tiết kiệm RAM
    logging_steps=5,        
    report_to="none"
)

# 5. Khởi tạo Trainer (Bỏ eval_dataset để không lo lỗi tham số đánh giá)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_sub,
    data_collator=default_data_collator
)

# 6. Train
print("⚡ Đang huấn luyện...")
trainer.train()

# 7. Dọn dẹp RAM trước khi lưu
print("🧹 Đang dọn dẹp bộ nhớ đệm...")
del trainer
gc.collect()

# 8. Lưu mô hình (Đây là đích đến cuối cùng của bạn)
print("💾 Đang lưu mô hình vào thư mục my_finetuned_vimmrc...")
model.save_pretrained("my_finetuned_vimmrc")
tokenizer.save_pretrained("my_finetuned_vimmrc")

print("✅ TẤT CẢ ĐÃ XONG! Hãy kiểm tra thư mục my_finetuned_vimmrc.")