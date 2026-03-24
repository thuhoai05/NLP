from transformers import AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer, DataCollatorForSeq2Seq, T5Tokenizer
from datasets import load_from_disk
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f" đang dùng thiết bị: {device}")

# 1. Load dữ liệu đã xử lý thành công từ bước trước
tokenized_dataset = load_from_disk("processed_data_generative")

# 2. LOAD TOKENIZER TỪ LOCAL (Để diệt lỗi KeyError: 0)
# Dùng đúng đường dẫn thư mục chứa file spiece.model bạn đã tải
tokenizer_path = "./vit5_tokenizer" 
tokenizer = T5Tokenizer.from_pretrained(tokenizer_path, use_fast=False)

# Load Model ViT5
model_name = "VietAI/vit5-base"
model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)

# 3. Cấu hình huấn luyện
training_args = Seq2SeqTrainingArguments(
    output_dir="./results_vit5",
    eval_strategy="epoch",
    learning_rate=5e-5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    weight_decay=0.01,
    save_total_limit=3,
    num_train_epochs=3, 
    predict_with_generate=True,
    fp16=torch.cuda.is_available(), # Sẽ tự động dùng nếu máy bạn có card NVIDIA
    logging_steps=50,
    save_strategy="epoch"
)

# 4. Data Collator
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["validation"],
    processing_class=tokenizer,
    data_collator=data_collator,
)

print("🚀 Bắt đầu huấn luyện mô hình Generative... (Quá trình này có thể mất khá lâu)")
trainer.train()

# 5. Lưu kết quả cuối cùng
model.save_pretrained("my_generative_model")
tokenizer.save_pretrained("my_generative_model")
print("✅ Chúc mừng! Bạn đã luyện xong 'bí kíp' Generative QA.")