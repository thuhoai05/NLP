from datasets import load_from_disk
from transformers import AutoModelForSeq2SeqLM, Trainer, TrainingArguments, AutoTokenizer

# ✅ Load data đã preprocess
dataset = load_from_disk("generative")

# 🔥 giảm data cho nhanh (có thể tăng sau)
dataset["train"] = dataset["train"].select(range(3000))
dataset["validation"] = dataset["validation"].select(range(500))

# Load model
model = AutoModelForSeq2SeqLM.from_pretrained("VietAI/vit5-base")

# Training config
training_args = TrainingArguments(
    output_dir="./vit5_qa",
    per_device_train_batch_size=2,
    per_device_eval_batch_size=4,
    num_train_epochs=1,  # ❗ tăng lên 1 (0.01 là quá ít)
    learning_rate=3e-5,
    logging_steps=100,
    save_steps=500,
    evaluation_strategy="no",
    save_total_limit=2,
    fp16=False
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
)

# Train
trainer.train()

# Save
trainer.save_model("./vit5_qa")
tokenizer = AutoTokenizer.from_pretrained("VietAI/vit5-base")
tokenizer.save_pretrained("./vit5_qa")