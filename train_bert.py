from transformers import AutoModelForQuestionAnswering, TrainingArguments, Trainer, DefaultDataCollator

model = AutoModelForQuestionAnswering.from_pretrained(model_checkpoint)

training_args = TrainingArguments(
    output_dir="bert_qa_model",
    eval_strategy="epoch",       # SỬA: bỏ chữ 'uation' đi, chỉ để 'eval_strategy'
    learning_rate=2e-5,
    per_device_train_batch_size=16, 
    num_train_epochs=2,             
    weight_decay=0.01,
    save_strategy="epoch",
    fp16=True,                      
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    data_collator=DefaultDataCollator(),
)

trainer.train()
trainer.save_model("my_finetuned_vimmrc")
print("🔥 XONG RỒI! Bạn đã có mô hình xịn.")
