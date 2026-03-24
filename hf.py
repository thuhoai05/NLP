from datasets import load_dataset

# 1. Tải dataset
print("📥 Đang tải dataset...")
dataset = load_dataset("ntphuc149/ViSpanExtractQA")

# 2. Lấy toàn bộ Context (ngữ cảnh) duy nhất
# Dataset thường có nhiều câu hỏi cho cùng 1 context, ta dùng set() để tránh trùng lặp
contexts = set()

# Duyệt qua các tập (train, validation, test nếu có)
for split in dataset.keys():
    for item in dataset[split]:
        contexts.add(item['context'].strip())

# 3. Lưu vào file knowledge_base.txt
with open("knowledge_base.txt", "w", encoding="utf-8") as f:
    for ctx in contexts:
        f.write(ctx + "\n\n") # Cách nhau 2 dòng để FAISS dễ phân biệt

print(f"✅ Đã lưu {len(contexts)} đoạn văn vào knowledge_base.txt")