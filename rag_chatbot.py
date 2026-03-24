import torch
from transformers import T5Tokenizer, AutoModelForSeq2SeqLM
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import os

# --- CẤU HÌNH ĐƯỜNG DẪN ---
base_path = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(base_path, "my_generative_model") 
knowledge_file = os.path.join(base_path, "knowledge_base.txt") 

device = "cuda" if torch.cuda.is_available() else "cpu"

# 1. LOAD MODEL
print("🧠 Đang nạp não bộ ViT5...")
tokenizer = T5Tokenizer.from_pretrained(model_path)
model = AutoModelForSeq2SeqLM.from_pretrained(model_path).to(device)

print("🔎 Đang chuẩn bị hệ thống tìm kiếm (FAISS)...")
retriever_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

# 2. CHUẨN BỊ KHO KIẾN THỨC
try:
    with open(knowledge_file, "r", encoding="utf-8") as f:
        documents = [d.strip() for d in f.read().split("\n\n") if d.strip()]
except FileNotFoundError:
    print(f"❌ LỖI: Không tìm thấy file {knowledge_file}")
    exit()

if len(documents) == 0:
    print("❌ LỖI: File kiến thức trống!")
    exit()

print(f"📚 Đang nạp {len(documents)} đoạn văn vào Vector Database...")
doc_embeddings = retriever_model.encode(documents)
doc_embeddings = np.array(doc_embeddings).astype('float32')

dimension = doc_embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(doc_embeddings)

# 3. CÁC HÀM XỬ LÝ
def get_relevant_context(question, top_k=2):
    # Lấy top 2 đoạn văn liên quan nhất thay vì 1
    question_embedding = retriever_model.encode([question])
    distances, indices = index.search(np.array(question_embedding).astype('float32'), top_k)
    
    # Gộp 2 đoạn văn lại để cung cấp ngữ cảnh rộng hơn
    retrieved_docs = [documents[i] for i in indices[0]]
    combined_context = " ".join(retrieved_docs)
    return combined_context, retrieved_docs

def generate_answer(question, context):
    # Format CHUẨN NHẤT cho ViT5 QA
    input_text = f"question: {question} context: {context}"
    inputs = tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True).to(device)
    
    with torch.no_grad():
        # Dùng num_beams=4 để nó tìm câu trả lời chính xác nhất thay vì sáng tạo linh tinh
        outputs = model.generate(inputs.input_ids, max_length=100, num_beams=4, early_stopping=True)
    
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# --- CHƯƠNG TRÌNH CHÍNH ---
print("✅ Hệ thống đã sẵn sàng! Gõ 'exit' để thoát.")
while True:
    user_query = input("\n👤 Bạn hỏi: ")
    if user_query.lower() == 'exit': break
    
    # Bước 1: Tìm đoạn văn
    context, raw_docs = get_relevant_context(user_query, top_k=2)
    
    # [QUAN TRỌNG] In ra xem FAISS tìm được cái gì để bắt lỗi
    print("\n" + "-"*40)
    print(f"🕵️ [DEBUG] Hệ thống tìm thấy đoạn văn:")
    print(f"1️⃣ {raw_docs[0][:200]}...") # In 200 ký tự đầu của đoạn 1
    if len(raw_docs) > 1:
        print(f"2️⃣ {raw_docs[1][:200]}...") # In 200 ký tự đầu của đoạn 2
    print("-"*40)
    
    # Bước 2: Sinh câu trả lời
    answer = generate_answer(user_query, context)
    print(f"\n🤖 Chatbot: {answer}")