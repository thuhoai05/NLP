import torch
import gc # Thư viện giải phóng bộ nhớ
from retriever import retrieve # Gọi ĐÚNG tên hàm của bạn
from reader_extractive import answer_question_extractive
from reader import answer_question as answer_question_generative

def run_safe_comparison(query):
    try:
        # 1. Tìm ngữ cảnh
        print("\n🔍 Đang lục tìm tài liệu...")
        contexts = retrieve(query, top_k=2) # Lấy 2 đoạn liên quan nhất từ BM25
        
        # GỘP CÁC ĐOẠN LẠI (Sửa lỗi list of strings)
        combined_context = contexts[0]  # chỉ lấy top-1

        if not combined_context.strip():
            print("❌ Không tìm thấy ngữ cảnh phù hợp!")
            return

        print(f"📌 Ngữ cảnh tìm được: {combined_context[:200]}...")

        # 2. Chạy Extractive BERT
        print("⏳ BERT đang suy nghĩ...")
        ans_bert, _ = answer_question_extractive(query, combined_context)
        
        # Giải phóng bộ nhớ đệm (Rất tốt cho máy local)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

        # 3. Chạy Generative ViT5
        print("⏳ ViT5 đang suy nghĩ...")
        ans_vit5, _ = answer_question_generative(query, combined_context)

        # 4. In kết quả
        print("\n" + "="*50)
        print(f"❓ Câu hỏi: {query}")
        print(f"🎯 BERT (Cắt chữ): {ans_bert}")
        print(f"🎯 ViT5 (Sinh câu): {ans_vit5}")
        print("="*50)

    except Exception as e:
        print(f"💥 Có lỗi xảy ra: {e}")

if __name__ == "__main__":
    print("="*60)
    print("🤖 CHATBOT RAG SONG SONG (Extractive vs Generative)")
    print("="*60)
    while True:
        q = input("\nNhập câu hỏi (gõ 'exit' để thoát): ")
        if q.lower() == 'exit': 
            print("Tạm biệt!")
            break
        run_safe_comparison(q)