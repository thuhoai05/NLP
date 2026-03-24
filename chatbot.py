import retriever
import reader

def ask_me(question):
    # Lấy top 5 đoạn văn từ BM25
    contexts = retriever.retrieve(question, top_k=5)
    valid_results = []

    for ctx in contexts:
        ans, score = reader.answer_question(question, ctx)
        
        # Ngưỡng tin cậy tối thiểu để trả lời
        if score < 0.25 or len(ans.strip()) == 0: continue
        
        # QA thường trích xuất cụm từ, nếu trả về quá dài (nguyên đoạn văn) thì bỏ qua
        if len(ans.split()) > 15: continue

        # SỬA Ở ĐÂY: Lưu thêm "context": ctx vào từ điển
        valid_results.append({"answer": ans, "score": score, "context": ctx})

    if valid_results:
        # Chọn đáp án có điểm cao nhất
        best = max(valid_results, key=lambda x: x['score'])
        print(f"\n🤖 Chatbot: {best['answer']} (Độ tin cậy: {best['score']:.2%})")
        # SỬA Ở ĐÂY: In thêm dẫn chứng ra màn hình
        print(f"📖 Dẫn chứng: {best['context']}") 
    else:
        print("\n🤖 Chatbot: Xin lỗi, mình không tìm thấy câu trả lời chính xác trong tài liệu.")

if __name__ == "__main__":
    print("="*50)
    print("🌟 HỆ THỐNG HỎI ĐÁP THÔNG MINH (BM25 + XLM-R) 🌟")
    print("="*50)
    
    while True:
        user_q = input("\n💬 Nhập câu hỏi của bạn (hoặc 'exit' để thoát): ")
        if user_q.lower() in ['exit', 'quit', 'thoát']:
            print("Tạm biệt bạn!")
            break
        
        if user_q.strip() == "":
            continue
            
        ask_me(user_q)