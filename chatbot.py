import retriever  # File chứa hàm tìm kiếm BM25
import reader    # File chứa hàm answer_question (XLM-RoBERTa)

# Trong file chatbot.py
def ask_me(question):
    contexts = retriever.retrieve(question, top_k=5) # Lấy nhiều hơn để so sánh
    valid_results = []

    for ctx in contexts:
        ans, score = reader.answer_question(question, ctx)
        
        # LOGIC GOM LỖI CHUNG:
        # 1. Bỏ qua nếu score quá thấp (mô hình đang đoán mò)
        if score < 0.30: continue 
        
        # 2. Bỏ qua nếu đáp án quá dài (QA thường là cụm từ ngắn, không phải cả đoạn văn)
        if len(ans.split()) > 10: continue
        
        # 3. Trừ điểm nếu đáp án lặp lại chính câu hỏi (Lỗi phổ biến của AI)
        if ans.lower() in question.lower() and len(ans.split()) > 3: continue

        valid_results.append({"answer": ans, "score": score})

    # Chọn đáp án có score cao nhất sau lọc
    if valid_results:
        best = max(valid_results, key=lambda x: x['score'])
        print(f"\n🤖 ĐÁP ÁN CUỐI CÙNG: {best['answer']} ({best['score']:.2%})")
    else:
        print("\n🤖 Chatbot: Xin lỗi, thông tin này chưa có trong dữ liệu của mình.")
        
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