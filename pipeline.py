from retriever import retrieve
from reader import answer_question

def pipeline(question):
    # Bước 1: Retriever tìm các đoạn văn liên quan nhất
    print(f"🔍 Đang tìm kiếm tài liệu cho câu hỏi: {question}")
    candidate_contexts = retrieve(question, top_k=3) # Lấy top 3 đoạn văn
    
    best_answer = None
    max_confidence = -1

    # Bước 2: Reader đọc từng đoạn văn và tìm câu trả lời
    for i, context in enumerate(candidate_contexts):
        ans, conf = answer_question(question, context)
        print(f"  - Thử đoạn văn {i+1} (Độ tin cậy: {conf:.2f}): {ans}")
        
        if conf > max_confidence:
            max_confidence = conf
            best_answer = ans

    return best_answer, max_confidence

# TEST HỆ THỐNG
if __name__ == "__main__":
    query = "Ai là cha của Donald Trump?"
    final_ans, score = pipeline(query)
    print(f"\nKẾT QUẢ CUỐI CÙNG: {final_ans} (Confidence: {score:.2f})")