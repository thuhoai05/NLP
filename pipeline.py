from retriever import retrieve
from reader import answer_question

def pipeline(question):
    print(f"🔍 Đang truy vấn dữ liệu...")
    candidate_contexts = retrieve(question, top_k=3) # Top 3 là tối ưu nhất cho pipeline
    
    best_answer = None
    max_confidence = -1
    best_context=""

    for context in candidate_contexts:
        ans, conf = answer_question(question, context)
        if conf > max_confidence:
            max_confidence = conf
            best_answer = ans
            best_context = context

    return best_answer, max_confidence, best_context

# TEST HỆ THỐNG
if __name__ == "__main__":
    query = "Ai là cha của Donald Trump?"
    final_ans, score = pipeline(query)
    print(f"\nKẾT QUẢ CUỐI CÙNG: {final_ans} (Confidence: {score:.2f})")