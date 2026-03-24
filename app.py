import gradio as gr

# Import các hàm từ các file hệ thống của bạn
from retriever import retrieve
from reader_extractive import answer_question_extractive
from reader import answer_question as answer_question_generative

def qa_system_dual(question):
    # Tránh trường hợp người dùng bấm Submit mà không nhập gì
    if not question.strip():
        return "", "", "Vui lòng nhập câu hỏi."

    # 1. Tìm ngữ cảnh (Retriever)
    # Hàm retrieve của bạn trả về một List. Chúng ta lấy top_k=1 để có đoạn văn khớp nhất.
    context_list = retrieve(question, top_k=1)
    
    if not context_list or len(context_list) == 0:
        return "Không có câu trả lời", "Không có câu trả lời", "Không tìm thấy dẫn chứng phù hợp trong tài liệu."
    
    # Lấy phần tử đầu tiên trong list (đây là một String)
    context = context_list[0]
    
    # 2. Đưa context (String) cho 2 mô hình đọc
    # BERT (Extractive)
    try:
        ans_bert, _ = answer_question_extractive(question, context)
    except Exception as e:
        ans_bert = f"Lỗi BERT: {str(e)}"

    # ViT5 (Generative)
    try:
        ans_vit5, _ = answer_question_generative(question, context)
    except Exception as e:
        ans_vit5 = f"Lỗi ViT5: {str(e)}"
    
    return ans_bert, ans_vit5, context

# --- GIAO DIỆN GRADIO (Giữ nguyên cấu trúc đẹp của bạn) ---
with gr.Blocks(theme=gr.themes.Soft(), css="footer {visibility: hidden}") as demo:
    gr.Markdown("<h1 style='text-align: center; color: #FFA500;'>🤖 Hệ thống Hỏi Đáp RAG Tiếng Việt</h1>")
    gr.Markdown("<p style='text-align: center;'>So sánh trực tiếp mô hình Trích xuất (Extractive) và mô hình Sinh (Generative)</p>")
    
    with gr.Column(scale=1):
        question_input = gr.Textbox(
            label="Câu hỏi của bạn", 
            placeholder="Ví dụ: Bác Hồ sinh năm bao nhiêu?",
            lines=2
        )
        
        with gr.Row():
            btn_clear = gr.Button("Clear 🗑️")
            btn_submit = gr.Button("Submit 🚀", variant="primary")
    
    gr.HTML("<hr>") # Đường kẻ phân cách
    
    # Hiển thị dẫn chứng tìm được (Context)
    context_output = gr.Textbox(label="📄 Dẫn chứng trích xuất từ tài liệu (BM25 Retriever)", lines=4, interactive=False)
    
    gr.HTML("<br>")
    
    # Chia 2 cột song song để so sánh 2 mô hình
    with gr.Row():
        answer_bert = gr.Textbox(label="🎯 Trả lời bởi BERT (Extractive)", interactive=False, lines=3)
        answer_vit5 = gr.Textbox(label="✍️ Trả lời bởi ViT5 (Generative)", interactive=False, lines=3)

    # Xử lý sự kiện khi bấm nút Submit
    btn_submit.click(
        fn=qa_system_dual, 
        inputs=question_input, 
        outputs=[answer_bert, answer_vit5, context_output]
    )
    
    # Xử lý nút Clear (làm sạch 4 ô)
    btn_clear.click(
        fn=lambda: ("", "", "", ""), 
        inputs=None, 
        outputs=[question_input, answer_bert, answer_vit5, context_output]
    )

if __name__ == "__main__":
    print("🚀 Đang khởi động hệ thống giao diện Gradio...")
    # Chạy local
    demo.launch(share=False)