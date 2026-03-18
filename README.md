# Vietnamese Question Answering System

Project: Automatic Question Answering for Vietnamese

Pipeline:
Retriever: BM25
Reader: BERT / PhoBERT

Dataset: ViSpanExtractQA

## Setup

Create environment

python -m venv qa_env

Activate environment

qa_env\Scripts\activate

Install dependencies

pip install -r requirements.txt

1️⃣ Bước 1: Phân tích dữ liệu (EDA)
File: eda.py

Mục đích: Xem dữ liệu trông như thế nào, dài ngắn ra sao để viết báo cáo.

Lệnh: python eda.py

Kết quả: Hiện biểu đồ phân bố độ dài câu hỏi/đoạn văn.

2️⃣ Bước 2: Tiền xử lý dữ liệu (Preprocessing)
File: preprocess_for_training.py

Mục đích: Chuyển văn bản thô thành dạng "số" (Token ID) để AI hiểu được.

Lệnh: python preprocess_for_training.py

Kết quả: Tạo ra thư mục processed_dataset/. (Bắt buộc phải có thư mục này mới chạy được bước 3).

3️⃣ Bước 3: Huấn luyện mô hình (Fine-tuning)
File: train.py

Mục đích: Dạy cho mô hình XLM-RoBERTa cách trả lời câu hỏi tiếng Việt dựa trên bộ dữ liệu của bạn.

Lệnh: python train.py

Kết quả: Tạo ra thư mục "bộ não" my_finetuned_vimmrc/. (Đây là file quan trọng nhất để Chatbot hoạt động thông minh).

4️⃣ Bước 4: Xây dựng bộ tìm kiếm (Indexing)
File: build_index.py

Mục đích: Tạo "mục lục" cho toàn bộ các đoạn văn bản để khi hỏi, máy tìm kiếm nhanh hơn.

Lệnh: python build_index.py

Kết quả: Tạo ra file bm25_index.pkl. (Thiếu file này Chatbot sẽ báo lỗi không tìm thấy tài liệu).

5️⃣ Bước 5: Đánh giá & Sử dụng (Evaluation & Chat)
Đây là lúc bạn thu hoạch thành quả:

Đánh giá lấy số liệu nộp bài: python evaluate_final.py (Để lấy điểm F1/EM).

Mở Chatbot đi demo: python chatbot.py (Để hỏi đáp trực tiếp).
