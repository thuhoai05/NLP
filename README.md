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
Chạy các file theo thứ tự đánh số dưới đây:

1️⃣ eda.py: Phân tích thống kê tập dữ liệu (độ dài câu, phân bố nhãn) để có cái nhìn tổng quan trước khi huấn luyện.

2️⃣ preprocess_for_training.py: Làm sạch văn bản, xử lý các dấu câu tiếng Việt và chuyển đổi sang dạng Token ID cho mô hình.

3️⃣ train.py: Thực hiện quá trình Fine-tuning mô hình XLM-RoBERTa trên bộ dữ liệu ViSpanExtractQA.

4️⃣ build_index.py: Tạo bộ chỉ mục (Index) cho retriever. File này sẽ tạo ra bm25_index.pkl để hệ thống không phải load lại toàn bộ dữ liệu thô mỗi khi chạy.

5️⃣ evaluate_final.py: Chạy đánh giá trên tập test để xuất các chỉ số Accuracy, F1-score và Exact Match (EM).

6️⃣ chatbot.py: Giao diện dòng lệnh (CLI) để tương tác trực tiếp với hệ thống.
