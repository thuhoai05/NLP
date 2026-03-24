# 🤖 Vietnamese RAG Question Answering System

Hệ thống hỏi đáp (QA) tiếng Việt dựa trên kiến trúc **RAG (Retrieval-Augmented Generation)**, so sánh hiệu năng giữa hai phương pháp **Extractive (BERT)** và **Generative (ViT5)**.

## 🌟 Tính năng chính
- **Retriever:** Sử dụng thuật toán BM25 để truy xuất ngữ cảnh từ tập tri thức tùy chỉnh.
- **Reader (Extractive):** Sử dụng mô hình `vi-mrc-base` để trích xuất câu trả lời trực tiếp từ văn bản.
- **Reader (Generative):** Sử dụng mô hình `ViT5` đã được fine-tune để sinh câu trả lời tự nhiên.
- **Giao diện:** Tích hợp UI trực quan với Gradio cho phép so sánh kết quả song song.

## 📂 Cấu trúc dự án
- `app.py`: File chạy giao diện chính.
- `retriever.py`: Module tìm kiếm văn bản (BM25).
- `reader.py` & `reader_extractive.py`: Các module xử lý câu trả lời.
- `clean_data.py`: Tiền xử lý dữ liệu đầu vào.
- `knowledge_base.txt`: Tập dữ liệu tri thức của hệ thống.

## 🚀 Hướng dẫn cài đặt

1. Clone repository:
   ```bash
   git clone [LINK_CỦA_BẠN]
