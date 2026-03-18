1. Mục tiêu project
Xây dựng hệ thống hỏi đáp tiếng Việt (Vietnamese Question Answering System)
gồm 2 thành phần:
- Retriever: tìm đoạn văn liên quan
- Reader: trích xuất câu trả lời

2. Kiến trúc hệ thống
User Question
     ↓
[Retriever - TF-IDF / BM25]
     ↓
Top-k đoạn văn liên quan
     ↓
[Reader - PhoBERT / vi-mrc]
     ↓
Answer span (start, end)
     ↓
Final Answer
  
