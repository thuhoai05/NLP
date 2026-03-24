🧠 Hệ thống Hỏi Đáp Tiếng Việt (BERT vs ViT5)
📌 Tổng quan

Dự án này xây dựng một hệ thống Question Answering (QA) cho tiếng Việt với hai hướng tiếp cận:

🔍 Extractive QA sử dụng BERT (trích xuất câu trả lời từ context)
✨ Generative QA sử dụng ViT5 (sinh câu trả lời)

Mục tiêu là:

So sánh hiệu năng giữa 2 mô hình
Đánh giá khả năng tổng quát hóa
Phân tích ưu/nhược điểm của từng phương pháp
🎯 Mục tiêu
Xây dựng pipeline QA hoàn chỉnh
Huấn luyện và đánh giá 2 loại mô hình
So sánh bằng các chỉ số:
Exact Match (EM)
F1 Score
Phân tích lỗi và khả năng generalization
🧱 Cấu trúc thư mục
### 1. Thành phần lõi (Core)
- `app.py`: File chạy chính, khởi tạo giao diện Web.
- `retriever.py`: Module tìm kiếm và truy xuất thông tin bằng BM25.
- `reader_extractive.py`: Sử dụng mô hình `vi-mrc-base` để tìm câu trả lời trực tiếp từ văn bản.
- `reader.py`: Sử dụng mô hình `ViT5` để tổng hợp và sinh câu trả lời hoàn chỉnh.

### 2. Xử lý dữ liệu (Data & Training)
- `clean_data.py`: Làm sạch `knowledge_base.txt`, loại bỏ rác và chuẩn hóa tiếng Việt.
- `preprocess_for_training.py`: Chuẩn bị tập dữ liệu huấn luyện từ Hugging Face cho ViT5.
- `reindex.py`: Cập nhật bộ nhớ tìm kiếm khi thay đổi dữ liệu tri thức.
- `train_bert.py` & `train_vit5.py`: Code dùng để huấn luyện/Fine-tune các mô hình.
- `evaluate_final.py`: Đánh giá hiệu năng mô hình qua chỉ số EM và F1.
⚙️ Cài đặt
1. Clone repo
git clone https://github.com/thuhoai05/NLP.git
cd NLP
2. Tạo môi trường
python -m venv qa_env
qa_env\Scripts\activate   # Windows
3. Cài thư viện
pip install -r requirements.txt
📊 Dataset
Dataset QA tiếng Việt
Format:
{
  "question": "...",
  "context": "...",
  "answer_text": "..."
}
Sử dụng:
Train: ~3000 mẫu
Test: 200 → 5000 mẫu

⚠️ Lưu ý: tập train nhỏ → dễ overfitting

🏋️ Huấn luyện
🔹 Bước 1: Preprocess
python preprocess_for_training.py
🔹 Bước 2: Train ViT5
python train_vit5.py
🧪 Đánh giá

Chạy:

python evaluate_final.py
Metrics:
Exact Match (EM): trùng khớp chính xác
F1 Score: đo độ tương đồng

📈 Kết quả
🔹 Test trên 200 mẫu
Mô hình	EM	F1
BERT	80.50%	86.56%
ViT5	70.00%	79.99%
🔹 Test trên 5000 mẫu
Mô hình	EM	F1
BERT	46.06%	66.40%
ViT5	29.64%	53.72%

📊 Phân tích
🔥 Nhận xét chính
Hiệu năng giảm mạnh khi tăng test size (200 → 5000)
→ cho thấy mô hình overfitting do dữ liệu train ít
⚖️ So sánh BERT vs ViT5
Tiêu chí	BERT	ViT5
Loại	Extractive	Generative
Ưu điểm	Chính xác cao	Linh hoạt
Nhược điểm	Không sinh được	Khó train

👉 BERT tốt hơn vì:

câu trả lời thường nằm sẵn trong context

👉 ViT5 kém hơn vì:

sinh lại câu → dễ lệch so với ground truth
❗ Phân tích lỗi
Ví dụ:
Question	Ground Truth	Prediction
Ai là chủ tịch FLC?	Trịnh Văn Quyết	Ông Trịnh Văn Quyết

👉 EM = 0 nhưng F1 cao
→ do khác biểu diễn nhưng đúng nội dung

⚠️ Hạn chế
Dataset nhỏ (~3000 mẫu)
Context bị cắt (max 512 tokens)
Generative model khó tối ưu hơn
Evaluation phụ thuộc string matching
🚀 Hướng phát triển
Tăng dữ liệu huấn luyện
Tăng số epoch
Dùng model lớn hơn (mT5)
Cải thiện evaluation (semantic similarity)
🧠 Công nghệ sử dụng
Python
PyTorch
HuggingFace Transformers
BERT
ViT5
