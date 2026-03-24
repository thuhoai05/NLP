import pickle
from rank_bm25 import BM25Okapi
from pyvi import ViTokenizer
import string

def preprocess(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    return ViTokenizer.tokenize(text).split()

print("⏳ Đang nạp dữ liệu sạch và tạo Index mới...")

# 1. Đọc file đã clean
with open("knowledge_base.txt", "r", encoding="utf-8") as f:
    corpus = [line.strip() for line in f if line.strip()]

# 2. Tokenize (tiền xử lý)
tokenized_corpus = [preprocess(doc) for doc in corpus]

# 3. Tạo model BM25
bm25 = BM25Okapi(tokenized_corpus)

# 4. Lưu lại vào file pkl (Ghi đè lên file cũ)
with open("bm25_index.pkl", "wb") as f:
    pickle.dump((bm25, corpus), f)

print("✅ Đã cập nhật Index thành công! Giờ hệ thống đã cực kỳ nhạy bén.")