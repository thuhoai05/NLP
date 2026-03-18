from rank_bm25 import BM25Okapi
import pickle
import string
from pyvi import ViTokenizer

# Dùng lại hàm preprocess giống hệt lúc build index
def preprocess_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokenized_text = ViTokenizer.tokenize(text)
    return tokenized_text.split()

print("Loading BM25 index...")
with open("bm25_index.pkl", "rb") as f:
    bm25, corpus = pickle.load(f)
print("BM25 ready!")

def retrieve(query, top_k=10):
    # Cập nhật stopwords theo dạng từ ghép (có dấu gạch dưới)
    stopwords = {"là", "của", "nào", "và", "được", "trong", "với", "những"}
    
    # Tiền xử lý query
    raw_tokens = preprocess_text(query)
    
    # Lọc stopwords
    tokenized_query = [w for w in raw_tokens if w not in stopwords]
    
    # Lấy điểm số
    scores = bm25.get_scores(tokenized_query)
    
    # Lấy top k
    top_n = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
    
    results = [corpus[i] for i in top_n]
    return results

# Test thử
query = "Ai là cha của Donald Trump"
contexts = retrieve(query)

for i, c in enumerate(contexts):
    print(f"\n--- Context {i+1} ---")
    print(c)