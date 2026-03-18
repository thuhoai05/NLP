from datasets import load_dataset
from rank_bm25 import BM25Okapi
import pickle
import string
from pyvi import ViTokenizer

def preprocess_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    return ViTokenizer.tokenize(text).split()

print("Loading ViSpanExtractQA dataset...")
dataset = load_dataset("ntphuc149/ViSpanExtractQA")

# 1. Thu thập tất cả context từ các tập (train, validation, test)
all_contexts = []
for split in dataset.keys():
    all_contexts.extend(dataset[split]["context"])

# 2. Loại bỏ các context trùng lặp để index nhẹ và chính xác hơn
unique_contexts = list(set(all_contexts))
print(f"Total unique contexts to index: {len(unique_contexts)}")

print("Tokenizing...")
tokenized_corpus = [preprocess_text(doc) for doc in unique_contexts]

print("Building BM25 index...")
bm25 = BM25Okapi(tokenized_corpus)

# Lưu cả unique_contexts thay vì corpus cũ
with open("bm25_index.pkl", "wb") as f:
    pickle.dump((bm25, unique_contexts), f, protocol=pickle.HIGHEST_PROTOCOL)

print("Done! Index is ready for ViSpanExtractQA.")