from rank_bm25 import BM25Okapi
import pickle
import string
from pyvi import ViTokenizer
import os

def preprocess_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokenized_text = ViTokenizer.tokenize(text)
    return tokenized_text.split()

print("Loading BM25 index...")

BASE_DIR = os.path.dirname(__file__)
index_path = os.path.join(BASE_DIR, "bm25_index.pkl")

with open(index_path, "rb") as f:
    bm25, corpus = pickle.load(f)

print("BM25 ready!")

def retrieve(query, top_k=5):
    # ❗ GIỮ LẠI nhiều từ hơn (đừng filter mạnh quá)
    tokenized_query = preprocess_text(query)

    scores = bm25.get_scores(tokenized_query)

    top_n = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]

    return [corpus[i] for i in top_n]