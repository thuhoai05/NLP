# 🧠 Vietnamese Question Answering System using RAG

## 📌 Introduction
This project implements a **Question Answering (QA) system** for Vietnamese using the **Retrieval-Augmented Generation (RAG)** approach.

The system combines:
- 🔎 **Retriever (BM25)** to find relevant context
- 🤖 **Reader models**:
  - Extractive: BERT
  - Generative: ViT5

---

## 🏗️ System Architecture

Pipeline:

1. Input Question  
2. Retriever (BM25)  
3. Reader (BERT / ViT5)  
4. Output Answer  

👉 Uses **Sparse Retrieval (BM25)** – traditional RAG

---

## 📂 Project Structure
NLP/
│── data/
│── preprocessing/
│── retriever/
│── models/
│── training/
│── evaluation/
│── notebooks/
│── app.py
│── reindex.py # 🔥 build BM25 index
│── requirements.txt
---

## ⚙️ Setup & Run

### 1. Install dependencies
```bash
pip install -r requirements.txt
### 2. Build Index (IMPORTANT)

Before running the QA system, you must build the BM25 index:

python reindex.py
### 3. Train Models (optional)
python train.py
### 4. Run QA system
python app.py
📊 Dataset
Format: (context, question, answer)
Context: ~200–300 tokens
Answer: ~5–15 tokens
Mostly factoid questions
⚙️ Methods
🔹 Retriever
BM25 (Sparse Retrieval)
Fast but keyword-based
🔹 BERT (Extractive)
Predict answer span from context
🔹 ViT5 (Generative)
Generate answer text
Flexible but may hallucinate
