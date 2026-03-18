import collections
import string
import re
from tqdm import tqdm
from datasets import load_dataset
from reader import answer_question 

def normalize_answer(s):
    """Loại bỏ dấu câu, mạo từ và khoảng trắng thừa"""
    def white_space_fix(text):
        return " ".join(text.split())
    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)
    def lower(text):
        return text.lower()
    return white_space_fix(remove_punc(lower(s)))

def compute_f1(prediction, ground_truth):
    pred_tokens = normalize_answer(prediction).split()
    gt_tokens = normalize_answer(ground_truth).split()
    
    if len(pred_tokens) == 0 or len(gt_tokens) == 0:
        return int(pred_tokens == gt_tokens)
    
    common = collections.Counter(pred_tokens) & collections.Counter(gt_tokens)
    num_same = sum(common.values())
    
    if num_same == 0:
        return 0
    
    precision = 1.0 * num_same / len(pred_tokens)
    recall = 1.0 * num_same / len(gt_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1

def compute_exact(prediction, ground_truth):
    return int(normalize_answer(prediction) == normalize_answer(ground_truth))

# --- BẮT ĐẦU ĐÁNH GIÁ ---
print("🚀 Đang tải tập dữ liệu TEST...")
dataset = load_dataset("ntphuc149/ViSpanExtractQA")
test_data = dataset["test"]

# Lưu ý: Nếu chạy CPU chậm, có thể chọn test 100-200 câu trước
# test_data = test_data.select(range(100)) 

f1_total = 0
em_total = 0

print(f"📊 Đang đánh giá trên {len(test_data)} mẫu...")
for ex in tqdm(test_data):
    # Lấy câu trả lời từ mô hình của bạn
    pred, _ = answer_question(ex["question"], ex["context"])
    pred = pred if pred else "" # Tránh lỗi None
    
    f1_total += compute_f1(pred, ex["answer_text"])
    em_total += compute_exact(pred, ex["answer_text"])

final_f1 = (f1_total / len(test_data)) * 100
final_em = (em_total / len(test_data)) * 100

print("\n" + "="*30)
print(f"✅ KẾT QUẢ ĐÁNH GIÁ CUỐI CÙNG:")
print(f"📍 Exact Match (EM): {final_em:.2f}%")
print(f"📍 F1-Score:         {final_f1:.2f}%")
print("="*30)