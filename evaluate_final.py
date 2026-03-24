# đánh giá song song cả 2 mô hình
import collections
import string
from tqdm import tqdm
from datasets import load_dataset

from reader_extractive import answer_question_extractive
from reader import answer_question as answer_question_generative


def normalize_answer(s):
    """Chuẩn hóa text"""
    def white_space_fix(text): return " ".join(text.split())
    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)
    def lower(text): return text.lower()
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

    precision = num_same / len(pred_tokens)
    recall = num_same / len(gt_tokens)
    return (2 * precision * recall) / (precision + recall)


def compute_exact(prediction, ground_truth):
    return int(normalize_answer(prediction) == normalize_answer(ground_truth))


# --- LOAD DATA ---
print("🚀 Đang tải tập dữ liệu TEST...")
dataset = load_dataset("ntphuc149/ViSpanExtractQA")
test_data = dataset["test"]

# 👉 CHỌN SỐ LƯỢNG TEST (khuyên: 1000)
test_data = test_data.select(range(5000))

scores = {
    "extractive": {"f1": 0.0, "em": 0.0},
    "generative": {"f1": 0.0, "em": 0.0}
}

total_samples = len(test_data)
print(f"📊 Đang đánh giá trên {total_samples} mẫu...")

# --- EVALUATE ---
for ex in tqdm(test_data):

    # ✅ FIX: dataset này dùng answer_text
    gt = ex["answer_text"] if ex["answer_text"] else ""

    # ===== Extractive =====
    pred_ext, _ = answer_question_extractive(ex["question"], ex["context"])
    pred_ext = pred_ext.strip()

    # cắt gọn tránh dư câu
    pred_ext = pred_ext.split(".")[0]

    scores["extractive"]["f1"] += compute_f1(pred_ext, gt)
    scores["extractive"]["em"] += compute_exact(pred_ext, gt)

    # ===== Generative =====
    pred_gen, _ = answer_question_generative(ex["question"], ex["context"])
    pred_gen = pred_gen.strip() if pred_gen else ""

    scores["generative"]["f1"] += compute_f1(pred_gen, gt)
    scores["generative"]["em"] += compute_exact(pred_gen, gt)


# --- TÍNH % ---
for model_type in scores:
    scores[model_type]["f1"] = (scores[model_type]["f1"] / total_samples) * 100
    scores[model_type]["em"] = (scores[model_type]["em"] / total_samples) * 100


# --- IN KẾT QUẢ ---
print("\n" + "="*50)
print(f"🏆 KẾT QUẢ SO SÁNH (Test: {total_samples} câu)")
print("-"*50)
print(f"{'Mô hình':<20} | {'Exact Match (EM)':<15} | {'F1-Score':<15}")
print("-"*50)
print(f"{'Extractive (BERT)':<20} | {scores['extractive']['em']:>13.2f}% | {scores['extractive']['f1']:>13.2f}%")
print(f"{'Generative (ViT5)':<20} | {scores['generative']['em']:>13.2f}% | {scores['generative']['f1']:>13.2f}%")
print("="*50)