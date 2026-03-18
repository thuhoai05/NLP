import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datasets import load_dataset

# Bước 1: Load dữ liệu thực tế
dataset = load_dataset("ntphuc149/ViSpanExtractQA")
train_df = pd.DataFrame(dataset['train'])

def run_eda(df):
    print("--- 1. Thống kê số lượng ---")
    total_samples = len(df)
    print(f"Tổng số mẫu dữ liệu: {total_samples}")

    # 2. Tính độ dài (tính theo số từ - whitespace)
    df['question_len'] = df['question'].apply(lambda x: len(str(x).split()))
    df['context_len'] = df['context'].apply(lambda x: len(str(x).split()))

    print("\n--- 2. Thống kê độ dài ---")
    print(df[['question_len', 'context_len']].describe())

    # 3. Vẽ biểu đồ phân bố (Phần này đưa vào báo cáo cực đẹp)
    plt.figure(figsize=(12, 5))

    # Biểu đồ cho câu hỏi
    plt.subplot(1, 2, 1)
    sns.histplot(df['question_len'], kde=True, color='blue')
    plt.title('Phân bố độ dài Câu hỏi')
    plt.xlabel('Số lượng từ')
    plt.ylabel('Tần suất')

    # Biểu đồ cho đoạn văn
    plt.subplot(1, 2, 2)
    sns.histplot(df['context_len'], kde=True, color='green')
    plt.title('Phân bố độ dài Đoạn văn (Context)')
    plt.xlabel('Số lượng từ')
    plt.ylabel('Tần suất')

    plt.tight_layout()
    plt.show()

# Giả sử 'train_df' là dữ liệu bạn load từ dataset ntphuc149/ViSpanExtractQA
# Bước 2: Gọi hàm để nó thực hiện thống kê và vẽ
run_eda(train_df)