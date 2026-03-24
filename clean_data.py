import re

def clean_vietnamese_text(text):
    # 1. Loại bỏ các trích dẫn kiểu [1], [2], [chú thích...]
    text = re.sub(r'\[\d+\]', '', text)
    text = re.sub(r'\[chú thích \d+\]', '', text)
    
    # 2. Loại bỏ các ký tự đặc biệt lạ (giữ lại các dấu câu cơ bản)
    # Bạn có thể thêm các ký tự muốn xóa vào đây
    text = re.sub(r'[\t\r]', ' ', text)
    
    # 3. Chuẩn hóa khoảng trắng (biến nhiều dấu cách thành 1)
    text = re.sub(r'\s+', ' ', text)
    
    # 4. Đảm bảo dấu câu dính liền từ trước và cách từ sau
    # Ví dụ: "Hà Nội ,Việt Nam" -> "Hà Nội, Việt Nam"
    text = re.sub(r'\s+([,.!?;:])', r'\1', text)
    text = re.sub(r'([,.!?;:])(?=[^\s])', r'\1 ', text)
    
    return text.strip()

def process_file(input_path, output_path):
    with open(input_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    cleaned_lines = []
    seen = set() # Để xóa dòng trùng lặp

    for line in lines:
        cleaned = clean_vietnamese_text(line)
        if len(cleaned) > 20: # Chỉ giữ lại những đoạn có nghĩa (trên 20 ký tự)
            if cleaned not in seen:
                cleaned_lines.append(cleaned)
                seen.add(cleaned)

    with open(output_path, 'w', encoding='utf-8') as f:
        for line in cleaned_lines:
            f.write(line + '\n')

    print(f"✅ Đã dọn dẹp xong! Giảm từ {len(lines)} xuống còn {len(cleaned_lines)} dòng.")

if __name__ == "__main__":
    process_file("knowledge_base.txt", "knowledge_base_clean.txt")