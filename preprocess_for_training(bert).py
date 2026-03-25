from datasets import load_dataset
from transformers import AutoTokenizer

model_checkpoint = "nguyenvulebinh/vi-mrc-base"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

def preprocess_function(examples):
    questions = [q.strip() for q in examples["question"]]
    inputs = tokenizer(
        questions, examples["context"],
        max_length=384, truncation="only_second",
        return_offsets_mapping=True, padding="max_length",
    )
    offset_mapping = inputs.pop("offset_mapping")
    start_positions, end_positions = [], []

    for i, offset in enumerate(offset_mapping):
        context, answer = examples["context"][i], examples["answer_text"][i]
        if isinstance(answer, list): answer = answer[0]
        start_char = context.find(answer)
        if start_char == -1 or answer == "":
            start_positions.append(0)
            end_positions.append(0)
            continue
        end_char = start_char + len(answer)
        sequence_ids = inputs.sequence_ids(i)
        idx = 0
        while idx < len(sequence_ids) and sequence_ids[idx] != 1: idx += 1
        context_start = idx
        while idx < len(sequence_ids) and sequence_ids[idx] == 1: idx += 1
        context_end = idx - 1
        if context_start >= len(sequence_ids) or offset[context_start][0] > end_char or offset[context_end][1] < start_char:
            start_positions.append(0)
            end_positions.append(0)
        else:
            curr_idx = context_start
            while curr_idx <= context_end and offset[curr_idx][0] <= start_char: curr_idx += 1
            start_positions.append(curr_idx - 1)
            curr_idx = context_end
            while curr_idx >= context_start and offset[curr_idx][1] >= end_char: curr_idx -= 1
            end_positions.append(curr_idx + 1)
    inputs["start_positions"], inputs["end_positions"] = start_positions, end_positions
    return inputs

# Chạy tiền xử lý (Trên Colab chạy vèo cái là xong)
raw_datasets = load_dataset("ntphuc149/ViSpanExtractQA")
raw_datasets["train"] = raw_datasets["train"].select(range(15000))
raw_datasets["validation"] = raw_datasets["validation"].select(range(2000))
tokenized_datasets = raw_datasets.map(preprocess_function, batched=True)
