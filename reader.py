from transformers import AutoTokenizer, AutoModelForQuestionAnswering
import torch
model_name = "./my_finetuned_vimmrc"
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
model = AutoModelForQuestionAnswering.from_pretrained(model_name)

def answer_question(question, context):
    # Bây giờ hàm này sẽ hiểu 'tokenizer' và 'model' là gì
    inputs = tokenizer(question, context, return_tensors="pt", truncation=True, max_length=512)
    
    with torch.no_grad():
        outputs = model(**inputs)

    start_probs = torch.softmax(outputs.start_logits, dim=1)
    end_probs = torch.softmax(outputs.end_logits, dim=1)

    start_idx = torch.argmax(start_probs)
    end_idx = torch.argmax(end_probs) + 1
    
    confidence = (torch.max(start_probs) + torch.max(end_probs)).item() / 2

    answer = tokenizer.decode(inputs["input_ids"][0][start_idx:end_idx], skip_special_tokens=True)
    answer = answer.strip()

    if confidence < 0.25 or not answer:
        return None, 0
        
    return answer, confidence