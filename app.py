import gradio as gr
from pipeline import pipeline

def qa_system(question):
    answer, score = pipeline(question)
    if answer:
        return f"{answer} (confidence: {score:.2f})"
    return "Không tìm thấy câu trả lời"

demo = gr.Interface(
    fn=qa_system,
    inputs="text",
    outputs="text",
    title="Vietnamese QA System",
    description="BM25 + Transformer QA"
)

if __name__ == "__main__":
    demo.launch()