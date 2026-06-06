import gradio as gr
import sys
sys.path.append("sentiment-analysis")

from src.utils import load_model, predict

model, tokenizer = load_model(
    checkpoint_path="checkpoints/best.pt",
    model_checkpoint="distilbert-base-uncased-finetuned-sst-2-english",
    dropout=0.3
)

def classify(text):
    label, confidence = predict(text, model, tokenizer)
    return f"{label.upper()} ({confidence:.1%})"

demo = gr.Interface(
    fn=classify,
    inputs=gr.Textbox(
        lines=4,
        placeholder="Write a movie review here..."
    ),
    outputs=gr.Text(label="Sentiment"),
    title="Movie Sentiment Classifier - DistilBERT",
    description="DistilBERT fine-tuned on 50K IMDB reviews — 92.13% accuracy",
    examples=[
        ["This movie was absolutely amazing!"],
        ["Terrible waste of time, avoid at all costs."],
        ["Great acting but the story was boring."]
    ]
)

demo.launch()