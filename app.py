import os
import gradio as gr
from transformers import pipeline

classifier = pipeline(
    "sentiment-analysis",
    model="philschmid/MiniLM-L6-H384-uncased-sst2"
)


def analyze_sentiment(text):
    result = classifier(text)

    if result[0]["label"] == "LABEL_1":
        return "Positive"
    else:
        return "Negative"


app = gr.Interface(
    fn=analyze_sentiment,
    inputs=gr.Textbox(label="Enter Text"),
    outputs=gr.Textbox(label="Sentiment"),
    title="Sentiment Analysis App",
    description="Enter a sentence to check if it is Positive or Negative."
)


if __name__ == "__main__":
    app.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", 7860))
    )
