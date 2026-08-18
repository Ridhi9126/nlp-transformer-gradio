import os
import gradio as gr
from transformers import pipeline


# Model will be loaded only when needed
sentiment_model = None


def analyze_sentiment(text):
    global sentiment_model

    if not text.strip():
        return "Please enter some text.", 0.0

    # Load model only on first prediction
    if sentiment_model is None:
        sentiment_model = pipeline(
            "sentiment-analysis",
            model="M-FAC/bert-tiny-finetuned-sst2"
        )

    result = sentiment_model(text)[0]

    return result["label"], result["score"]


with gr.Blocks() as demo:

    gr.Markdown(
        "# 🧠 NLP Sentiment Analysis"
    )

    gr.Markdown(
        "Enter a sentence and the Transformer model "
        "will predict whether the sentiment is positive or negative."
    )

    text_input = gr.Textbox(
        label="Enter your text",
        placeholder="I really enjoyed this movie!"
    )

    analyze_button = gr.Button(
        "Analyze Sentiment"
    )

    sentiment_output = gr.Textbox(
        label="Sentiment"
    )

    confidence_output = gr.Number(
        label="Confidence"
    )

    analyze_button.click(
        fn=analyze_sentiment,
        inputs=text_input,
        outputs=[
            sentiment_output,
            confidence_output
        ]
    )


demo.launch(
    server_name="0.0.0.0",
    server_port=int(os.environ.get("PORT", 10000))
)
