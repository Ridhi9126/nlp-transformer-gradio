import os
import gradio as gr
from transformers import pipeline


# Sentiment Analysis model
sentiment = pipeline(
    "sentiment-analysis",
    model="lxyuan/distilbert-base-multilingual-cased-sentiments-student"
)


def analyze_sentiment(text):
    result = sentiment(text)

    return result[0]["label"], result[0]["score"]


# Gradio Interface
with gr.Blocks() as demo:

    gr.Markdown("# NLP Transformer - Sentiment Analysis")

    gr.Markdown(
        "Enter a sentence and the Transformer model will "
        "predict whether the sentiment is positive or negative."
    )

    text_input = gr.Textbox(
        label="Enter your text",
        placeholder="I really enjoyed this movie!"
    )

    analyze_button = gr.Button("Analyze Sentiment")

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
