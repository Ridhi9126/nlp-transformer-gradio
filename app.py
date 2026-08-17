import os
import gradio as gr
from transformers import pipeline

sentiment = pipeline("sentiment-analysis")


def analyze_sentiment(text):
    result = sentiment(text)

    return result[0]["label"], result[0]["score"]

generator = pipeline(
    "text-generation",
    model="gpt2"
)


def generate_text(prompt):
    result = generator(
        prompt,
        max_new_tokens=50,
        temperature=0.7
    )

    return result[0]["generated_text"]

with gr.Blocks() as demo:

    gr.Markdown("# NLP Transformer Application")

    with gr.Tab("Sentiment Analysis"):

        sentiment_input = gr.Textbox(
            label="Enter text",
            placeholder="I really enjoyed this movie!"
        )

        sentiment_button = gr.Button("Analyze Sentiment")

        sentiment_output = gr.Textbox(
            label="Sentiment"
        )

        confidence_output = gr.Number(
            label="Confidence"
        )

        sentiment_button.click(
            fn=analyze_sentiment,
            inputs=sentiment_input,
            outputs=[
                sentiment_output,
                confidence_output
            ]
        )

    with gr.Tab("Text Generation"):

        generation_input = gr.Textbox(
            label="Enter prompt",
            placeholder="Artificial Intelligence is..."
        )

        generation_button = gr.Button("Generate")

        generation_output = gr.Textbox(
            label="Generated Text",
            lines=8
        )

        generation_button.click(
            fn=generate_text,
            inputs=generation_input,
            outputs=generation_output
        )
        
demo.launch(
    server_name="0.0.0.0",
    server_port=int(os.environ.get("PORT", 7860))
)