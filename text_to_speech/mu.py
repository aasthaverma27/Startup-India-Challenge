from transformers import pipeline
from TTS.api import TTS
import torch
import os
import gradio as gr


device = "cuda" if torch.cuda.is_available() else "cpu"
os.makedirs("outputs", exist_ok=True)

classifier = pipeline("text-classification", model="facebook/roberta-hate-speech-dynabench-r4-target")

tts = TTS(model_name="tts_models/multilingual/multi-dataset/xtts_v2").to(device)

def is_text_toxic(text, threshold=0.6):
    results = classifier(text)
    for result in results:
        label = result["label"].lower()
        score = result["score"]
        print(f"Label: {label}, Score: {score:.2f}")
        if ("hate" in label or "offensive" in label or "toxic" in label) and score >= threshold:
            return True
    return False


def process_text(text):
    if is_text_toxic(text):
        return "⚠️ Toxic or abusive content detected. No audio generated.", None
    else:
        audio_path = "./outputs/output.wav"
        tts.tts_to_file(text=text, file_path=audio_path)
        return "✅ Text is clean. Audio generated successfully.", audio_path

interface = gr.Interface(
    fn=process_text,
    inputs=gr.TextArea(label="Enter Text in Any Language"),
    outputs=[
        gr.Textbox(label="Status"),
        gr.Audio(label="Output Audio", type="filepath")
    ],
    title="Multilingual Safe Text-to-Speech",
    description="Detects toxicity in various languages and converts clean text to speech."
)

interface.launch()
