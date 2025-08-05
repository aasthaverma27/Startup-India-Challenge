from transformers import pipeline
from TTS.api import TTS
import torch
import os
import gradio as gr

device = "cuda" if torch.cuda.is_available() else "cpu"
os.makedirs("outputs", exist_ok=True)

classifier = pipeline("text-classification", model="unitary/toxic-bert")
tts = TTS(model_name="tts_models/en/ljspeech/fast_pitch").to(device)

def is_text_toxic(text, threshold=0.8):
    results = classifier(text)
    for result in results:
        label = result["label"]
        score = result["score"]
        print(f"Label: {label}, Score: {score:.2f}")
        if label.lower() == "toxic" and score >= threshold:
            return True
    return False

def process_text(text):
    if is_text_toxic(text):
        return "Toxic or abusive content detected.", None
    else:
        audio_path = "./outputs/output.wav"
        tts.tts_to_file(text=text, file_path=audio_path)
        return "Text is clean. Audio generated successfully.", audio_path

interface = gr.Interface(
    fn=process_text,
    inputs=gr.TextArea(label="Enter Text"),
    outputs=[
        gr.Textbox(label="Status"),
        gr.Audio(label="Output Audio", type="filepath")
    ],
    title="Safe Text-to-Speech"
)

interface.launch()
