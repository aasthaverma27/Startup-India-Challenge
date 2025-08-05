import os
import gradio as gr
import requests
import torch
from transformers import pipeline
from deep_translator import GoogleTranslator

ELEVENLABS_API_KEY = "your api key"
VOICE_ID = "JS6C6yu2x9Byh4i1a8lX"

device = "cuda" if torch.cuda.is_available() else "cpu"
os.makedirs("outputs", exist_ok=True)
classifier = pipeline("text-classification", model="martin-ha/toxic-comment-model")

def translate_to_english(text):
    return GoogleTranslator(source='auto', target='en').translate(text)

def is_text_toxic(text, threshold=0.7):
    translated = translate_to_english(text)
    print(f"[Translated for classification] → {translated}")
    results = classifier(translated)
    for result in results:
        label = result["label"].lower()
        score = result["score"]
        print(f"Label: {label}, Score: {score:.2f}")
        if label in ["toxic", "toxicity", "toxic_comment", "1"] and score >= threshold:
            return True
    return False

def detect_emotion(text):
    text = text.lower()
    if "happy" in text or "खुश" in text:
        return "happy"
    elif "angry" in text or "गुस्सा" in text:
        return "angry"
    elif "sad" in text or "उदास" in text:
        return "sad"
    else:
        return "neutral"

def get_style_from_emotion(emotion):
    return {
        "happy": 1.0,
        "angry": 0.7,
        "sad": 0.1,
        "neutral": 0.5
    }.get(emotion, 0.5)

def process_text(text):
    if not text.strip():
        return "❌ Please enter some text.", None

    if is_text_toxic(text):
        return "⚠️ Toxic or abusive content detected. Please use clean language.", None

    emotion = detect_emotion(text)
    style_value = get_style_from_emotion(emotion)

    url = f"https://api.elevenlabs.io/v1/text-to-speech/{VOICE_ID}"
    headers = {
        "xi-api-key": ELEVENLABS_API_KEY,
        "Content-Type": "application/json"
    }

    payload = {
        "text": text,
        "model_id": "eleven_multilingual_v2",
        "voice_settings": {
            "stability": 0.5,
            "similarity_boost": 0.75,
            "style": style_value,
            "use_speaker_boost": True
        }
    }

    try:
        response = requests.post(url, headers=headers, json=payload)
        if response.status_code == 200:
            output_path = "outputs/output.mp3"
            with open(output_path, "wb") as f:
                f.write(response.content)
            return f"✅ Emotion: {emotion.capitalize()} | Audio generated successfully.", output_path
        else:
            return f"❌ Failed to generate audio. API Error: {response.status_code} - {response.text}", None
    except Exception as e:
        return f"❌ Error: {str(e)}", None

interface = gr.Interface(
    fn=process_text,
    inputs=gr.TextArea(label="Enter Hindi or English Text"),
    outputs=[
        gr.Textbox(label="Status"),
        gr.Audio(label="Generated Audio", type="filepath")
    ],
    title="Safe Hindi/English TTS with Emotion & Toxic Filter",
    description="Type a sentence. Detects emotion (happy, sad, angry), blocks toxic content, and generates expressive speech."
)

interface.launch()
