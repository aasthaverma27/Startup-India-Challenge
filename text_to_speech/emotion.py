import os
import gradio as gr
import requests
import torch
from transformers import pipeline
from deep_translator import GoogleTranslator

ELEVENLABS_API_KEY = "sk_0355b20cdaa6384a71337cb48a6010167b9b6491fb37c736"
VOICE_ID = "JS6C6yu2x9Byh4i1a8lX"

device = "cuda" if torch.cuda.is_available() else "cpu"
os.makedirs("outputs", exist_ok=True)

toxicity_classifier = pipeline("text-classification", model="martin-ha/toxic-comment-model", device=0 if device=="cuda" else -1)

emotion_classifier = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", return_all_scores=True, device=0 if device=="cuda" else -1)

def translate_to_english(text):
    return GoogleTranslator(source='auto', target='en').translate(text)

def is_text_toxic(text, threshold=0.7):
    translated = translate_to_english(text)
    print(f"[Translated for classification] → {translated}")
    results = toxicity_classifier(translated)
    for result in results:
        label = result["label"].lower()
        score = result["score"]
        print(f"Toxicity Check → Label: {label}, Score: {score:.2f}")
        if label in ["toxic", "toxicity", "toxic_comment", "1"] and score >= threshold:
            return True
    return False

def detect_emotion(text):
    translated = translate_to_english(text)
    results = emotion_classifier(translated)[0]
    top_emotion = max(results, key=lambda x: x['score'])
    print(f"[Emotion Detected] → {top_emotion['label']} ({top_emotion['score']:.2f})")
    return top_emotion['label'].lower()

def get_style_from_emotion(emotion):
    return {
        "joy": 1.0,
        "anger": 0.7,
        "sadness": 0.1,
        "neutral": 0.5,
        "fear": 0.3,
        "disgust": 0.2,
        "surprise": 0.9,
        "others": 0.5
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
    description="Type a sentence. Detects real emotion using a transformer model, blocks toxic content, and generates expressive speech using ElevenLabs."
)

interface.launch()
sou