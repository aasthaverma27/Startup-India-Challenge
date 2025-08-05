import os
import uuid
import time
import logging
import requests
import gradio as gr
import torch
import psutil
import GPUtil
from transformers import pipeline
from deep_translator import GoogleTranslator

# Logging
logging.basicConfig(level=logging.INFO)

# Constants
ELEVENLABS_API_KEY = "your api key"
device = "cuda" if torch.cuda.is_available() else "cpu"
device_id = 0 if device == "cuda" else -1
os.makedirs("outputs", exist_ok=True)

# Models
toxicity_classifier = pipeline("text-classification", model="martin-ha/toxic-comment-model", device=device_id)
emotion_classifier = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", return_all_scores=True, device=device_id)
zero_shot_classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli", device=device_id)

# Labels and voices
AMBIE_LABELS = ["news", "personal", "story", "announcement"]
AMBIENCE_TO_VOICE = {
    "news": {
        "voice_id": "EXAVITQu4vr4xnSDxMaL",
        "style": 0.6,
        "stability": 0.7,
        "similarity_boost": 0.85
    },
    "personal": {
        "voice_id": "JS6C6yu2x9Byh4i1a8lX",
        "style": 0.9,
        "stability": 0.4,
        "similarity_boost": 0.75
    },
    "story": {
        "voice_id": "21m00Tcm4TlvDq8ikWAM",
        "style": 0.8,
        "stability": 0.6,
        "similarity_boost": 0.80
    },
    "announcement": {
        "voice_id": "ErXwobaYiN019PkySvjV",
        "style": 0.7,
        "stability": 0.8,
        "similarity_boost": 0.6
    }
}

# Helper functions
def translate_to_english(text):
    return GoogleTranslator(source='auto', target='en').translate(text)

def is_text_toxic(text, threshold=0.7):
    translated = translate_to_english(text)
    results = toxicity_classifier(translated)
    for result in results:
        label = result["label"].lower()
        score = result["score"]
        if label in ["toxic", "toxicity", "toxic_comment", "1"] and score >= threshold:
            return True
    return False

def detect_emotion(text):
    translated = translate_to_english(text)
    results = emotion_classifier(translated)[0]
    if not results:
        return "neutral"
    top_emotion = max(results, key=lambda x: x['score'])
    return top_emotion['label'].lower() if top_emotion['score'] > 0.3 else "neutral"

def detect_ambience(text):
    translated = translate_to_english(text)
    result = zero_shot_classifier(translated, AMBIE_LABELS)
    return result["labels"][0]

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

def get_system_metrics(start_time, end_time):
    latency = end_time - start_time
    process = psutil.Process(os.getpid())
    ram_usage = process.memory_info().rss / (1024 ** 2)
    cpu_usage = psutil.cpu_percent(interval=0.5)
    gpu_usage = "N/A"
    if torch.cuda.is_available():
        gpus = GPUtil.getGPUs()
        if gpus:
            gpu_usage = f"{gpus[0].memoryUsed} MB / {gpus[0].memoryTotal} MB"
    return latency, ram_usage, cpu_usage, gpu_usage

# Main TTS processing function
def process_text(text, selected_ambience):
    if not text.strip():
        return "Please enter some text.", None, ""

    start_time = time.time()

    if is_text_toxic(text):
        return "Toxic or abusive content detected. Please use clean language.", None, ""

    emotion = detect_emotion(text)
    style_value = get_style_from_emotion(emotion)

    ambience = detect_ambience(text) if selected_ambience == "Auto" else selected_ambience.lower()
    voice_settings = AMBIENCE_TO_VOICE.get(ambience, AMBIENCE_TO_VOICE["personal"])
    voice_id = voice_settings["voice_id"]

    url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
    headers = {
        "xi-api-key": ELEVENLABS_API_KEY,
        "Content-Type": "application/json"
    }

    payload = {
        "text": text,
        "model_id": "eleven_multilingual_v2",
        "voice_settings": {
            "stability": voice_settings["stability"],
            "similarity_boost": voice_settings["similarity_boost"],
            "style": style_value,
            "use_speaker_boost": True
        }
    }

    try:
        response = requests.post(url, headers=headers, json=payload)
        end_time = time.time()
        latency, ram, cpu, gpu = get_system_metrics(start_time, end_time)

        metrics = (
            f"Inference Time: {latency:.2f} sec | "
            f"RAM: {ram:.1f} MB | CPU: {cpu:.1f}% | GPU: {gpu}"
        )

        if response.status_code == 200:
            unique_id = str(uuid.uuid4())
            output_path = f"outputs/output_{unique_id}.mp3"
            with open(output_path, "wb") as f:
                f.write(response.content)
            status = f"Emotion: {emotion.capitalize()} | Ambience: {ambience.capitalize()} | Audio generated successfully."
            return status, output_path, metrics
        else:
            return f"Failed to generate audio. API Error: {response.status_code} - {response.text}", None, metrics
    except Exception as e:
        return f"Error: {str(e)}", None, ""

# Gradio UI
interface = gr.Interface(
    fn=process_text,
    inputs=[
        gr.TextArea(label="Enter Hindi or English Text"),
        gr.Dropdown(label="Ambience Type", choices=["Auto", "News", "Personal", "Story", "Announcement"], value="Auto")
    ],
    outputs=[
        gr.Textbox(label="Status"),
        gr.Audio(label="Generated Audio", type="filepath"),
        gr.Textbox(label="🧪 Metrics: Latency, RAM, CPU, GPU")
    ],
    title="Safe Hindi/English TTS with Emotion & Ambience",
    description="Enter any Hindi/English sentence. Detects emotion and ambience, filters toxic content, and generates expressive speech with real-time system metrics."
)

interface.launch()
