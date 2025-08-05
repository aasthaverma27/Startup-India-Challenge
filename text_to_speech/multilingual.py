from transformers import VitsModel, AutoTokenizer, pipeline
import torch
import os
import gradio as gr
import scipy.io.wavfile as wavfile
import tempfile
from gtts import gTTS

device = "cuda" if torch.cuda.is_available() else "cpu"
os.makedirs("outputs", exist_ok=True)

try:
    classifier = pipeline("text-classification", model="unitary/toxic-bert")
except:
    classifier = pipeline("text-classification", model="martin-ha/toxic-comment-model")

model_name = "facebook/mms-tts-hin"  #hindi model
model = VitsModel.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
model.to(device)

def is_text_toxic(text, threshold=0.5):
    results = classifier(text)
    print(f"Toxicity check results: {results}")
    
    if isinstance(results, list):
        for result in results:
            label = result["label"]
            score = result["score"]
            print(f"Label: {label}, Score: {score:.2f}")
            if label.lower() in ["toxic", "toxicity", "1"] and score >= threshold:
                return True
    else:
        label = results["label"]
        score = results["score"]
        print(f"Label: {label}, Score: {score:.2f}")
        if label.lower() in ["toxic", "toxicity", "1"] and score >= threshold:
            return True
    
    return False

def process_text(text):
    if not text.strip():
        return "Please enter some text to convert to speech.", None
    if is_text_toxic(text):
        return "Toxic or abusive content detected.", None
    
    try:
        inputs = tokenizer(text, return_tensors="pt").to(device)
        with torch.no_grad():
            output = model(inputs["input_ids"])
        waveform = output.waveform.squeeze().cpu().numpy()
        audio_path = "./outputs/output.wav"
        sample_rate = model.config.sampling_rate
        wavfile.write(audio_path, sample_rate, waveform)
        return "Text is clean. Audio generated successfully.", audio_path
        
    except Exception as e:
        try:
            tts = gTTS(text=text, lang='hi', slow=False)
            audio_path = "./outputs/output.mp3"
            tts.save(audio_path)
            return "Text is clean. Audio generated successfully with Google TTS.", audio_path
        except:
            return f"Error generating audio: {str(e)}", None

interface = gr.Interface(
    fn=process_text,
    inputs=gr.TextArea(label="Enter Hindi Text"),
    outputs=[
        gr.Textbox(label="Status"),
        gr.Audio(label="Output Audio", type="filepath")
    ],
    title="Safe Hindi Text-to-Speech"
)

interface.launch()