import torch
import os
from TTS.api import TTS
import gradio as gr 
device="cuda" if torch.cuda.is_available() else "cpu"

my_text="this is sample audio"
def create_audio(text=my_text) :
    tts = TTS(model_name="tts_models/en/ljspeech/fast_pitch").to(device)
    return tts.tts_to_file(text=text,file_path='./outputs/blue.wav')
create_audio()

interface=gr.Interface(fn=create_audio,
                       inputs=gr.TextArea(label="Enter Text"),
                       outputs=gr.Audio(label='Output Audio'))

interface.launch()