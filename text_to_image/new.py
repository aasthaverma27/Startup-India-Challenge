### File: main.py

import gradio as gr
from diffusers import StableDiffusionPipeline
import torch
from authtoken import auth_token

# Load model
model_id = "runwayml/stable-diffusion-v1-5"
device = "cuda" if torch.cuda.is_available() else "cpu"

pipe = StableDiffusionPipeline.from_pretrained(
    model_id,
    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    token=auth_token
)
pipe.to(device)
pipe.enable_attention_slicing()

# Define image generation function
def generate_image(prompt):
    if not prompt.strip():
        return None
    image = pipe(prompt, guidance_scale=8.5, num_inference_steps=30).images[0]
    return image

# Create Gradio interface with styling
with gr.Blocks() as demo:
    gr.Markdown(
        """
        # 🎨 Stable Bud
        _Generate stunning images from text using Stable Diffusion_

        Just enter your creative prompt below and let the AI do the rest!
        """
    )

    with gr.Row():
        with gr.Column(scale=2):
            prompt = gr.Textbox(label="Enter your prompt", placeholder="e.g. A futuristic city at sunset", lines=2)
            generate_btn = gr.Button("Generate Image 🎨")

        with gr.Column(scale=3):
            output_image = gr.Image(label="Generated Image")

    generate_btn.click(fn=generate_image, inputs=prompt, outputs=output_image, show_progress=True)

# Launch the app
demo.launch()

