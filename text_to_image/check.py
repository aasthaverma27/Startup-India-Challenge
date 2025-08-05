# main.py

import torch
from diffusers import StableDiffusionPipeline
from PIL import Image, ImageDraw, ImageFont
import gradio as gr

# ✅ Load Stable Diffusion v1.4
pipe = StableDiffusionPipeline.from_pretrained(
    "CompVis/stable-diffusion-v1-4",
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    revision="fp16" if torch.cuda.is_available() else None,
)
pipe.to("cuda" if torch.cuda.is_available() else "cpu")
pipe.enable_attention_slicing()


# ✏️ Function to overlay text on image
def overlay_text(image: Image.Image, text: str) -> Image.Image:
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype("arial.ttf", 60)
    except:
        font = ImageFont.load_default()

    # Center the text at bottom
    bbox = draw.textbbox((0, 0), text, font=font)
    w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]

    position = ((image.width - w) // 2, image.height - h - 30)
    draw.text(position, text, fill=(255, 0, 0), font=font)  # Red text
    return image


def generate_poster(prompt, caption):
    prompt += ", poster, plain background, centered composition"
    image = pipe(prompt=prompt, guidance_scale=7.5, num_inference_steps=25).images[0]
    final = overlay_text(image, caption)
    return final


with gr.Blocks() as demo:
    gr.Markdown("## 🧠 Poster Generator (Stable Diffusion + English Text)")

    with gr.Row():
        prompt = gr.Textbox(label="Scene Prompt", placeholder="e.g. Sunset background")
        caption = gr.Textbox(label="Text to overlay", placeholder="e.g. THINK BIG")
        btn = gr.Button("Generate Poster")

    output = gr.Image(label="Final Poster")
    btn.click(fn=generate_poster, inputs=[prompt, caption], outputs=output)

demo.launch()
