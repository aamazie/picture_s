#picture_s software

import os
from PIL import Image
from diffusers import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline
import torch

# Function to load an image if a path is provided
def load_image(image_path):
    if os.path.exists(image_path):
        return Image.open(image_path).convert("RGB")
    else:
        print(f"Image at {image_path} not found.")
        return None

# Create the /pix directory if it doesn't exist
output_dir = "pix"
os.makedirs(output_dir, exist_ok=True)

# Load the pre-trained Stable Diffusion model
text2img_pipeline = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5").to("cuda")
img2img_pipeline = StableDiffusionImg2ImgPipeline.from_pretrained("runwayml/stable-diffusion-v1-5").to("cuda")

# Ask for user input
image_path = input("Enter the directory of an image to modify, or press Enter to generate a new image: ").strip()
prompt = input("Enter your prompt: ").strip()

if image_path:
    # Load the input image
    init_image = load_image(image_path)
    if init_image:
        # Modify the existing image based on the prompt
        strength = 0.75  # Degree of modification (0.0-1.0)
        image = img2img_pipeline(prompt=prompt, init_image=init_image, strength=strength, num_inference_steps=50).images[0]
    else:
        print("Proceeding with text-to-image generation.")
        image = text2img_pipeline(prompt).images[0]
else:
    # Generate a new image from the prompt
    image = text2img_pipeline(prompt).images[0]

# Save the image as a JPEG in the /pix folder
output_path = os.path.join(output_dir, "output.jpg")
image.save(output_path)

print(f"Image saved to {output_path}")
