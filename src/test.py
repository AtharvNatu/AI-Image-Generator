import os
from model_inference import GenAI

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMAGE_DIR = os.path.join(BASE_DIR, "images")

# Model Download Path (WARNING : Every Model Can Take Up To 10-15 Gb Disk Space)
MODEL_DIR = os.path.join(BASE_DIR, "models")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(IMAGE_DIR, exist_ok=True)

# Set Text Prompt
text_prompt = "Jungle Fire"

# Set Image Download(Save) Path
image_path = os.path.join(IMAGE_DIR, "bunnies.png")

# Select Model By Creating Object
# Available Model IDs
# 1) CompVis/stable-diffusion-v1-4

server = GenAI(
    model_id="CompVis/stable-diffusion-v1-4",
    model_path=MODEL_DIR,
    log_progress=True
)

# Generate Image
server.generate_image(
    promptText=text_prompt,
    outputPath=image_path,
    seed=123,
    num_images=2,
    image_format="PNG"
)

