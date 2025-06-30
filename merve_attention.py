import os
import random
import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoProcessor, AutoModelForVision2Seq
import matplotlib.pyplot as plt
from PIL import Image

# === Set seeds for reproducibility ===
def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)

seed_everything(42)

# === Create output directory ===
output_dir = "attention_output"
os.makedirs(output_dir, exist_ok=True)

# === Model & processor ===
model_id = "HuggingFaceTB/SmolVLM-256M-Instruct"
model = AutoModelForVision2Seq.from_pretrained(model_id).eval()
processor = AutoProcessor.from_pretrained(model_id)
vision_encoder = model.model.vision_model
text_encoder = model.model.text_model

# === Device ===
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)

# === Projection from text hidden size (576) to vision hidden size (768) ===
torch.manual_seed(42)
text_to_vision_proj = torch.nn.Linear(576, 768).to(device)

# === Load dataset ===
dataset = load_dataset("merve/vqav2-small", split="validation[:7]")

# === Vision model details ===
patch_size = model.config.vision_config.patch_size
hidden_size = model.config.vision_config.hidden_size
grid_size = 512 // patch_size  # e.g., 32

# === Process and visualize ===
for i, sample in enumerate(dataset):
    image = sample["image"].convert("RGB").resize((512, 512))
    question = sample["question"]

    # === Add <image> token to prompt ===
    prompt = "<image> " + question
    images = [image]

    # === repare input for model.generate() ===
    inputs = processor(images=images, text=prompt, return_tensors="pt").to(device)

    # === Generate model output ===
    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=20)
        answer = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

    # === Vision encoder ===
    pixel_values = inputs["pixel_values"][:, 0]  # first image slot
    with torch.no_grad():
        vision_out = vision_encoder(pixel_values)
    patch_embeddings = vision_out.last_hidden_state[0][:-1]  # [1023, 768]

    # === Text encoder ===
    text_inputs = processor.tokenizer(question, return_tensors="pt", padding=True, truncation=True).to(device)
    with torch.no_grad():
        text_outputs = text_encoder(**text_inputs)
    question_embedding = text_outputs.last_hidden_state.mean(dim=1)
    question_embedding_proj = text_to_vision_proj(question_embedding)

    # === Cosine similarity ===
    patch_norm = F.normalize(patch_embeddings, dim=-1)
    question_norm = F.normalize(question_embedding_proj, dim=-1)
    sim = torch.matmul(patch_norm, question_norm.squeeze(0))

    if sim.shape[0] != 1024:
        sim = torch.cat([sim, torch.zeros(1024 - sim.shape[0], device=sim.device)])
    sim_map = sim.view(32, 32).detach().cpu().numpy()

    # === Plot and save ===
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))

    axs[0].imshow(image)
    axs[0].set_title(f"Image {i+1}\nQ: {question}", fontsize=10)
    axs[0].axis("off")

    im = axs[1].imshow(sim_map, cmap="plasma")
    axs[1].set_title(f"Patch Similarity\nA: {answer}", fontsize=10)
    plt.colorbar(im, ax=axs[1])
    plt.suptitle(f"Image {i+1}", fontsize=14)
    plt.tight_layout()

    filename = os.path.join(output_dir, f"image_{i+1}_attention.png")
    plt.savefig(filename)
    plt.close()
    print(f"Saved: {filename} | Answer: {answer}")
