import os
from datasets import load_dataset
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from transformers import AutoProcessor, AutoModelForImageTextToText
import torch

# === Load model and processor ===
model_id = "HuggingFaceTB/SmolVLM-256M-Instruct"
processor = AutoProcessor.from_pretrained(model_id)
model = AutoModelForImageTextToText.from_pretrained(model_id).eval()
vision_encoder = model.model.vision_model

# === Load Merve dataset ===
dataset = load_dataset("merve/vqav2-small", split="validation[:2]")

# === Patch and model info ===
patch_size = model.config.vision_config.patch_size  # 16
hidden_size = model.config.vision_config.hidden_size  # 768

# === Font setup for overlay ===
try:
    font = ImageFont.truetype("DejaVuSans.ttf", 12)
except:
    font = ImageFont.load_default()

def overlay_patch_indices(image: Image.Image, patch_size: int) -> Image.Image:
    """Overlay every 4th patch index over the image."""
    draw = ImageDraw.Draw(image)
    w, h = image.size
    num_x = w // patch_size
    num_y = h // patch_size
    idx = 0
    for y in range(num_y):
        for x in range(num_x):
            if idx % 4 != 0:
                idx += 1
                continue
            cx = x * patch_size + 4
            cy = y * patch_size + 2
            draw.text((cx + 1, cy + 1), str(idx), fill="black", font=font)
            draw.text((cx, cy), str(idx), fill="white", font=font)
            idx += 1
    return image

# === Plot setup ===
fig, axs = plt.subplots(len(dataset), 2, figsize=(12, 4 * len(dataset)))

# === Process and visualize ===
for i, sample in enumerate(dataset):
    # 1. Original image and question
    orig_img = sample["image"].convert("RGB").resize((512, 512))
    question = sample["question"]
    prompt = f"<image>\nQuestion: {question}\nAnswer:"

    # 2. Processor (image + text)
    inputs = processor(images=orig_img, text=prompt, return_tensors="pt")
    img_tensor = inputs["pixel_values"][:, 0]  # only slot 0 image

    # 3. Run vision encoder only
    with torch.no_grad():
        vision_out = vision_encoder(img_tensor)
    tokens = vision_out.last_hidden_state.squeeze(0).cpu().numpy()  # [num_patches, hidden_size]

    # 4. PCA projection
    pca = PCA(n_components=2)
    proj = pca.fit_transform(tokens)
    patch_indices = np.arange(tokens.shape[0])

    # 5. Left: image overlay with patch IDs
    patch_img = overlay_patch_indices(orig_img.copy(), patch_size)
    axs[i, 0].imshow(patch_img)
    axs[i, 0].set_title(f"Image {i+1}: Patch Index Overlay\nQ: {question}")
    axs[i, 0].axis("off")

    # 6. Right: PCA colored by patch index
    sc = axs[i, 1].scatter(proj[:, 0], proj[:, 1], c=patch_indices, cmap="plasma")
    axs[i, 1].set_title(f"Image {i+1}: PCA of Vision Tokens")
    axs[i, 1].set_xlabel("PC1"); axs[i, 1].set_ylabel("PC2")
    axs[i, 1].grid(True)
    fig.colorbar(sc, ax=axs[i, 1], label="Patch Index")

plt.tight_layout()
output_path = "../outputs/Merve patch index and pca.png"
os.makedirs(os.path.dirname(output_path), exist_ok=True)
plt.savefig(output_path)
print(f"Saved visualization to: {output_path}")
plt.show()
