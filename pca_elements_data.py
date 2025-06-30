import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from transformers import AutoProcessor, AutoModelForVision2Seq
import torch

# === Load model and processor ===
model_id = "HuggingFaceTB/SmolVLM-256M-Instruct"
processor = AutoProcessor.from_pretrained(model_id)
model = AutoModelForVision2Seq.from_pretrained(model_id).eval()
vision_encoder = model.model.vision_model

# === Set image directory and patch info ===
image_dir = "exported_elements_dataset/000"
image_filenames = sorted(os.listdir(image_dir))[:2]  # first 2 images
patch_size = model.config.vision_config.patch_size  # Should be 16
hidden_size = model.config.vision_config.hidden_size  # Should be 768

# === Font for patch number overlay ===
try:
    font = ImageFont.truetype("arial.ttf", 10)
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

# === Process and visualize each image ===
fig, axs = plt.subplots(len(image_filenames), 2, figsize=(10, 3.5 * len(image_filenames)))

for i, filename in enumerate(image_filenames):
    # --- Load and preprocess image ---
    img_path = os.path.join(image_dir, filename)
    orig_img = Image.open(img_path).convert("RGB").resize((512, 512))  # Resized to match model input
    processed = processor(images=orig_img, return_tensors="pt")
    img_tensor = processed["pixel_values"][:, 0]  # first slot only → shape: [1, 3, 512, 512]

    # --- Run through vision encoder ---
    with torch.no_grad():
        vision_out = vision_encoder(img_tensor)
    tokens = vision_out.last_hidden_state.squeeze(0).cpu().numpy()  # [num_patches, hidden_size]

    # --- PCA for token projection ---
    pca = PCA(n_components=2)
    proj = pca.fit_transform(tokens)
    patch_indices = np.arange(tokens.shape[0])

    # === 1. Show original image with patch numbers ===
    patch_img = overlay_patch_indices(orig_img.copy(), patch_size)
    axs[i, 0].imshow(patch_img)
    axs[i, 0].set_title(f"Image {i+1}: Patch Index Overlay")
    axs[i, 0].axis("off")

    # === 2. Show PCA of token embeddings colored by patch index ===
    sc = axs[i, 1].scatter(proj[:, 0], proj[:, 1], c=patch_indices, cmap="plasma")
    axs[i, 1].set_title(f"Image {i+1}: PCA of Tokens (Patch Coloring)")
    axs[i, 1].set_xlabel("PC1")
    axs[i, 1].set_ylabel("PC2")
    axs[i, 1].grid(True)
    fig.colorbar(sc, ax=axs[i, 1], label="Patch Index")

plt.tight_layout()
plt.show()
