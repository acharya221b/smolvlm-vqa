
# smolvlm_elements_eval_tcav.py

from transformers import AutoProcessor, AutoModelForVision2Seq, BitsAndBytesConfig
from PIL import Image
from pathlib import Path
import torch
import pandas as pd
from tqdm import tqdm
import numpy as np
import csv
from sklearn.metrics import accuracy_score

# --- Config ---
MODEL_ID = "HuggingFaceTB/SmolVLM-Synthetic"
IMAGE_ROOT = Path("exported_elements_dataset")
LABELS_CSV = IMAGE_ROOT / "labels.csv"
OUTPUT_CSV = "vlm_tcav_results.csv"
TOKEN_DUMP = "visual_tokens.npy"
N_SAMPLES = 10
PROMPT = "<image> Describe the number of objects, their shape and colour in the image. Also mention the pattern on each of the objects."

# --- Model loading ---
quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
processor = AutoProcessor.from_pretrained(MODEL_ID)
model = AutoModelForVision2Seq.from_pretrained(
    MODEL_ID,
    quantization_config=quant_config,
    device_map="auto"
)
vision_encoder = getattr(model.model, "vision_model", None)
if vision_encoder is None:
    raise AttributeError("No vision encoder found.")
model.eval()

# --- Load labels and subset ---
df = pd.read_csv(LABELS_CSV).head(N_SAMPLES)

# --- Inference & token extraction ---
records = []
visual_tokens = []

for idx, row in tqdm(df.iterrows(), total=len(df)):
    image_path = Path(row["image_path"])
    image = Image.open(image_path).convert("RGB")

    # Generate output
    inputs = processor(images=[image], text=[PROMPT], return_tensors="pt").to(device)
    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=64)
        response = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()

    # Extract visual tokens
    pixel_values = processor(images=image, return_tensors="pt")["pixel_values"].to(device).to(torch.float16)
    single_img_tensor = pixel_values[0, 0].unsqueeze(0)  # Extract only the first image slot
    with torch.no_grad():
        vision_outputs = vision_encoder(single_img_tensor)
    token_out = vision_outputs.last_hidden_state.squeeze(0).cpu().numpy()
    visual_tokens.append(token_out)

    # Check if predicted text contains ground truth shape/texture/color
    gt_shape = str(row["shape"]).lower()
    gt_color = str(row["color"]).lower()
    gt_texture = str(row["texture"]).lower()
    pred_match = all(x in response.lower() for x in [gt_shape, gt_color, gt_texture])

    records.append({
        "image_path": str(image_path),
        "prompt": PROMPT,
        "model_output": response,
        "label_color": gt_color,
        "label_shape": gt_shape,
        "label_texture": gt_texture,
        "match_all_concepts": int(pred_match)
    })

# --- Save predictions ---
pd.DataFrame(records).to_csv(OUTPUT_CSV, index=False)

# --- Save tokens for TCAV ---
visual_tokens = np.array(visual_tokens)  # shape [N, 64, hidden_dim]
np.save(TOKEN_DUMP, visual_tokens)

# --- Print quick stats ---
matches = [r["match_all_concepts"] for r in records]
acc = accuracy_score([1]*len(matches), matches)
print(f"Concept-level matching accuracy: {acc:.2f}")
print(f"Saved visual tokens to: {TOKEN_DUMP}")
print(f"Saved model outputs to: {OUTPUT_CSV}")
