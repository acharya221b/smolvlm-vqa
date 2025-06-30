from transformers import AutoProcessor, AutoModelForVision2Seq, BitsAndBytesConfig
from PIL import Image
from pathlib import Path
import torch
import csv
import pandas as pd
from tqdm import tqdm

# Configuration
MODEL_ID = "HuggingFaceTB/SmolVLM-Base"
IMAGE_ROOT = Path("exported_elements_dataset")  # Your exported dataset path
LABELS_CSV = IMAGE_ROOT / "labels.csv"
OUTPUT_CSV = "vlm_outputs_Base.csv"

torch.cuda.empty_cache()

# ---------------- LOAD MODEL ----------------
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
).to(device)

# Load dataset labels
df = pd.read_csv(LABELS_CSV)

# Select only the first 10 samples
df = df.head(10)

# Run inference and collect outputs
rows = []
for idx, row in tqdm(df.iterrows(), total=len(df)):
    image_path = Path(row["image_path"])
    image = Image.open(image_path).convert("RGB")

    # Provide dummy prompt with <image> token to allow image embedding injection
    dummy_prompt = "<image>"

    # Process image + dummy prompt
    inputs = processor(images=[image], text=[dummy_prompt], return_tensors="pt")

    # Filter out only relevant inputs (no extra keys like 'rows', 'cols')
    inputs = {
        "input_ids": inputs["input_ids"].to(device),
        "pixel_values": inputs["pixel_values"].to(device),
        "attention_mask": inputs["attention_mask"].to(device),
    }

    # Generate output
    generated_ids = model.generate(
        **inputs,
        max_new_tokens=64,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
    )
    response = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()


    print(f"{idx+1:02d}: {response}")

    rows.append({
        "image_path": str(image_path),
        "model_output": response,
        "label_color": row["color"],
        "label_shape": row["shape"],
        "label_texture": row["texture"]
    })

# Save outputs
with open(OUTPUT_CSV, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)

print(f"Saved model outputs to: {OUTPUT_CSV}")
