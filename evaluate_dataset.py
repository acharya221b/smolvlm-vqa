from datasets import load_dataset, get_dataset_split_names
import matplotlib.pyplot as plt
from transformers import AutoProcessor, AutoModelForImageTextToText
from PIL import Image
import torch
from tqdm import tqdm
import evaluate  # HuggingFace Evaluate library for BLEU
import re
import pandas as pd

# Setup
print("CUDA available:", torch.cuda.is_available())
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# dataset = load_dataset("SimulaMet-HOST/Kvasir-VQA", split="raw[:5]")
#
# for i, sample in enumerate(dataset):
#     image = sample["image"]
#     question = sample["question"]
#     answer = sample["answer"]
#
#     plt.imshow(image)
#     plt.title(f"Q: {question}\nA: {answer}")
#     plt.axis("off")
#     plt.show()

# Load dataset
dataset = load_dataset("SimulaMet-HOST/Kvasir-VQA", split="raw")

# Select a smaller subset for faster evaluation (change N as needed)
N = 100  # Number of samples to evaluate on
dataset = dataset.select(range(N))  # Comment this line to run on full dataset

# Load model & processor
processor = AutoProcessor.from_pretrained("HuggingFaceTB/SmolVLM-256M-Instruct")
model = AutoModelForImageTextToText.from_pretrained("HuggingFaceTB/SmolVLM-256M-Instruct").to(device).eval()

# Metrics
bleu = evaluate.load("bleu")
total, exact_matches = 0, 0
correct_count = 0
predictions = []
references = []

# Normalization function
def normalize_answer(text):
    text = text.lower().strip()
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    return text

# Evaluation loop
print(f"Running evaluation on {len(dataset)} samples...")

for sample in tqdm(dataset):
    try:
        image = sample["image"]
        question = "<image> " + sample["question"]
        gt_answer = sample["answer"]

        # Process & run model
        inputs = processor(images=[image], text=[question], return_tensors="pt").to(device)
        outputs = model.generate(**inputs)
        pred = processor.decode(outputs[0], skip_special_tokens=True)

        # Save for metrics
        predictions.append(pred)
        references.append([gt_answer])

        # Compute exact match
        if normalize_answer(pred) == normalize_answer(gt_answer):
            exact_matches += 1

        # Compute accuracy for yes/no
        if gt_answer.lower() in ['yes', 'no']:
            if normalize_answer(pred) == normalize_answer(gt_answer):
                correct_count += 1

        total += 1

    except Exception as e:
        print(f"Error processing sample: {e}")

# Compute BLEU
bleu_result = bleu.compute(predictions=predictions, references=references)

# Print results
print("\n=== Evaluation Results ===")
print(f"Total Samples: {total}")
print(f"Exact Match (EM): {exact_matches / total:.2%}")
print(f"Yes/No Accuracy: {correct_count / total:.2%}")
print(f"BLEU Score: {bleu_result['bleu']:.4f}")

pd.DataFrame({
    "question": [s["question"] for s in dataset],
    "ground_truth": [s["answer"] for s in dataset],
    "prediction": predictions
}).to_csv("smolvlm_predictions.csv", index=False)
