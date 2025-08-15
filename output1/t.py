import torch
from transformers import AutoModelForImageClassification, AutoImageProcessor
from PIL import Image
import requests

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Selected device:", device)

model_name = "google/vit-base-patch32-384"
model = AutoModelForImageClassification.from_pretrained(model_name).to(device).eval()
processor = AutoImageProcessor.from_pretrained(model_name)

img = Image.new("RGB", (384, 384), color=(255, 255, 255))  # Dummy-Bild
inputs = processor(images=img, return_tensors="pt")
inputs = {k: v.to(device) for k, v in inputs.items()}

print("Model device:", next(model.parameters()).device)
print("Input device:", inputs["pixel_values"].device)

import time
torch.cuda.synchronize()
start = time.perf_counter()

with torch.no_grad():
    logits = model(**inputs).logits[:, 0]

torch.cuda.synchronize()
end = time.perf_counter()
print(f"Inference Time: {end - start:.4f} s")
