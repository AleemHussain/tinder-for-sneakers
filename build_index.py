from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import torch
import os
import numpy as np
from annoy import AnnoyIndex
from db import init_db, insert_sneaker

# Load FashionCLIP
model = CLIPModel.from_pretrained("patrickjohncyh/fashion-clip")
processor = CLIPProcessor.from_pretrained("patrickjohncyh/fashion-clip")
model.eval()

# Prepare image directory and Annoy index
image_dir = "shoes"
embedding_dim = 512
index = AnnoyIndex(embedding_dim, "angular")
id_to_filename = []

valid_ext = [".jpg", ".jpeg", ".png"]

# Initialize database
init_db()

# Process images: embed, index, and insert metadata into database
for idx, filename in enumerate(os.listdir(image_dir)):
    if not any(filename.lower().endswith(ext) for ext in valid_ext):
        continue
    image_path = os.path.join(image_dir, filename)
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")

    with torch.no_grad():
        embedding = model.get_image_features(**inputs)[0].cpu().numpy()

    index.add_item(idx, embedding)
    id_to_filename.append(filename)

    # Prepare metadata (placeholder)
    brand = "unknown"
    color = "unknown"
    description = filename.replace("_", " ").split(".")[0]

    # Insert into database
    insert_sneaker(filename, brand, color, description, idx)

# Save Annoy index and filename mapping
index.build(10)
index.save("shoe_index.ann")
np.save("id_to_filename.npy", id_to_filename)