import open_clip
from PIL import Image
import torch
import os
import numpy as np
from annoy import AnnoyIndex

# Load OpenCLIP model and preprocessing pipeline
model, _, preprocess = open_clip.create_model_and_transforms(
    model_name="ViT-B-32",
    pretrained="laion2b_s34b_b79k"
)
model.eval()

# Image directory and index setup
image_dir = "shoes"
embedding_dim = 512
index = AnnoyIndex(embedding_dim, "angular")
id_to_filename = []

# Only process valid image files
valid_ext = [".jpg", ".jpeg", ".png"]

for idx, filename in enumerate(os.listdir(image_dir)):
    if not any(filename.lower().endswith(ext) for ext in valid_ext):
        continue
    image_path = os.path.join(image_dir, filename)
    image = preprocess(Image.open(image_path).convert("RGB")).unsqueeze(0)
    with torch.no_grad():
        embedding = model.encode_image(image).cpu().numpy()[0]
    index.add_item(idx, embedding)
    id_to_filename.append(filename)

# Build and save the Annoy index
index.build(10)
index.save("shoe_index.ann")
np.save("id_to_filename.npy", id_to_filename)

print("✅ Index successfully built.")
